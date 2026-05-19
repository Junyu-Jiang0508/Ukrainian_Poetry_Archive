"""Bridge to R / ``survival::clogit`` for Chamberlain conditional logistic regression.

The Python ``statsmodels.discrete.conditional_models.ConditionalLogit`` backend
is unsuitable for author-stratified Bernoulli trials at this corpus's scale: it
uses a Python-recursive partition function whose recursion depth is linear in
stratum size, and reliably crashes with ``RecursionError`` on author strata
containing the ~1000 trials seen in the four-cell expansion (even after raising
``sys.setrecursionlimit`` to 50 000). ``survival::clogit`` runs the
Chamberlain-1980 conditional MLE in C via the ``coxph`` partial-likelihood
machinery and handles strata of arbitrary size without recursion.

This module mirrors :mod:`utils.r_glmm_runner`: an ``Rscript`` subprocess
exchanges data through CSVs in a temporary directory and returns a result
object exposing ``params`` (pandas Series indexed by R-style term names) and
``cov_params()`` (pandas DataFrame).

Failure modes are explicit:

* Missing ``Rscript`` raises :class:`RClogitEnvironmentError`.
* Missing ``survival`` package raises :class:`RClogitEnvironmentError` with an
  installation hint.
* Convergence problems are surfaced in
  :class:`RClogitFitResult.convergence_message`.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class RClogitEnvironmentError(RuntimeError):
    """Raised when Rscript or the ``survival`` R package is unavailable."""


class RClogitFitError(RuntimeError):
    """Raised when ``clogit`` itself fails (separation, singular design, ...)."""


@dataclass(frozen=True)
class RClogitFitResult:
    """Lightweight result object mirroring ``statsmodels`` discrete-model returns."""

    params: pd.Series
    cov: pd.DataFrame
    n_obs: int
    n_strata: int
    n_strata_informative: int
    log_likelihood: float
    convergence_message: str
    formula: str

    def cov_params(self) -> pd.DataFrame:
        return self.cov


# R script: parse args, read the trial-level CSV, fit clogit, write coef/vcov/meta.
# ``y`` is binary (0/1), ``stratum`` is the author label. ``predictors_formula``
# is the right-hand side passed in from Python without ``strata(...)`` — we
# append the strata term R-side so the Python formula constant stays free of
# the R-specific ``strata()`` token.
_R_SCRIPT_TEMPLATE = r"""
suppressPackageStartupMessages({
  if (!requireNamespace("survival", quietly = TRUE)) {
    cat("MISSING_SURVIVAL\n", file = stderr())
    quit(status = 3)
  }
  library(survival)
})

args <- commandArgs(trailingOnly = TRUE)
input_csv <- args[1]
out_dir <- args[2]
predictors_formula <- args[3]

d <- read.csv(input_csv, stringsAsFactors = FALSE)
d$author <- factor(d$author)
d$period3 <- factor(d$period3, levels = c("P1_2014_2021", "P2_2022_plus"))
d$y <- as.integer(d$y)
d$person <- as.integer(d$person)
d$number <- as.integer(d$number)

full_formula_str <- paste0("y ~ ", predictors_formula, " + strata(author)")
full_formula <- as.formula(full_formula_str)

fit <- tryCatch(
  clogit(full_formula, data = d, method = "exact"),
  error = function(e) {
    msg <- conditionMessage(e)
    write(paste0("CLOGIT_ERROR: ", msg), file = file.path(out_dir, "error.txt"))
    quit(status = 2)
  }
)

# Coefficients and variance-covariance.
coef_vec <- coef(fit)
vc <- as.matrix(vcov(fit))
# Align names: vcov rows/cols and coef names should match in survival::clogit.
coef_df <- data.frame(term = names(coef_vec),
                      estimate = unname(coef_vec),
                      se = sqrt(diag(vc)))
write.csv(coef_df, file = file.path(out_dir, "fixed_effects.csv"), row.names = FALSE)
vc_df <- as.data.frame(vc)
vc_df <- cbind(term = rownames(vc), vc_df)
write.csv(vc_df, file = file.path(out_dir, "vcov.csv"), row.names = FALSE)

# Strata diagnostics: clogit drops strata with no within-stratum variation in y.
# We compute the count of "informative" strata (those that contributed to the
# conditional likelihood) and total strata in the input.
strata_total <- length(unique(d$author))
informative <- sum(
  vapply(split(d$y, d$author),
         function(v) length(unique(v)) > 1L,
         logical(1))
)

# Convergence diagnostics from coxph internals.
conv_msg <- ""
if (!is.null(fit$info) && !is.null(fit$info$conv)) {
  conv_msg <- as.character(fit$info$conv)
}
# clogit / coxph stores number of iterations and the score residual norm; if
# the iteration count maxed out we surface that explicitly.
iter_used <- if (!is.null(fit$iter)) fit$iter else NA_integer_
if (!is.na(iter_used) && iter_used >= 25) {
  conv_msg <- paste0("max_iterations_reached=", iter_used, "; ", conv_msg)
}

meta <- list(
  n_obs = nrow(d),
  n_strata = strata_total,
  n_strata_informative = informative,
  log_likelihood = as.numeric(fit$loglik[length(fit$loglik)]),
  convergence_message = conv_msg,
  formula = full_formula_str
)
writeLines(jsonlite_or_naive(meta), file.path(out_dir, "meta.json"))
"""


# Minimal hand-rolled JSON emitter so we do not depend on the optional
# ``jsonlite`` R package. Matches the helper used by r_glmm_runner.
_R_JSON_HELPER = r"""
jsonlite_or_naive <- function(x) {
  parts <- character(0)
  for (nm in names(x)) {
    v <- x[[nm]]
    if (is.character(v)) {
      v_str <- paste0('"', gsub('\\\\', '\\\\\\\\', gsub('"', '\\\\"', v)), '"')
    } else if (length(v) == 0 || (length(v) == 1 && is.na(v))) {
      v_str <- "null"
    } else {
      v_str <- as.character(v)
    }
    parts <- c(parts, paste0('"', nm, '": ', v_str))
  }
  paste0("{", paste(parts, collapse = ", "), "}")
}
"""


def _check_rscript() -> str:
    rscript = shutil.which("Rscript")
    if not rscript:
        raise RClogitEnvironmentError(
            "Rscript not found on PATH. Install R (e.g. `apt-get install r-base`) "
            "and the `survival` R package (ships with base R)."
        )
    return rscript


def fit_clogit(
    trial_df: pd.DataFrame,
    predictors_formula: str,
    *,
    workdir: Path | None = None,
) -> RClogitFitResult:
    """Fit Chamberlain conditional logistic regression on trial-level data.

    Parameters
    ----------
    trial_df:
        One row per Bernoulli trial. Required columns: ``y`` (0/1 outcome),
        ``author`` (stratum), ``period3`` (factor with the two reference levels
        ``P1_2014_2021`` and ``P2_2022_plus``), ``person`` (integer-coded), and
        ``number`` (integer-coded). Any extra columns are ignored R-side.
    predictors_formula:
        Right-hand side of the ``y ~ ...`` formula, without the ``strata(author)``
        term. The runner appends ``+ strata(author)`` before calling ``clogit``.
    workdir:
        Optional directory for the R subprocess scratch files. A fresh
        ``tempfile.mkdtemp`` is used when ``None``.
    """
    rscript = _check_rscript()

    needed = {"y", "author", "period3", "person", "number"}
    missing = sorted(needed - set(trial_df.columns))
    if missing:
        raise ValueError(f"fit_clogit: missing columns {missing}")
    if not len(trial_df):
        raise ValueError("fit_clogit: trial_df is empty")

    tmpdir = Path(tempfile.mkdtemp(prefix="r_clogit_")) if workdir is None else Path(workdir)
    tmpdir.mkdir(parents=True, exist_ok=True)

    csv_path = tmpdir / "input.csv"
    script_path = tmpdir / "fit.R"
    trial_df[sorted(needed)].to_csv(csv_path, index=False)

    script_body = _R_JSON_HELPER + "\n" + _R_SCRIPT_TEMPLATE
    script_path.write_text(script_body, encoding="utf-8")

    cmd = [rscript, "--vanilla", str(script_path), str(csv_path), str(tmpdir), predictors_formula]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.returncode == 3:
        raise RClogitEnvironmentError(
            "R package `survival` is missing. It ships with base R; install "
            "via `Rscript -e 'install.packages(\"survival\")'` if your "
            "distribution stripped it."
        )

    error_file = tmpdir / "error.txt"
    if proc.returncode != 0:
        err_msg = error_file.read_text() if error_file.is_file() else (proc.stderr or "<no stderr>")
        raise RClogitFitError(
            f"clogit failed (returncode={proc.returncode}). Stderr:\n{proc.stderr}\n"
            f"Error file:\n{err_msg}"
        )

    fe_df = pd.read_csv(tmpdir / "fixed_effects.csv")
    fe_df.set_index("term", inplace=True)
    params = fe_df["estimate"].astype(float)

    vcov_df = pd.read_csv(tmpdir / "vcov.csv")
    vcov_df.set_index("term", inplace=True)
    vcov_df = vcov_df.reindex(index=params.index, columns=params.index)

    meta_raw = (tmpdir / "meta.json").read_text(encoding="utf-8")
    try:
        meta = json.loads(meta_raw)
    except json.JSONDecodeError as exc:
        raise RClogitFitError(f"Could not parse meta.json: {exc}\nContents:\n{meta_raw}") from exc

    return RClogitFitResult(
        params=params,
        cov=vcov_df.astype(float),
        n_obs=int(meta.get("n_obs", 0)),
        n_strata=int(meta.get("n_strata", 0)),
        n_strata_informative=int(meta.get("n_strata_informative", 0)),
        log_likelihood=float(meta.get("log_likelihood", np.nan)),
        convergence_message=str(meta.get("convergence_message", "") or ""),
        formula=str(meta.get("formula", predictors_formula)),
    )
