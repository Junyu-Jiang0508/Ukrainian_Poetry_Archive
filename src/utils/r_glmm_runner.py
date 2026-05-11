"""Bridge to R / lme4 for binomial generalized linear mixed-effects models.

The Python frequentist stack does not include a production-quality binomial GLMM
that matches ``lme4::glmer`` for inference on closed-cell attention-allocation
designs (incidental-parameter problems make unconditional MLE with an author
dummy unreliable when each author contributes only ~20 poems). This module
calls ``Rscript`` on a generated script, exchanges data through CSVs in a
temporary directory, and returns a lightweight result object exposing
``params`` (pandas Series indexed by R-style term names) and
``cov_params()`` (pandas DataFrame).

Failure modes are explicit: missing ``Rscript`` or missing ``lme4`` raise
:class:`RGlmmEnvironmentError` with an actionable message; convergence
problems are surfaced in :class:`RGlmmFitResult.convergence_message`.
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


class RGlmmEnvironmentError(RuntimeError):
    """Raised when Rscript or required R packages are unavailable."""


class RGlmmFitError(RuntimeError):
    """Raised when glmer fails (separation, non-positive-definite, etc.)."""


@dataclass(frozen=True)
class RGlmmFitResult:
    params: pd.Series
    cov: pd.DataFrame
    n_obs: int
    n_authors: int
    log_likelihood: float
    aic: float
    bic: float
    random_intercept_sd_author: float
    convergence_message: str
    optimizer: str
    formula: str
    family: str
    link: str

    def cov_params(self) -> pd.DataFrame:
        return self.cov


_R_SCRIPT_TEMPLATE = r"""
suppressPackageStartupMessages({
  if (!requireNamespace("lme4", quietly = TRUE)) {
    cat("MISSING_LME4\n", file = stderr())
    quit(status = 3)
  }
  library(lme4)
})

args <- commandArgs(trailingOnly = TRUE)
input_csv <- args[1]
out_dir <- args[2]
formula_str <- args[3]
optimizer <- args[4]

d <- read.csv(input_csv, stringsAsFactors = FALSE)
d$author <- factor(d$author)
d$period3 <- factor(d$period3, levels = c("P1_2014_2021", "P2_2022_plus"))

fit <- tryCatch(
  glmer(as.formula(formula_str),
        data = d, family = binomial(link = "logit"),
        control = glmerControl(optimizer = optimizer,
                               optCtrl = list(maxfun = 2e5))),
  error = function(e) {
    msg <- conditionMessage(e)
    write(paste0("GLMER_ERROR: ", msg), file = file.path(out_dir, "error.txt"))
    quit(status = 2)
  }
)

fe <- fixef(fit)
vc <- as.matrix(vcov(fit))
coef_df <- data.frame(term = names(fe),
                      estimate = unname(fe),
                      se = sqrt(diag(vc)))
write.csv(coef_df, file = file.path(out_dir, "fixed_effects.csv"), row.names = FALSE)
vc_df <- as.data.frame(vc)
vc_df <- cbind(term = rownames(vc), vc_df)
write.csv(vc_df, file = file.path(out_dir, "vcov.csv"), row.names = FALSE)

conv <- ""
if (!is.null(fit@optinfo$conv$lme4$messages)) {
  conv <- paste(fit@optinfo$conv$lme4$messages, collapse = " | ")
}
vc_author <- VarCorr(fit)$author
sigma_a <- if (is.null(vc_author)) NA_real_ else attr(vc_author, "stddev")[1]

meta <- list(
  n_obs = nobs(fit),
  n_authors = length(unique(d$author)),
  log_likelihood = as.numeric(logLik(fit)),
  AIC = AIC(fit),
  BIC = BIC(fit),
  random_intercept_sd_author = as.numeric(sigma_a),
  convergence_message = conv,
  family = "binomial",
  link = "logit",
  optimizer = optimizer,
  formula = formula_str
)
writeLines(jsonlite_or_naive(meta), file.path(out_dir, "meta.json"))
"""


# We define ``jsonlite_or_naive`` lazily inside the R script so that the same
# template can run on systems without jsonlite. The block below is appended to
# the script before execution so the function is in scope before ``writeLines``.
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
        raise RGlmmEnvironmentError(
            "Rscript not found on PATH. Install R (e.g. `apt-get install r-base`) "
            "and the `lme4` R package, or pip install rpy2 if you prefer an "
            "in-process bridge."
        )
    return rscript


def fit_glmer_binomial(
    long_df: pd.DataFrame,
    formula: str,
    *,
    optimizer: str = "bobyqa",
    workdir: Path | None = None,
) -> RGlmmFitResult:
    """Fit a binomial GLMM via R / lme4 ``glmer`` on a long table.

    Parameters
    ----------
    long_df:
        Must contain columns ``k`` (successes), ``n`` (trials), ``author``,
        ``period3`` (with levels ``P1_2014_2021`` and ``P2_2022_plus``), and any
        other fixed-effect covariates referenced by ``formula`` such as
        ``person`` and ``number``.
    formula:
        R-style formula string, for example
        ``cbind(k, n - k) ~ person * number * period3 + (1 | author)``.
    optimizer:
        ``glmerControl`` optimizer (default ``bobyqa``). ``Nelder_Mead`` is a
        useful fallback when bobyqa stalls near a singular fit.
    """
    rscript = _check_rscript()

    needed = {"k", "n", "author", "period3"}
    missing = sorted(needed - set(long_df.columns))
    if missing:
        raise ValueError(f"fit_glmer_binomial: missing columns {missing}")

    tmpdir = Path(tempfile.mkdtemp(prefix="r_glmm_")) if workdir is None else Path(workdir)
    tmpdir.mkdir(parents=True, exist_ok=True)

    csv_path = tmpdir / "input.csv"
    script_path = tmpdir / "fit.R"
    long_df.to_csv(csv_path, index=False)

    script_body = _R_JSON_HELPER + "\n" + _R_SCRIPT_TEMPLATE
    script_path.write_text(script_body, encoding="utf-8")

    cmd = [rscript, "--vanilla", str(script_path), str(csv_path), str(tmpdir), formula, optimizer]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode == 3:
        raise RGlmmEnvironmentError(
            "R package `lme4` is not installed. Install it via "
            "`Rscript -e 'install.packages(\"lme4\", repos=\"https://packagemanager.posit.co/cran/__linux__/noble/latest\")'` "
            "(adjust the binary repo for your distribution) or "
            "`install.packages(\"lme4\")` from CRAN with cmake/NLopt available."
        )
    error_file = tmpdir / "error.txt"
    if proc.returncode != 0:
        err_msg = error_file.read_text() if error_file.is_file() else (proc.stderr or "<no stderr>")
        raise RGlmmFitError(
            f"glmer failed (returncode={proc.returncode}). Stderr:\n{proc.stderr}\n"
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
        raise RGlmmFitError(f"Could not parse meta.json: {exc}\nContents:\n{meta_raw}") from exc

    return RGlmmFitResult(
        params=params,
        cov=vcov_df.astype(float),
        n_obs=int(meta.get("n_obs", 0)),
        n_authors=int(meta.get("n_authors", 0)),
        log_likelihood=float(meta.get("log_likelihood", np.nan)),
        aic=float(meta.get("AIC", np.nan)),
        bic=float(meta.get("BIC", np.nan)),
        random_intercept_sd_author=float(meta.get("random_intercept_sd_author", np.nan)),
        convergence_message=str(meta.get("convergence_message", "") or ""),
        optimizer=str(meta.get("optimizer", optimizer)),
        formula=str(meta.get("formula", formula)),
        family=str(meta.get("family", "binomial")),
        link=str(meta.get("link", "logit")),
    )
