"""Canonical ordered stage catalog for end-to-end analysis runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from utils.workspace import repository_root


@dataclass(frozen=True)
class PipelineStage:
    stage_id: str
    script_path: Path
    title: str
    description: str


def build_pipeline_catalog() -> list[PipelineStage]:
    """Return numbered stages in execution order."""
    root = repository_root()
    src = root / "src"
    rows = [
        ("00a", src / "00_filtering.py", "Layer 0/1 Filtering", "Split posts into poems and stanza rows."),
        ("00b", src / "00_gpt_human_review_batch.py", "Human Review Batch", "Batch adjudication for uncertain split rows."),
        ("01", src / "01_pronoun_detection.py", "Pronoun Detection", "Detect explicit and dropped pronouns."),
        ("02", src / "02_pronoun_cooccurrence.py", "Co-occurrence Network", "Build high-frequency pronoun-word network."),
        ("03", src / "03_pronoun_semantic_space.py", "Semantic Space", "FastText + UMAP clustering and drift."),
        ("04", src / "04_annotation_toolkit.py", "Annotation Toolkit", "Sampling, QA, and annotation prep utilities."),
        ("05", src / "05_annotate_pronouns.py", "Rule Annotation", "Heuristic pilot labeling on sampled pronouns."),
        ("06", src / "06_pronoun_projection.py", "Cross-lingual Projection", "Projection-based pronoun alignment."),
        ("07", src / "07_gpt_annotation.py", "GPT Annotation", "Stanza-level GPT pronoun annotation."),
        ("08", src / "08_adaptive_binning.py", "Adaptive Binning", "Temporal adaptive interval construction."),
        ("09", src / "09_breakpoint_regression.py", "Breakpoint Regression", "Interrupted trend and changepoint analysis."),
        ("10", src / "10_public_list_filter.py", "Public-list Filter", "Filter public corpus and derivative files."),
        ("11a", src / "11_layer0_layer1_to_run_filter.py", "To-run Filter", "Trim To_run layer0/layer1 with public-list rules."),
        ("11b", src / "11_gpt_annotate_full.py", "Full GPT Runner", "Wrapper for full/public GPT annotation runs."),
        ("12", src / "12_gpt_annotation_exploration.py", "Exploration", "Initial QA and distribution diagnostics."),
        ("13", src / "13_descriptive_statistics.py", "Descriptive Statistics", "Methodology and corpus overview tables."),
        ("14", src / "14_rq1_marginal_distributions.py", "RQ1 Marginals", "Core marginal distributions by period."),
        ("15", src / "15_key_cross_distributions.py", "RQ2 Cross Distributions", "Key cross-tabs for interpretable contrasts."),
        ("16", src / "16_author_heterogeneity.py", "RQ3 Heterogeneity", "Author-level robustness and composition checks."),
        ("17", src / "17_poem_level_metrics.py", "Poem Metrics", "Poem-level feature engineering and summaries."),
        ("18", src / "18_publication_priority_figures.py", "Publication Figures", "Priority publication figures P1-P4."),
    ]
    return [
        PipelineStage(
            stage_id=stage_id,
            script_path=script_path,
            title=title,
            description=description,
        )
        for stage_id, script_path, title, description in rows
    ]


def stage_by_id(stage_id: str) -> PipelineStage | None:
    """Lookup a single stage by exact ID."""
    wanted = stage_id.strip().lower()
    for stage in build_pipeline_catalog():
        if stage.stage_id.lower() == wanted:
            return stage
    return None
