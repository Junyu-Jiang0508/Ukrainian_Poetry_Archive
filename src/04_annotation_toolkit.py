"""Pronoun annotation helpers: sampling, IAA, XLM-R prep, reports."""

import json
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


class AnnotationToolkit:
    def __init__(self, pronoun_file: str):
        self.df = pd.read_csv(pronoun_file)
        self.annotation_memory = {
            "canonical_examples": {},
            "edge_cases": {},
            "team_decisions": {},
        }

    def stratified_sample(
        self,
        n_tokens: int = 500,
        time_col: str = "year",
        time_bins: Optional[List[Tuple[int, int]]] = None,
    ) -> pd.DataFrame:
        if time_bins is None:
            time_bins = [(2014, 2015), (2016, 2021), (2022, 2023), (2024, 2025)]

        df_ua = self.df[self.df["Language"] == "UA"].copy()

        df_1pl = df_ua[(df_ua["person"] == "1") & (df_ua["number"] == "Plur")]
        df_3pl = df_ua[(df_ua["person"] == "3") & (df_ua["number"] == "Plur")]

        n_1pl = int(n_tokens * 0.6)
        n_3pl = int(n_tokens * 0.4)

        sampled_frames = []

        for df_subset, n_target in [(df_1pl, n_1pl), (df_3pl, n_3pl)]:
            for start, end in time_bins:
                df_period = df_subset[
                    (df_subset[time_col] >= start) & (df_subset[time_col] <= end)
                ]
                n_period = int(n_target / len(time_bins))
                if len(df_period) >= n_period:
                    sample = df_period.sample(n=n_period, random_state=42)
                else:
                    sample = df_period
                sampled_frames.append(sample)

        result = pd.concat(sampled_frames, ignore_index=True)
        return result.sample(frac=1, random_state=42).reset_index(drop=True)

    def add_annotation_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        annotation_cols = {
            "referent_category": "",
            "referent_confidence": np.nan,
            "we_inclusivity": "",
            "we_incl_confidence": np.nan,
            "syntactic_position": "",
            "semantic_role": "",
            "discourse_function": "",
            "annotation_difficulty": "",
            "token_notes": "",
        }

        for col, default in annotation_cols.items():
            if col not in df.columns:
                df[col] = default

        return df

    def extract_context_windows(
        self,
        df: pd.DataFrame,
        text_col: str = "text",
        context_size: int = 2,
    ) -> pd.DataFrame:
        for i in range(-context_size, context_size + 1):
            df[f"context_{i:+d}"] = ""

        for idx, row in df.iterrows():
            if pd.isna(row[text_col]):
                continue

            sentences = row[text_col].split(".")
            sentences = [s.strip() for s in sentences if s.strip()]

            if "context" not in row or pd.isna(row["context"]):
                continue

            current_sent = row["context"].strip()

            try:
                sent_idx = sentences.index(current_sent)
            except ValueError:
                sent_idx = None
                for j, sent in enumerate(sentences):
                    if current_sent in sent or sent in current_sent:
                        sent_idx = j
                        break
                if sent_idx is None:
                    continue

            for i in range(-context_size, context_size + 1):
                target_idx = sent_idx + i
                if 0 <= target_idx < len(sentences):
                    df.at[idx, f"context_{i:+d}"] = sentences[target_idx]

        return df

    def compute_inter_annotator_agreement(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        columns: List[str],
    ) -> Dict[str, float]:
        try:
            from krippendorff import alpha
        except ImportError:
            print("pip install krippendorff")
            return {}

        results = {}

        for col in columns:
            if col not in df1.columns or col not in df2.columns:
                continue

            reliability_data = [df1[col].tolist(), df2[col].tolist()]

            try:
                alpha_value = alpha(
                    reliability_data=reliability_data, level_of_measurement="nominal"
                )
                results[col] = alpha_value
            except Exception as e:
                print(f"krippendorff {col}: {e}")
                results[col] = np.nan

        return results

    def consistency_check(self, df: pd.DataFrame) -> pd.DataFrame:
        issues = []

        for poem_id, group in df.groupby("ID"):
            if len(group) < 2:
                continue

            same_context = group[group.duplicated(subset=["context"], keep=False)]

            if len(same_context) > 0:
                for idx, row1 in same_context.iterrows():
                    for idx2, row2 in same_context.iterrows():
                        if idx >= idx2:
                            continue
                        if row1["context"] == row2["context"]:
                            if row1["referent_category"] != row2["referent_category"]:
                                issues.append(
                                    {
                                        "poem_id": poem_id,
                                        "token1_id": row1["token_id"],
                                        "token2_id": row2["token_id"],
                                        "context": row1["context"],
                                        "label1": row1["referent_category"],
                                        "label2": row2["referent_category"],
                                        "issue": "same_context_different_label",
                                    }
                                )

        uncertain_high_conf = df[
            (df["referent_category"] == "UNCERTAIN")
            & (df["referent_confidence"] == 3)
        ]
        for idx, row in uncertain_high_conf.iterrows():
            issues.append(
                {
                    "poem_id": row["ID"],
                    "token_id": row["token_id"],
                    "context": row["context"],
                    "label": row["referent_category"],
                    "confidence": row["referent_confidence"],
                    "issue": "uncertain_with_high_confidence",
                }
            )

        return pd.DataFrame(issues)

    def prepare_for_xlm_roberta(
        self, df: pd.DataFrame, output_dir: str = "./model_data"
    ) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        df_model = df.dropna(subset=["referent_category"]).copy()

        category_map = {
            cat: idx for idx, cat in enumerate(df_model["referent_category"].unique())
        }
        df_model["label"] = df_model["referent_category"].map(category_map)

        for col in ["year", "poem_length"]:
            if col in df_model.columns:
                if df_model[col].dtype != "object":
                    df_model[f"{col}_norm"] = (
                        (df_model[col] - df_model[col].min())
                        / (df_model[col].max() - df_model[col].min())
                    )

        train_size = int(0.7 * len(df_model))
        val_size = int(0.15 * len(df_model))

        df_shuffled = df_model.sample(frac=1, random_state=42).reset_index(drop=True)

        df_train = df_shuffled[:train_size]
        df_val = df_shuffled[train_size : train_size + val_size]
        df_test = df_shuffled[train_size + val_size :]

        df_train.to_csv(output_path / "train.csv", index=False, encoding="utf-8")
        df_val.to_csv(output_path / "val.csv", index=False, encoding="utf-8")
        df_test.to_csv(output_path / "test.csv", index=False, encoding="utf-8")

        with open(output_path / "label_map.json", "w", encoding="utf-8") as f:
            json.dump(category_map, f, ensure_ascii=False, indent=2)

        stats = {
            "total_samples": len(df_model),
            "train_samples": len(df_train),
            "val_samples": len(df_val),
            "test_samples": len(df_test),
            "n_categories": len(category_map),
            "category_distribution": df_model["referent_category"]
            .value_counts()
            .to_dict(),
        }

        with open(output_path / "dataset_stats.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        print(f"model_data -> {output_dir}")
        print(f"train {len(df_train)} val {len(df_val)} test {len(df_test)} labels {len(category_map)}")

    def generate_annotation_report(
        self, df: pd.DataFrame, output_file: str = "annotation_report.txt"
    ) -> None:

        report = []
        report.append("=" * 60)
        report.append("annotation report")
        report.append("=" * 60)
        report.append("")

        report.append("1. overview")
        report.append(f"   tokens {len(df)}")
        report.append(f"   labeled {df['referent_category'].notna().sum()}")
        report.append(
            f"   rate {df['referent_category'].notna().sum() / len(df) * 100:.1f}%"
        )
        report.append("")

        report.append("2. referent_category")
        if "referent_category" in df.columns:
            cat_dist = df["referent_category"].value_counts()
            for cat, count in cat_dist.items():
                pct = count / len(df) * 100
                report.append(f"   {cat}: {count} ({pct:.1f}%)")
        report.append("")

        report.append("3. we_inclusivity (1pl)")
        df_1pl = df[(df["person"] == "1") & (df["number"] == "Plur")]
        if "we_inclusivity" in df_1pl.columns:
            incl_dist = df_1pl["we_inclusivity"].value_counts()
            for incl, count in incl_dist.items():
                pct = count / len(df_1pl) * 100 if len(df_1pl) > 0 else 0
                report.append(f"   {incl}: {count} ({pct:.1f}%)")
        report.append("")

        report.append("4. annotation_difficulty")
        if "annotation_difficulty" in df.columns:
            diff_dist = df["annotation_difficulty"].value_counts()
            for diff, count in diff_dist.items():
                pct = count / len(df) * 100
                report.append(f"   {diff}: {count} ({pct:.1f}%)")
        report.append("")

        report.append("5. referent_confidence")
        if "referent_confidence" in df.columns:
            avg_conf = df["referent_confidence"].mean()
            report.append(f"   mean {avg_conf:.2f} / 3.0")
            conf_dist = df["referent_confidence"].value_counts().sort_index()
            for conf, count in conf_dist.items():
                pct = count / len(df) * 100
                report.append(f"   {conf}: {count} ({pct:.1f}%)")
        report.append("")

        report.append("6. year")
        if "year" in df.columns:
            year_dist = df["year"].value_counts().sort_index()
            for year, count in year_dist.items():
                report.append(f"   {year}: {count}")
        report.append("")

        report_text = "\n".join(report)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report_text)

        print(report_text)
        print(f"report -> {output_file}")


def main():
    pronoun_file = "./outputs/01_pronoun_detection/ukrainian_pronouns_detailed.csv"

    toolkit = AnnotationToolkit(pronoun_file)

    sample_df = toolkit.stratified_sample(
        n_tokens=500,
        time_col="year",
        time_bins=[(2014, 2015), (2016, 2021), (2022, 2023), (2024, 2025)],
    )

    sample_df = toolkit.add_annotation_columns(sample_df)

    sample_df = toolkit.extract_context_windows(sample_df, text_col="text", context_size=2)

    output_file = "./outputs/03_annotation/annotation_sample.csv"
    Path("./outputs/03_annotation").mkdir(exist_ok=True, parents=True)
    sample_df.to_csv(output_file, index=False, encoding="utf-8")

    print(f"sample -> {output_file} n={len(sample_df)}")
    print(
        "1pl",
        ((sample_df["person"] == "1") & (sample_df["number"] == "Plur")).sum(),
    )
    print(
        "3pl",
        ((sample_df["person"] == "3") & (sample_df["number"] == "Plur")).sum(),
    )


if __name__ == "__main__":
    from utils.workspace import prepare_analysis_environment

    prepare_analysis_environment(__file__, matplotlib_backend=None)
    main()
