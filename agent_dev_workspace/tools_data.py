"""Tools for data loading, inspection, filtering, and summary statistics."""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize
from langchain_core.tools import tool

# Ensure punkt tokenizer models are available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


def make_tools(session):
    """Build and return data tools bound to the given session."""

    @tool
    def list_files_tool(
        subdirectory: str = "",
    ) -> Dict[str, Any]:
        """Lists files and directories under a requested repository subdirectory.

        Scans the specified subdirectory path and returns sorted lists of files and
        subdirectories, enabling dataset discovery prior to loading.

        Args:
            subdirectory: Relative path to the folder to list (e.g. 'newdataset' or 'agent_dev').
                Defaults to root workspace directory if empty string.

        Returns:
            Dict containing result_id, subdirectory, files list, and directories list.
        """
        # Resolve target path relative to current working directory
        target_path = Path.cwd() / subdirectory if subdirectory else Path.cwd()

        if not target_path.exists():
            raise FileNotFoundError(f"Subdirectory '{subdirectory}' does not exist.")
        if not target_path.is_dir():
            raise ValueError(f"Path '{subdirectory}' is not a directory.")

        files = []
        directories = []

        # Iterate over contents, filtering out hidden items
        for entry in target_path.iterdir():
            if entry.name.startswith("."):
                continue
            if entry.is_file():
                files.append(entry.name)
            elif entry.is_dir():
                directories.append(entry.name)

        files.sort()
        directories.sort()

        result_id = session.next_result_id("list_files")
        args = {"subdirectory": subdirectory}
        summary = {
            "result_id": result_id,
            "subdirectory": subdirectory,
            "files": files,
            "directories": directories,
        }

        session.store_result("list_files_tool", args, summary)
        return summary

    @tool
    def inspect_data_tool(
        n_rows: int = 5,
    ) -> Dict[str, Any]:
        """Peeks at the first n_rows of the working dataset in session.dataframe.

        Displays document text, category labels, and row indices, safely replacing missing
        or NaN values with clear text indicators.

        Args:
            n_rows: Number of top rows to inspect (default 5).

        Returns:
            Dict containing result_id, total_rows, inspected_rows, columns, and sample_records list.
        """
        if session.dataframe is None:
            raise ValueError("session.dataframe is not loaded. Call load_dataset_tool first.")

        df = session.dataframe
        total_rows = len(df)
        cutoff = min(n_rows, total_rows)

        # Slice top rows
        df_slice = df.head(cutoff)

        # Identify category column name
        cat_col = "category_name" if "category_name" in df.columns else ("label" if "label" in df.columns else None)

        sample_records = []
        for idx, row in df_slice.iterrows():
            # Handle text extraction and missing value formatting
            raw_text = row.get("text", None)
            if pd.isna(raw_text) or raw_text is None or str(raw_text).strip() == "":
                text_val = "<missing>"
            else:
                text_val = str(raw_text)

            # Handle category extraction and missing value formatting
            if cat_col and cat_col in row:
                raw_cat = row[cat_col]
                if pd.isna(raw_cat) or raw_cat is None or str(raw_cat).strip() == "":
                    cat_val = "<missing>"
                else:
                    cat_val = str(raw_cat)
            else:
                cat_val = "<missing>"

            sample_records.append({
                "row_index": int(idx),
                "text": text_val,
                "category": cat_val,
            })

        result_id = session.next_result_id("inspect_data")
        args = {"n_rows": n_rows}
        summary = {
            "result_id": result_id,
            "total_rows": total_rows,
            "inspected_rows": cutoff,
            "columns": list(df.columns),
            "sample_records": sample_records,
        }

        session.store_result("inspect_data_tool", args, summary)
        return summary

    @tool
    def check_missing_tool(
        text_column: str = "text",
    ) -> Dict[str, Any]:
        """Audits the working dataset for missing, null, or empty text values.

        Identifies documents where text is NaN, None, or whitespace-only, returning count,
        percentage, and missing row indices.

        Args:
            text_column: Name of the text column to audit (default 'text').

        Returns:
            Dict containing result_id, total_documents, missing_count, missing_percentage,
            and missing_indices list.
        """
        if session.dataframe is None:
            raise ValueError("session.dataframe is not loaded. Call load_dataset_tool first.")

        if text_column not in session.dataframe.columns:
            raise ValueError(f"Column '{text_column}' not found in session.dataframe.")

        series = session.dataframe[text_column]
        total_documents = len(series)

        missing_indices = []
        for idx, val in enumerate(series):
            if pd.isna(val) or val is None or str(val).strip() == "":
                missing_indices.append(idx)

        missing_count = len(missing_indices)
        missing_percentage = float(round(100.0 * missing_count / total_documents, 2)) if total_documents > 0 else 0.0

        result_id = session.next_result_id("check_missing")
        args = {"text_column": text_column}
        summary = {
            "result_id": result_id,
            "total_documents": total_documents,
            "missing_count": missing_count,
            "missing_percentage": missing_percentage,
            "missing_indices": missing_indices,
        }

        session.store_result("check_missing_tool", args, summary)
        return summary

    @tool
    def check_duplicates_tool(
        subset: Optional[List[str]] = None,
        drop: bool = False,
    ) -> Dict[str, Any]:
        """Audits session.dataframe for duplicate documents and optionally drops all copies.

        Identifies duplicate rows based on subset columns. When drop=True, drops EVERY copy
        of duplicated rows using keep=False (matching Master's drop_duplicates(keep=False)).

        Args:
            subset: Optional list of column names to check for duplicates (defaults to ['text']).
            drop: If True, drops all copies of duplicated rows from session.dataframe.

        Returns:
            Dict containing result_id, total_documents_before, duplicate_count, drop_applied,
            and total_documents_after.
        """
        if session.dataframe is None:
            raise ValueError("session.dataframe is not loaded. Call load_dataset_tool first.")

        if subset is None:
            subset = ["text"] if "text" in session.dataframe.columns else list(session.dataframe.columns)

        total_before = len(session.dataframe)

        # Identify duplicate rows (keep='first' counts extra copies, keep=False identifies all copies)
        extra_duplicates_mask = session.dataframe.duplicated(subset=subset, keep="first")
        duplicate_count = int(extra_duplicates_mask.sum())

        if drop:
            # Drop every copy of duplicated rows using keep=False matching spec
            session.dataframe.drop_duplicates(subset=subset, keep=False, inplace=True)
            session.dataframe.reset_index(drop=True, inplace=True)

        total_after = len(session.dataframe)

        result_id = session.next_result_id("check_duplicates")
        args = {"subset": subset, "drop": drop}
        summary = {
            "result_id": result_id,
            "total_documents_before": total_before,
            "duplicate_count": duplicate_count,
            "drop_applied": drop,
            "total_documents_after": total_after,
        }

        session.store_result("check_duplicates_tool", args, summary)
        return summary

    @tool
    def sample_data_tool(
        n: Optional[int] = None,
        frac: Optional[float] = None,
        random_state: Optional[int] = 42,
    ) -> Dict[str, Any]:
        """Draws a random subsample of documents from session.dataframe.

        Subsamples the dataset by count n or fraction frac, using random_state for reproducible
        sampling outputs.

        Args:
            n: Number of documents to sample.
            frac: Fraction of total documents to sample (0.0 to 1.0).
            random_state: Integer seed for reproducible sampling outputs.

        Returns:
            Dict containing result_id, total_documents_before, sampled_count, random_state,
            and sampled_indices list.
        """
        if session.dataframe is None:
            raise ValueError("session.dataframe is not loaded. Call load_dataset_tool first.")

        total_before = len(session.dataframe)

        if n is None and frac is None:
            n = min(5, total_before)

        sampled_df = session.dataframe.sample(
            n=n,
            frac=frac,
            random_state=random_state,
            replace=False,
        )

        sampled_indices = [int(i) for i in sampled_df.index.tolist()]

        result_id = session.next_result_id("sample_data")
        args = {"n": n, "frac": frac, "random_state": random_state}
        summary = {
            "result_id": result_id,
            "total_documents_before": total_before,
            "sampled_count": len(sampled_indices),
            "random_state": random_state,
            "sampled_indices": sampled_indices,
        }

        session.store_result("sample_data_tool", args, summary, full_report=sampled_df)
        return summary

    @tool
    def describe_data_tool() -> Dict[str, Any]:
        """Computes summary statistics on document text length and plots a category boxplot.

        Calculates character count text_length per document, computes overall and per-category
        describe() statistics, and sets a boxplot figure on session.pending_figure.

        Returns:
            Dict containing result_id, overall_describe, and per_category_describe stats.
        """
        if session.dataframe is None:
            raise ValueError("session.dataframe is not loaded. Call load_dataset_tool first.")

        df = session.dataframe.copy()

        # Compute character length per document, handling missing text cleanly as 0 length
        df["text_length"] = df["text"].fillna("").astype(str).apply(len)

        # Helper function to convert .describe() series to dict with rounded floats
        def format_describe(s: pd.Series) -> Dict[str, float]:
            desc = s.describe()
            return {stat: float(round(val, 4)) for stat, val in desc.items()}

        overall_stats = format_describe(df["text_length"])

        # Identify category column name
        cat_col = "category_name" if "category_name" in df.columns else ("label" if "label" in df.columns else None)

        per_category_stats = {}
        if cat_col:
            for cat_name, group in df.groupby(cat_col):
                per_category_stats[str(cat_name)] = format_describe(group["text_length"])

        # Generate mandatory category boxplot figure
        fig, ax = plt.subplots(figsize=(8, 5))
        if cat_col:
            sns.boxplot(data=df, x=cat_col, y="text_length", ax=ax)
            ax.set_xlabel("Category")
        else:
            sns.boxplot(data=df, y="text_length", ax=ax)
        ax.set_ylabel("Text Length (characters)")
        ax.set_title("Document Text Length Distribution by Category")
        plt.tight_layout()

        # Set mandatory pending figure on session
        session.pending_figure = fig

        result_id = session.next_result_id("describe_data")
        summary = {
            "result_id": result_id,
            "overall_describe": overall_stats,
            "per_category_describe": per_category_stats,
            "text_lengths": df["text_length"].tolist(),
        }

        session.store_result("describe_data_tool", {}, summary, full_report=df[["text", cat_col, "text_length"]] if cat_col else df[["text", "text_length"]])
        return summary

    @tool
    def tokenize_tool(
        text_column: str = "text",
    ) -> Dict[str, Any]:
        """Tokenizes document text into lowercased unigrams using nltk.word_tokenize.

        Populates session.dataframe['unigrams'] and session.artifacts['unigrams'], safely
        returning empty lists [] for empty text values.

        Args:
            text_column: Name of the text column to tokenize (default 'text').

        Returns:
            Dict containing result_id, total_documents, total_tokens, avg_tokens_per_doc,
            and sample_tokens for the first 3 documents.
        """
        if session.dataframe is None:
            raise ValueError("session.dataframe is not loaded. Call load_dataset_tool first.")

        if text_column not in session.dataframe.columns:
            raise ValueError(f"Column '{text_column}' not found in session.dataframe.")

        unigrams_list = []
        for val in session.dataframe[text_column]:
            if pd.isna(val) or val is None or str(val).strip() == "":
                tokens = []
            else:
                tokens = word_tokenize(str(val).lower())
            unigrams_list.append(tokens)

        # Update session state
        session.dataframe["unigrams"] = unigrams_list
        session.artifacts["unigrams"] = unigrams_list

        total_documents = len(unigrams_list)
        total_tokens = sum(len(toks) for toks in unigrams_list)
        avg_tokens = float(round(total_tokens / total_documents, 2)) if total_documents > 0 else 0.0

        result_id = session.next_result_id("tokenize")
        args = {"text_column": text_column}
        summary = {
            "result_id": result_id,
            "total_documents": total_documents,
            "total_tokens": total_tokens,
            "avg_tokens_per_doc": avg_tokens,
            "sample_tokens": unigrams_list[:3],
        }

        session.store_result("tokenize_tool", args, summary)
        return summary

    return [
        list_files_tool,
        inspect_data_tool,
        check_missing_tool,
        check_duplicates_tool,
        sample_data_tool,
        describe_data_tool,
        tokenize_tool,
    ]
