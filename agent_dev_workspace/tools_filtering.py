"""Tools for feature selection and variance/correlation filtering."""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from scipy import stats
from langchain_core.tools import tool


def make_tools(session):
    """Build and return filtering tools bound to the given session."""

    @tool
    def variance_filter_tool(
        threshold: float = 0.0,
    ) -> Dict[str, Any]:
        """Filters feature matrix terms by variance threshold.

        Calculates term variance across all documents in session.feature_matrix, flags near-constant
        terms below the threshold for removal, and formats full report for visualization.

        Args:
            threshold: Minimum variance threshold for retaining a feature term. Features with
                variance strictly less than threshold are flagged as removed.

        Returns:
            Dict containing result_id, threshold, total_terms, n_kept, n_removed, kept_terms,
            removed_terms, and variance_scores dictionary.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError("session.feature_matrix and session.feature_names must be built first using build_dtm_tool.")

        # Convert sparse matrix columns to dense numpy array to compute per-term variance
        dense_matrix = session.feature_matrix.toarray()
        variances = np.var(dense_matrix, axis=0)

        # Build full report DataFrame sorted by variance descending
        # First column: term name ('feature'), Second column: variance metric ('variance')
        df_report = pd.DataFrame({
            "feature": session.feature_names,
            "variance": variances,
        }).sort_values(by="variance", ascending=False).reset_index(drop=True)

        # Partition terms into kept and removed based on threshold
        kept_mask = df_report["variance"] >= threshold
        df_kept = df_report[kept_mask]
        df_removed = df_report[~kept_mask]

        kept_terms = df_kept["feature"].tolist()
        removed_terms = df_removed["feature"].tolist()

        variance_scores = {
            row["feature"]: round(float(row["variance"]), 6)
            for _, row in df_report.iterrows()
        }

        result_id = session.next_result_id("variance_filter")
        args = {"threshold": threshold}
        summary = {
            "result_id": result_id,
            "threshold": threshold,
            "total_terms": len(session.feature_names),
            "n_kept": len(kept_terms),
            "n_removed": len(removed_terms),
            "kept_terms": kept_terms,
            "removed_terms": removed_terms,
            "variance_scores": variance_scores,
        }

        # Store result with full_report formatted with 'feature' as 1st col and 'variance' as 2nd col
        session.store_result("variance_filter_tool", args, summary, full_report=df_report)
        return summary

    @tool
    def pearson_filter_tool(
        target_class: str,
        threshold: float = 0.0,
    ) -> Dict[str, Any]:
        """Filters feature terms by Pearson correlation with a one-vs-rest target class.

        Computes linear Pearson correlation r between each feature column and a binary indicator
        vector for target_class, sorting and formatting the report for visualization.

        Args:
            target_class: The target class label to evaluate correlation against.
            threshold: Minimum Pearson correlation coefficient for keeping a feature term.
                Features with correlation strictly less than threshold are flagged as removed.

        Returns:
            Dict containing result_id, target_class, threshold, total_terms, n_kept, n_removed,
            kept_terms, removed_terms, and pearson_scores dictionary.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError("session.feature_matrix and session.feature_names must be built first using build_dtm_tool.")

        # Determine target class labels per document
        if session.dataframe is not None and "category_name" in session.dataframe.columns:
            labels = session.dataframe["category_name"].astype(str).values
        elif session.dataframe is not None and "label" in session.dataframe.columns:
            labels = session.dataframe["label"].astype(str).values
        elif session.labels is not None:
            labels = np.asarray(session.labels, dtype=str)
        else:
            raise ValueError("No document labels found in session.dataframe or session.labels.")

        # Create binary indicator vector for target_class
        binary_target = (labels == str(target_class)).astype(float)

        dense_matrix = session.feature_matrix.toarray()
        pearson_r_list = []

        # Compute Pearson r for each feature column
        for col_idx in range(dense_matrix.shape[1]):
            col_vals = dense_matrix[:, col_idx]
            # Check for zero variance to avoid division by zero / NaN
            if np.std(col_vals) == 0 or np.std(binary_target) == 0:
                r_val = 0.0
            else:
                r_val, _ = stats.pearsonr(col_vals, binary_target)
                if np.isnan(r_val):
                    r_val = 0.0
            pearson_r_list.append(float(r_val))

        # Build full report DataFrame sorted by pearson_r descending
        # First column: 'feature', Second column: 'pearson_r'
        df_report = pd.DataFrame({
            "feature": session.feature_names,
            "pearson_r": pearson_r_list,
        }).sort_values(by="pearson_r", ascending=False).reset_index(drop=True)

        kept_mask = df_report["pearson_r"] >= threshold
        df_kept = df_report[kept_mask]
        df_removed = df_report[~kept_mask]

        kept_terms = df_kept["feature"].tolist()
        removed_terms = df_removed["feature"].tolist()

        pearson_scores = {
            row["feature"]: round(float(row["pearson_r"]), 6)
            for _, row in df_report.iterrows()
        }

        result_id = session.next_result_id("pearson_filter")
        args = {"target_class": target_class, "threshold": threshold}
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "total_terms": len(session.feature_names),
            "n_kept": len(kept_terms),
            "n_removed": len(removed_terms),
            "kept_terms": kept_terms,
            "removed_terms": removed_terms,
            "pearson_scores": pearson_scores,
        }

        session.store_result("pearson_filter_tool", args, summary, full_report=df_report)
        return summary

    @tool
    def spearman_filter_tool(
        target_class: str,
        threshold: float = 0.0,
    ) -> Dict[str, Any]:
        """Filters feature terms by Spearman rank correlation with a one-vs-rest target class.

        Computes monotonic Spearman rank correlation rho between each feature column and a binary indicator
        vector for target_class, sorting and formatting the report for visualization.

        Args:
            target_class: The target class label to evaluate correlation against.
            threshold: Minimum Spearman rank correlation coefficient for keeping a feature term.
                Features with correlation strictly less than threshold are flagged as removed.

        Returns:
            Dict containing result_id, target_class, threshold, total_terms, n_kept, n_removed,
            kept_terms, removed_terms, and spearman_scores dictionary.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError("session.feature_matrix and session.feature_names must be built first using build_dtm_tool.")

        # Determine target class labels per document
        if session.dataframe is not None and "category_name" in session.dataframe.columns:
            labels = session.dataframe["category_name"].astype(str).values
        elif session.dataframe is not None and "label" in session.dataframe.columns:
            labels = session.dataframe["label"].astype(str).values
        elif session.labels is not None:
            labels = np.asarray(session.labels, dtype=str)
        else:
            raise ValueError("No document labels found in session.dataframe or session.labels.")

        # Create binary indicator vector for target_class
        binary_target = (labels == str(target_class)).astype(float)

        dense_matrix = session.feature_matrix.toarray()
        spearman_r_list = []

        # Compute Spearman rank correlation for each feature column
        for col_idx in range(dense_matrix.shape[1]):
            col_vals = dense_matrix[:, col_idx]
            if np.std(col_vals) == 0 or np.std(binary_target) == 0:
                r_val = 0.0
            else:
                r_val, _ = stats.spearmanr(col_vals, binary_target)
                if np.isnan(r_val):
                    r_val = 0.0
            spearman_r_list.append(float(r_val))

        # Build full report DataFrame sorted by spearman_r descending
        # First column: 'feature', Second column: 'spearman_r'
        df_report = pd.DataFrame({
            "feature": session.feature_names,
            "spearman_r": spearman_r_list,
        }).sort_values(by="spearman_r", ascending=False).reset_index(drop=True)

        kept_mask = df_report["spearman_r"] >= threshold
        df_kept = df_report[kept_mask]
        df_removed = df_report[~kept_mask]

        kept_terms = df_kept["feature"].tolist()
        removed_terms = df_removed["feature"].tolist()

        spearman_scores = {
            row["feature"]: round(float(row["spearman_r"]), 6)
            for _, row in df_report.iterrows()
        }

        result_id = session.next_result_id("spearman_filter")
        args = {"target_class": target_class, "threshold": threshold}
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "total_terms": len(session.feature_names),
            "n_kept": len(kept_terms),
            "n_removed": len(removed_terms),
            "kept_terms": kept_terms,
            "removed_terms": removed_terms,
            "spearman_scores": spearman_scores,
        }

        session.store_result("spearman_filter_tool", args, summary, full_report=df_report)
        return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]
