"""Tools for dataset exploration and similarity metrics."""

from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.tools import tool


def make_tools(session):
    """Build and return exploration tools bound to the given session."""

    @tool
    def cosine_similarity_tool(
        doc_idx1: int,
        doc_idx2: int,
    ) -> Dict[str, Any]:
        """Calculates cosine similarity between two documents in session.dataframe.

        Uses the count_vectorizer stored in session.artifacts (or session.feature_matrix)
        to compare document vectors, safely returning 0.0 for empty text documents.

        Args:
            doc_idx1: Integer row index of the first document in session.dataframe.
            doc_idx2: Integer row index of the second document in session.dataframe.

        Returns:
            Dict containing result_id, doc_idx1, doc_idx2, and similarity score.
        """
        if session.dataframe is None:
            raise ValueError("session.dataframe must be loaded before calculating similarity.")

        n_docs = len(session.dataframe)
        if doc_idx1 < 0 or doc_idx1 >= n_docs or doc_idx2 < 0 or doc_idx2 >= n_docs:
            raise ValueError(f"doc_idx1 ({doc_idx1}) and doc_idx2 ({doc_idx2}) must be valid indices between 0 and {n_docs - 1}.")

        # Retrieve vectors directly from session.feature_matrix if available, or transform text
        if session.feature_matrix is not None:
            vec1 = session.feature_matrix[doc_idx1]
            vec2 = session.feature_matrix[doc_idx2]
        elif "count_vectorizer" in session.artifacts:
            vectorizer = session.artifacts["count_vectorizer"]
            texts = session.dataframe["text"].fillna("").astype(str).tolist()
            vec1 = vectorizer.transform([texts[doc_idx1]])
            vec2 = vectorizer.transform([texts[doc_idx2]])
        else:
            raise ValueError("Either session.feature_matrix or count_vectorizer artifact must be present.")

        # Check for zero norm / empty vectors to handle division by zero
        norm1 = np.linalg.norm(vec1.toarray())
        norm2 = np.linalg.norm(vec2.toarray())

        if norm1 == 0.0 or norm2 == 0.0:
            sim_val = 0.0
        else:
            sim_matrix = cosine_similarity(vec1, vec2)
            sim_val = float(sim_matrix[0, 0])
            if np.isnan(sim_val):
                sim_val = 0.0

        result_id = session.next_result_id("cosine_similarity")
        args = {"doc_idx1": doc_idx1, "doc_idx2": doc_idx2}
        summary = {
            "result_id": result_id,
            "doc_idx1": doc_idx1,
            "doc_idx2": doc_idx2,
            "similarity": round(sim_val, 4),
        }

        session.store_result("cosine_similarity_tool", args, summary)
        return summary

    @tool
    def feature_correlation_matrix_tool(
        top_n: int = 20,
    ) -> Dict[str, Any]:
        """Calculates feature-vs-feature Pearson correlation matrix over top terms by variance.

        Selects up to top_n terms with highest variance across all documents, computes their
        pairwise Pearson correlation matrix, plots an annotated heatmap stored in
        session.pending_figure, and returns the matrix and term names.

        Args:
            top_n: Maximum number of top-variance terms to include (default 20).

        Returns:
            Dict containing result_id, top_n_included, term_names, and correlation_matrix
            (nested list of rounded floats).
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError("session.feature_matrix and session.feature_names must be built first using build_dtm_tool.")

        dense_matrix = session.feature_matrix.toarray()
        n_docs, total_terms = dense_matrix.shape

        if total_terms == 0 or n_docs == 0:
            raise ValueError("session.feature_matrix must contain non-empty data to compute correlations.")

        # Compute term variances across documents
        variances = np.var(dense_matrix, axis=0)

        # Select top_n features by variance (descending order)
        cutoff = min(top_n, total_terms)
        top_indices = np.argsort(-variances)[:cutoff]

        top_terms = [session.feature_names[i] for i in top_indices]
        top_matrix = dense_matrix[:, top_indices]

        # Compute Pearson correlation matrix across top terms
        # rowvar=False because variables/terms are columns
        corr_matrix = np.corrcoef(top_matrix, rowvar=False)

        # Handle 1x1 scalar or 2D matrix, and replace NaNs (from zero-variance terms) with 0.0 (1.0 on diagonal)
        if cutoff == 1:
            corr_matrix = np.array([[1.0]])
        else:
            np.fill_diagonal(corr_matrix, 1.0)
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        # Plot annotated heatmap
        fig, ax = plt.subplots(figsize=(max(6, cutoff * 0.8), max(5, cutoff * 0.6)))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="vlag",
            xticklabels=top_terms,
            yticklabels=top_terms,
            ax=ax,
            center=0,
            vmin=-1,
            vmax=1,
        )
        ax.set_title(f"Feature Correlation Matrix (Top {cutoff} Terms by Variance)")
        plt.tight_layout()

        # Set pending figure on session
        session.pending_figure = fig

        result_id = session.next_result_id("feature_correlation")
        corr_matrix_list = [[round(float(val), 4) for val in row] for row in corr_matrix]

        args = {"top_n": top_n}
        summary = {
            "result_id": result_id,
            "top_n_included": cutoff,
            "term_names": top_terms,
            "correlation_matrix": corr_matrix_list,
        }

        session.store_result("feature_correlation_matrix_tool", args, summary)
        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]
