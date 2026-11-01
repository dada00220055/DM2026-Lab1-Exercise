"""Tools for Document-Term Matrix (DTM) construction and analysis."""

from typing import List, Optional, Union, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_core.tools import tool
from sklearn.feature_extraction.text import CountVectorizer


def make_tools(session):
    """Build and return DTM-related tools bound to the given session."""

    @tool
    def build_dtm_tool(
        ngram_range: List[int] = [1, 1],
        min_df: Union[int, float] = 1,
        max_df: Union[int, float] = 1.0,
        max_features: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Builds a document-term matrix (DTM) from session.dataframe text column.

        Tokenizes document text and computes word occurrence counts across the dataset,
        populating session.feature_matrix and session.feature_names, and calculating
        matrix sparsity.

        Args:
            ngram_range: Lower and upper boundary of n-grams to extract, as [min_n, max_n].
                Default is [1, 1] for unigrams.
            min_df: Minimum document frequency for vocabulary terms (absolute count if int,
                fraction if float).
            max_df: Maximum document frequency for vocabulary terms (absolute count if int,
                fraction if float).
            max_features: Maximum number of vocabulary features to retain, sorted by term frequency.

        Returns:
            Dict containing result_id, n_documents, n_terms, non_zero count, total_elements,
            and sparsity_pct.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            raise ValueError("session.dataframe must be loaded and contain a 'text' column before building DTM.")

        # Convert List[int] ngram_range to Tuple[int, int] for CountVectorizer
        ngram_tuple = (int(ngram_range[0]), int(ngram_range[1]))

        # Extract text, filling missing values with empty string
        texts = session.dataframe["text"].fillna("").astype(str).tolist()

        # Initialize and fit CountVectorizer
        vectorizer = CountVectorizer(
            ngram_range=ngram_tuple,
            min_df=min_df,
            max_df=max_df,
            max_features=max_features,
        )

        feature_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out().tolist()

        # Store matrix, feature names, and vectorizer artifact in session
        session.feature_matrix = feature_matrix
        session.feature_names = feature_names
        session.artifacts["count_vectorizer"] = vectorizer

        # Sync document labels if category_name or label column exists
        if "category_name" in session.dataframe.columns:
            session.set_labels(session.dataframe["category_name"].values)
        elif "label" in session.dataframe.columns:
            session.set_labels(session.dataframe["label"].values)

        # Compute sparsity metrics
        n_documents, n_terms = feature_matrix.shape
        non_zero = int(feature_matrix.nnz)
        total_elements = int(n_documents * n_terms)
        sparsity_pct = float(100.0 * (1.0 - non_zero / total_elements)) if total_elements > 0 else 0.0

        args = {
            "ngram_range": ngram_range,
            "min_df": min_df,
            "max_df": max_df,
            "max_features": max_features,
        }

        result_id = session.next_result_id("build_dtm")
        summary = {
            "result_id": result_id,
            "n_documents": n_documents,
            "n_terms": n_terms,
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": round(sparsity_pct, 4),
        }

        session.store_result("build_dtm_tool", args, summary)
        return summary

    @tool
    def term_frequency_tool(
        top_n: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Calculates global term frequencies across all documents in session.feature_matrix.

        Sums term counts column-wise across the document-term matrix.

        Args:
            top_n: Optional limit on the number of highest-frequency terms to include in the summary.
                If None, returns frequencies for all terms in the vocabulary.

        Returns:
            Dict containing result_id, total_terms, and term_frequencies (dictionary mapping terms to count).
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError("session.feature_matrix and session.feature_names must be built first using build_dtm_tool.")

        # Compute column sums across all documents
        col_sums = np.asarray(session.feature_matrix.sum(axis=0)).flatten()

        # Create DataFrame with term names and frequencies
        df_freq = pd.DataFrame({
            "term": session.feature_names,
            "frequency": col_sums
        }).sort_values(by="frequency", ascending=False).reset_index(drop=True)

        # Store full report
        result_id = session.next_result_id("term_frequency")

        # Slice top_n for summary output if specified
        if top_n is not None and top_n > 0:
            summary_df = df_freq.head(top_n)
        else:
            summary_df = df_freq

        term_frequencies = dict(zip(summary_df["term"], [int(x) for x in summary_df["frequency"]]))

        args = {"top_n": top_n}
        summary = {
            "result_id": result_id,
            "total_terms": len(session.feature_names),
            "term_frequencies": term_frequencies,
        }

        session.store_result("term_frequency_tool", args, summary, full_report=df_freq)
        return summary

    @tool
    def dtm_heatmap_tool(
        n_terms: int = 20,
        n_documents: int = 20,
    ) -> Dict[str, Any]:
        """Generates a heatmap visualization of a slice of the document-term matrix.

        Slices the raw document-term matrix by positional document and term order,
        plots an annotated heatmap stored in session.pending_figure, and returns the
        sliced matrix values.

        Args:
            n_terms: Maximum number of terms (columns) to slice from the vocabulary order.
            n_documents: Maximum number of documents (rows) to slice from existing row order.

        Returns:
            Dict containing result_id, n_terms_included, n_documents_included, term_names,
            document_indices, and sliced_matrix (nested list of row counts).
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError("session.feature_matrix and session.feature_names must be built first using build_dtm_tool.")

        # Determine slice bounds
        total_docs, total_vocab = session.feature_matrix.shape
        doc_cutoff = min(n_documents, total_docs)
        term_cutoff = min(n_terms, total_vocab)

        # Slice sub-matrix and convert to dense numpy array
        sliced = session.feature_matrix[:doc_cutoff, :term_cutoff].toarray()
        sliced_terms = session.feature_names[:term_cutoff]
        doc_indices = list(range(doc_cutoff))

        # Generate Heatmap plot
        fig, ax = plt.subplots(figsize=(max(6, term_cutoff * 0.8), max(4, doc_cutoff * 0.5)))
        sns.heatmap(
            sliced,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            xticklabels=sliced_terms,
            yticklabels=[f"Doc {i}" for i in doc_indices],
            ax=ax,
            cbar=True,
        )
        ax.set_title(f"DTM Heatmap (First {doc_cutoff} Docs x First {term_cutoff} Terms)")
        ax.set_xlabel("Terms")
        ax.set_ylabel("Documents")
        plt.tight_layout()

        # Set pending figure on session
        session.pending_figure = fig

        result_id = session.next_result_id("dtm_heatmap")
        sliced_matrix_list = [[int(val) for val in row] for row in sliced]

        args = {"n_terms": n_terms, "n_documents": n_documents}
        summary = {
            "result_id": result_id,
            "n_terms_included": term_cutoff,
            "n_documents_included": doc_cutoff,
            "term_names": sliced_terms,
            "document_indices": doc_indices,
            "sliced_matrix": sliced_matrix_list,
        }

        session.store_result("dtm_heatmap_tool", args, summary)
        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]
