"""Tools for dimensionality reduction and label encoding."""

from typing import Dict, Any, Optional, List, Literal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelBinarizer
import umap
from langchain_core.tools import tool


def make_tools(session):
    """Build and return reduction tools bound to the given session."""

    @tool
    def reduce_dimensions_tool(
        method: Literal["pca", "tsne", "umap"] = "pca",
        n_components: int = 2,
        random_state: Optional[int] = 42,
        perplexity: float = 30.0,
        n_neighbors: int = 15,
    ) -> Dict[str, Any]:
        """Reduces feature matrix dimensionality to 2D coordinates and plots a scatter figure.

        Applies PCA, t-SNE, or UMAP to session.feature_matrix, stores coordinates in session.artifacts,
        and generates a 2D scatter plot colored by category set on session.pending_figure.

        Args:
            method: Reduction technique, one of 'pca', 'tsne', or 'umap'. Default is 'pca'.
            n_components: Number of output dimensions (default 2).
            random_state: Seed for reproducibility across probabilistic algorithms.
            perplexity: Perplexity parameter for t-SNE (must be less than n_samples).
            n_neighbors: Number of neighboring points for UMAP (must be less than n_samples).

        Returns:
            Dict containing result_id, method, coordinates, and explained_variance_ratio (PCA only).
        """
        if session.feature_matrix is None:
            raise ValueError("session.feature_matrix must be built first using build_dtm_tool.")

        dense_matrix = session.feature_matrix.toarray()
        n_samples, n_features = dense_matrix.shape

        if n_samples < 2:
            raise ValueError("At least 2 documents are required to perform dimensionality reduction.")

        explained_variance_ratio = None

        # Execute reduction algorithm
        if method.lower() == "pca":
            reducer = PCA(n_components=n_components, random_state=random_state)
            coords = reducer.fit_transform(dense_matrix)
            explained_variance_ratio = [float(val) for val in reducer.explained_variance_ratio_]

        elif method.lower() == "tsne":
            # Adjust perplexity if sample size is smaller than requested perplexity
            adj_perplexity = min(perplexity, max(1.0, float(n_samples - 1)))
            reducer = TSNE(
                n_components=n_components,
                perplexity=adj_perplexity,
                random_state=random_state,
                init="random",
            )
            coords = reducer.fit_transform(dense_matrix)

        elif method.lower() == "umap":
            # Adjust n_neighbors if sample size is smaller than requested n_neighbors
            adj_neighbors = min(n_neighbors, max(2, n_samples - 1))
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=adj_neighbors,
                random_state=random_state,
            )
            coords = reducer.fit_transform(dense_matrix)
        else:
            raise ValueError(f"Unsupported method '{method}'. Must be one of 'pca', 'tsne', or 'umap'.")

        # Save coordinates in session artifacts
        session.artifacts[f"reduced_{method.lower()}"] = coords

        # Retrieve document category labels for color mapping in plot
        labels = None
        if session.dataframe is not None and "category_name" in session.dataframe.columns:
            labels = session.dataframe["category_name"].astype(str).tolist()
        elif session.dataframe is not None and "label" in session.dataframe.columns:
            labels = session.dataframe["label"].astype(str).tolist()
        elif session.labels is not None:
            labels = [str(lbl) for lbl in session.labels]
        else:
            labels = [f"Doc {i}" for i in range(n_samples)]

        # Generate mandatory category-colored scatter plot
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            x=coords[:, 0],
            y=coords[:, 1],
            hue=labels,
            style=labels,
            s=100,
            ax=ax,
        )
        ax.set_title(f"2D Dimensionality Reduction ({method.upper()})")
        ax.set_xlabel("Component 1" if method.lower() == "pca" else "Dimension 1")
        ax.set_ylabel("Component 2" if method.lower() == "pca" else "Dimension 2")
        plt.tight_layout()

        # Set mandatory pending figure on session
        session.pending_figure = fig

        coords_list = [[round(float(pt[0]), 4), round(float(pt[1]), 4)] for pt in coords]

        result_id = session.next_result_id(f"reduce_{method.lower()}")
        args = {
            "method": method,
            "n_components": n_components,
            "random_state": random_state,
            "perplexity": perplexity,
            "n_neighbors": n_neighbors,
        }
        summary = {
            "result_id": result_id,
            "method": method.lower(),
            "reduced_coords": coords_list,
        }
        if explained_variance_ratio is not None:
            summary["explained_variance_ratio"] = [round(v, 4) for v in explained_variance_ratio]

        session.store_result("reduce_dimensions_tool", args, summary)
        return summary

    @tool
    def binarize_labels_tool() -> Dict[str, Any]:
        """One-hot encodes category labels into a binary indicator matrix.

        Fits a LabelBinarizer on document category labels, stores the binarizer and binary matrix
        in session.artifacts, and returns the encoding matrix.

        Returns:
            Dict containing result_id, n_documents, classes, encoded_shape, and binary_matrix.
        """
        if session.dataframe is None and session.labels is None:
            raise ValueError("session.dataframe or session.labels must be loaded before binarizing labels.")

        # Determine target class labels per document
        if session.dataframe is not None and "category_name" in session.dataframe.columns:
            raw_labels = session.dataframe["category_name"].astype(str).values
        elif session.dataframe is not None and "label" in session.dataframe.columns:
            raw_labels = session.dataframe["label"].astype(str).values
        elif session.labels is not None:
            raw_labels = np.asarray(session.labels, dtype=str)
        else:
            raise ValueError("No valid category labels found in session.dataframe or session.labels.")

        lb = LabelBinarizer()
        binary_matrix = lb.fit_transform(raw_labels)

        # Handle binary single-column output from LabelBinarizer for 2 classes
        if len(lb.classes_) == 2 and binary_matrix.shape[1] == 1:
            # Expand to 2 columns [class0, class1] for clarity
            col0 = 1 - binary_matrix.flatten()
            col1 = binary_matrix.flatten()
            binary_matrix = np.column_stack([col0, col1])

        classes_list = [str(c) for c in lb.classes_]

        # Store artifacts on session
        session.artifacts["label_binarizer"] = lb
        session.artifacts["binarized_labels"] = binary_matrix

        matrix_list = [[int(val) for val in row] for row in binary_matrix]

        result_id = session.next_result_id("binarize_labels")
        summary = {
            "result_id": result_id,
            "n_documents": len(raw_labels),
            "classes": classes_list,
            "encoded_shape": list(binary_matrix.shape),
            "binary_matrix": matrix_list,
        }

        # Format full report DataFrame
        df_report = pd.DataFrame(binary_matrix, columns=classes_list)

        session.store_result("binarize_labels_tool", {}, summary, full_report=df_report)
        return summary

    return [reduce_dimensions_tool, binarize_labels_tool]
