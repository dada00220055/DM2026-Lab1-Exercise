"""Tests for tools_reduction.py using sample_fixture.csv and known answers from TEST_FIXTURE.md."""

import numpy as np
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_reduction import make_tools as make_reduction_tools


def test_reduce_dimensions_pca_known_answer_7():
    """Test reduce_dimensions_tool with method='pca' against known answer #7 in TEST_FIXTURE.md."""
    session = SessionState()

    # Load dataset & build DTM
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    reduction_tools = make_reduction_tools(session)
    reduce_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    res = reduce_tool.invoke({"method": "pca", "n_components": 2})

    expected_coords = [
        [2.3669, 0.8871],
        [1.7039, 0.2642],
        [1.0408, -0.3587],
        [-2.0759, 1.0129],
        [-1.4554, 0.3405],
        [-0.8349, -0.3319],
        [-0.8349, -0.3319],
        [0.0894, -1.4822],
    ]

    actual_coords = np.array(res["reduced_coords"])
    expected_arr = np.array(expected_coords)

    # Assert PCA coordinates and explained variance ratio
    assert np.allclose(actual_coords, expected_arr, atol=1e-2)
    assert np.allclose(res["explained_variance_ratio"], [0.7532, 0.1965], atol=1e-2)

    # Assert mandatory figure plot was created and set on session
    assert session.pending_figure is not None
    assert "reduced_pca" in session.artifacts


def test_reduce_dimensions_tsne_known_answer_8():
    """Test reduce_dimensions_tool with method='tsne' against known answer #8 in TEST_FIXTURE.md."""
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    reduction_tools = make_reduction_tools(session)
    reduce_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    res = reduce_tool.invoke({
        "method": "tsne",
        "perplexity": 2,
        "random_state": 42
    })

    assert len(res["reduced_coords"]) == 8
    assert session.pending_figure is not None
    assert "reduced_tsne" in session.artifacts


def test_reduce_dimensions_umap_known_answer_9():
    """Test reduce_dimensions_tool with method='umap' against known answer #9 in TEST_FIXTURE.md."""
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    reduction_tools = make_reduction_tools(session)
    reduce_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    res = reduce_tool.invoke({
        "method": "umap",
        "n_neighbors": 3,
        "random_state": 42
    })

    assert len(res["reduced_coords"]) == 8
    assert session.pending_figure is not None
    assert "reduced_umap" in session.artifacts


def test_binarize_labels_tool():
    """Test binarize_labels_tool one-hot encoding on sample_fixture.csv."""
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    reduction_tools = make_reduction_tools(session)
    binarize_tool = next(t for t in reduction_tools if t.name == "binarize_labels_tool")

    res = binarize_tool.invoke({})

    assert res["n_documents"] == 8
    assert res["classes"] == ["catA", "catB"]
    assert res["encoded_shape"] == [8, 2]

    # catA rows (0,1,2,7) should be [1, 0], catB rows (3,4,5,6) should be [0, 1]
    matrix = res["binary_matrix"]
    assert matrix[0] == [1, 0]
    assert matrix[1] == [1, 0]
    assert matrix[2] == [1, 0]
    assert matrix[3] == [0, 1]
    assert matrix[4] == [0, 1]
    assert matrix[5] == [0, 1]
    assert matrix[6] == [0, 1]
    assert matrix[7] == [1, 0]

    assert "label_binarizer" in session.artifacts
    assert "binarized_labels" in session.artifacts
