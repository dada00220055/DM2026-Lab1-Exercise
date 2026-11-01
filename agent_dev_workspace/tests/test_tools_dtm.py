"""Tests for tools_dtm.py using sample_fixture.csv and known answers from TEST_FIXTURE.md."""

import numpy as np
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools


def test_build_dtm_tool_known_answer_4():
    """Test build_dtm_tool against known answer #4 in TEST_FIXTURE.md.

    Known Answer #4 specs:
    - Vocabulary: alpha, always, beta, delta, gamma (5 terms).
    - Matrix shape: 8 documents x 5 terms.
    - Sparsity: 21 non-zero cells out of 40 total elements.
    - Sparsity percentage: 100 * (1 - 21/40) = 47.5%.
    """
    # Create single session instance
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load sample_fixture.csv into session
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Instantiate student's tools bound to the same session
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")

    # Invoke build_dtm_tool
    result = build_dtm_tool.invoke({})

    # Assert returned summary values against known answer #4
    assert result["n_documents"] == 8
    assert result["n_terms"] == 5
    assert result["non_zero"] == 21
    assert result["total_elements"] == 40
    assert np.isclose(result["sparsity_pct"], 47.5, atol=1e-3)

    # Assert session state updates
    assert session.feature_matrix is not None
    assert session.feature_matrix.shape == (8, 5)
    assert session.feature_names == ["alpha", "always", "beta", "delta", "gamma"]
    assert "count_vectorizer" in session.artifacts


def test_term_frequency_tool_known_answer_14():
    """Test term_frequency_tool against known answer #14 in TEST_FIXTURE.md.

    Known Answer #14 specs:
    - alpha: 6
    - always: 7
    - beta: 3
    - delta: 4
    - gamma: 7
    """
    # Create single session instance per test function
    session = SessionState()

    # Load dataset
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Instantiate student tools
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    term_frequency_tool = next(t for t in dtm_tools if t.name == "term_frequency_tool")

    # Build DTM first
    build_dtm_tool.invoke({})

    # Invoke term_frequency_tool
    result = term_frequency_tool.invoke({})

    # Expected frequencies from Known Answer #14
    expected_freqs = {
        "alpha": 6,
        "always": 7,
        "beta": 3,
        "delta": 4,
        "gamma": 7,
    }

    assert result["total_terms"] == 5
    assert result["term_frequencies"] == expected_freqs


def test_dtm_heatmap_tool_known_answer_21():
    """Test dtm_heatmap_tool against known answer #21 in TEST_FIXTURE.md."""
    # Create single session instance per test function
    session = SessionState()

    # Load dataset and build DTM
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    dtm_heatmap_tool = next(t for t in dtm_tools if t.name == "dtm_heatmap_tool")

    build_dtm_tool.invoke({})

    # Test default full slice (n_terms=20, n_documents=20)
    res_full = dtm_heatmap_tool.invoke({"n_terms": 20, "n_documents": 20})

    expected_full_matrix = [
        [3, 1, 1, 0, 0],
        [2, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 1, 0, 1, 3],
        [0, 1, 0, 1, 2],
        [0, 1, 0, 1, 1],
        [0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0],
    ]

    assert res_full["n_terms_included"] == 5
    assert res_full["n_documents_included"] == 8
    assert res_full["term_names"] == ["alpha", "always", "beta", "delta", "gamma"]
    assert res_full["sliced_matrix"] == expected_full_matrix
    assert session.pending_figure is not None

    # Test specific slice (n_terms=2, n_documents=3)
    res_slice = dtm_heatmap_tool.invoke({"n_terms": 2, "n_documents": 3})

    expected_corner = [
        [3, 1],
        [2, 1],
        [1, 1],
    ]

    assert res_slice["n_terms_included"] == 2
    assert res_slice["n_documents_included"] == 3
    assert res_slice["term_names"] == ["alpha", "always"]
    assert res_slice["sliced_matrix"] == expected_corner
