"""Tests for tools_exploration.py using sample_fixture.csv and known answers from TEST_FIXTURE.md."""

import numpy as np
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_exploration import make_tools as make_exploration_tools


def test_cosine_similarity_tool_known_answer_16():
    """Test cosine_similarity_tool against known answer #16 in TEST_FIXTURE.md.

    Known Answer #16 specs:
    - Doc 0 vs Doc 1: 0.9847
    - Doc 0 vs Doc 3: 0.0909
    - Doc 3 vs Doc 4: 0.9847
    - Doc 0 vs Doc 7 (empty text): 0.0
    """
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

    exploration_tools = make_exploration_tools(session)
    cosine_similarity_tool = next(t for t in exploration_tools if t.name == "cosine_similarity_tool")

    # 0 vs 1 (both catA, near identical)
    res_0_1 = cosine_similarity_tool.invoke({"doc_idx1": 0, "doc_idx2": 1})
    assert np.isclose(res_0_1["similarity"], 0.9847, atol=1e-3)

    # 0 vs 3 (catA vs catB, share only 'always')
    res_0_3 = cosine_similarity_tool.invoke({"doc_idx1": 0, "doc_idx2": 3})
    assert np.isclose(res_0_3["similarity"], 0.0909, atol=1e-3)

    # 3 vs 4 (both catB, near identical)
    res_3_4 = cosine_similarity_tool.invoke({"doc_idx1": 3, "doc_idx2": 4})
    assert np.isclose(res_3_4["similarity"], 0.9847, atol=1e-3)

    # 0 vs 7 (catA vs empty text)
    res_0_7 = cosine_similarity_tool.invoke({"doc_idx1": 0, "doc_idx2": 7})
    assert res_0_7["similarity"] == 0.0


def test_feature_correlation_matrix_tool_known_answer_19():
    """Test feature_correlation_matrix_tool against known answer #19 in TEST_FIXTURE.md."""
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

    exploration_tools = make_exploration_tools(session)
    feature_correlation_tool = next(t for t in exploration_tools if t.name == "feature_correlation_matrix_tool")

    res = feature_correlation_tool.invoke({"top_n": 20})

    expected_term_order = ["alpha", "gamma", "delta", "beta", "always"]
    assert res["term_names"] == expected_term_order
    assert res["top_n_included"] == 5

    expected_corr = [
        [1.0000, -0.5718, -0.6882, 0.8885, 0.2601],
        [-0.5718, 1.0000, 0.8307, -0.6435, 0.3140],
        [-0.6882, 0.8307, 1.0000, -0.7746, 0.3780],
        [0.8885, -0.6435, -0.7746, 1.0000, 0.2928],
        [0.2601, 0.3140, 0.3780, 0.2928, 1.0000],
    ]

    actual_matrix = np.array(res["correlation_matrix"])
    expected_matrix = np.array(expected_corr)

    assert np.allclose(actual_matrix, expected_matrix, atol=1e-3)
    assert session.pending_figure is not None
