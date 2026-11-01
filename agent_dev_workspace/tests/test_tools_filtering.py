"""Tests for tools_filtering.py using sample_fixture.csv and known answers from TEST_FIXTURE.md."""

import numpy as np
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_filtering import make_tools as make_filtering_tools


def test_variance_filter_tool_known_answer_5():
    """Test variance_filter_tool against known answer #5 in TEST_FIXTURE.md.

    Known Answer #5 specs:
    - Variances:
        alpha: 1.1875
        always: 0.109375
        beta: 0.234375
        delta: 0.25
        gamma: 1.109375
    - Threshold 0.15 keeps alpha, beta, delta, gamma and removes always.
    """
    # Create fresh session
    session = SessionState()

    # Load dataset
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Build DTM first
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Instantiate filtering tools
    filtering_tools = make_filtering_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")

    # Invoke variance filter with threshold=0.15
    res = variance_filter_tool.invoke({"threshold": 0.15})

    # Check calculated variance scores
    scores = res["variance_scores"]
    assert np.isclose(scores["alpha"], 1.1875, atol=1e-3)
    assert np.isclose(scores["always"], 0.109375, atol=1e-3)
    assert np.isclose(scores["beta"], 0.234375, atol=1e-3)
    assert np.isclose(scores["delta"], 0.25, atol=1e-3)
    assert np.isclose(scores["gamma"], 1.109375, atol=1e-3)

    # Check filtering results
    assert set(res["kept_terms"]) == {"alpha", "beta", "delta", "gamma"}
    assert res["removed_terms"] == ["always"]
    assert res["n_kept"] == 4
    assert res["n_removed"] == 1


def test_pearson_filter_tool_known_answer_6():
    """Test pearson_filter_tool against known answer #6 in TEST_FIXTURE.md for target_class='catB'.

    Known Answer #6 Pearson r specs:
    - alpha: -0.6882
    - always: 0.3780
    - beta: -0.7746
    - delta: 1.0000
    - gamma: 0.8307
    """
    # Fresh session
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

    filtering_tools = make_filtering_tools(session)
    pearson_filter_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")

    res = pearson_filter_tool.invoke({"target_class": "catB", "threshold": 0.0})

    scores = res["pearson_scores"]
    assert np.isclose(scores["alpha"], -0.6882, atol=1e-3)
    assert np.isclose(scores["always"], 0.3780, atol=1e-3)
    assert np.isclose(scores["beta"], -0.7746, atol=1e-3)
    assert np.isclose(scores["delta"], 1.0000, atol=1e-3)
    assert np.isclose(scores["gamma"], 0.8307, atol=1e-3)


def test_spearman_filter_tool_known_answer_6():
    """Test spearman_filter_tool against known answer #6 in TEST_FIXTURE.md for target_class='catB'.

    Known Answer #6 Spearman r specs:
    - alpha: -0.7500
    - always: 0.3780
    - beta: -0.7746
    - delta: 1.0000
    - gamma: 0.9363
    """
    # Fresh session
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

    filtering_tools = make_filtering_tools(session)
    spearman_filter_tool = next(t for t in filtering_tools if t.name == "spearman_filter_tool")

    res = spearman_filter_tool.invoke({"target_class": "catB", "threshold": 0.0})

    scores = res["spearman_scores"]
    assert np.isclose(scores["alpha"], -0.7500, atol=1e-3)
    assert np.isclose(scores["always"], 0.3780, atol=1e-3)
    assert np.isclose(scores["beta"], -0.7746, atol=1e-3)
    assert np.isclose(scores["delta"], 1.0000, atol=1e-3)
    assert np.isclose(scores["gamma"], 0.9363, atol=1e-3)
