"""Tests for tools_patterns.py using sample_fixture.csv and known answers from TEST_FIXTURE.md."""

import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools as make_patterns_tools


def test_mine_patterns_fpgrowth_known_answer_17():
    """Test mine_patterns_tool with algorithm='fpgrowth' against known answer #17 in TEST_FIXTURE.md."""
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    patterns_tools = make_patterns_tools(session)
    mine_tool = next(t for t in patterns_tools if t.name == "mine_patterns_tool")

    # Test tfidf filtering on catA
    res_catA = mine_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })

    assert res_catA["pattern_count"] == 1
    assert res_catA["patterns"][0]["pattern"] == ["alpha"]
    assert res_catA["patterns"][0]["support"] == 3

    # Test tfidf filtering on catB
    res_catB = mine_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })

    assert res_catB["pattern_count"] == 1
    assert res_catB["patterns"][0]["pattern"] == ["gamma"]
    assert res_catB["patterns"][0]["support"] == 4

    # Test variance filtering edge case (0 terms survive)
    res_var = mine_tool.invoke({
        "category_name": "catA",
        "filtering_method": "variance",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })

    assert res_var["patterns"] == []
    assert res_var["note"] == "no terms survived filtering"


def test_mine_patterns_topk_and_maxfpgrowth_known_answer_20():
    """Test mine_patterns_tool with topk and maxfpgrowth against known answer #20 in TEST_FIXTURE.md."""
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    patterns_tools = make_patterns_tools(session)
    mine_tool = next(t for t in patterns_tools if t.name == "mine_patterns_tool")

    # Test algorithm="topk" on catA
    res_topk = mine_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 1
    })

    assert res_topk["pattern_count"] == 1
    assert res_topk["patterns"][0]["pattern"] == ["alpha"]
    assert res_topk["patterns"][0]["support"] == 3

    # Test algorithm="maxfpgrowth" on catA (returns 0 patterns per Known Answer #20)
    res_max = mine_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "maxfpgrowth",
        "min_sup": 1
    })

    assert res_max["pattern_count"] == 0
    assert res_max["patterns"] == []
