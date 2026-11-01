"""Tests for tools_data.py using sample_fixture.csv and known answers from TEST_FIXTURE.md."""

import numpy as np
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools as make_data_tools


def test_list_files_tool_known_answer_11():
    """Test list_files_tool against known answer #11 in TEST_FIXTURE.md.

    Known Answer #11 specs:
    - Subdirectory: "newdataset"
    - Returned directories: []
    - Returned files: ["Reddit-stock-sentiment.csv"]
    """
    session = SessionState()
    data_tools = make_data_tools(session)
    list_files_tool = next(t for t in data_tools if t.name == "list_files_tool")

    res = list_files_tool.invoke({"subdirectory": "newdataset"})

    assert res["subdirectory"] == "newdataset"
    assert res["directories"] == []
    assert res["files"] == ["Reddit-stock-sentiment.csv"]


def test_inspect_data_tool_custom_rows_and_missing_data():
    """Test inspect_data_tool with custom n_rows and verify empty text handling."""
    session = SessionState()

    # Load dataset
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_data_tools(session)
    inspect_data_tool = next(t for t in data_tools if t.name == "inspect_data_tool")

    # Test custom n_rows=3
    res_3 = inspect_data_tool.invoke({"n_rows": 3})
    assert res_3["total_rows"] == 8
    assert res_3["inspected_rows"] == 3
    assert len(res_3["sample_records"]) == 3
    assert res_3["sample_records"][0]["text"] == "always alpha alpha alpha beta"
    assert res_3["sample_records"][0]["category"] == "catA"

    # Test n_rows exceeding total rows (n_rows=10) to verify empty text handling on row 7
    res_10 = inspect_data_tool.invoke({"n_rows": 10})
    assert res_10["inspected_rows"] == 8
    assert res_10["sample_records"][7]["text"] == "<missing>"


def test_inspect_data_tool_unloaded_dataframe_error():
    """Test that inspect_data_tool raises ValueError when session.dataframe is None."""
    session = SessionState()
    data_tools = make_data_tools(session)
    inspect_data_tool = next(t for t in data_tools if t.name == "inspect_data_tool")

    with pytest.raises(ValueError, match="session.dataframe is not loaded"):
        inspect_data_tool.invoke({"n_rows": 5})


def test_check_missing_tool_known_answer_2():
    """Test check_missing_tool against known answer #2 in TEST_FIXTURE.md.

    Known Answer #2 specs:
    - Exactly 1 row has missing/empty text (row 7).
    """
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_data_tools(session)
    check_missing_tool = next(t for t in data_tools if t.name == "check_missing_tool")

    res = check_missing_tool.invoke({"text_column": "text"})

    assert res["total_documents"] == 8
    assert res["missing_count"] == 1
    assert res["missing_indices"] == [7]
    assert res["missing_percentage"] == 12.5


def test_check_duplicates_tool_known_answer_3():
    """Test check_duplicates_tool against known answer #3 in TEST_FIXTURE.md.

    Known Answer #3 specs:
    - 1 duplicate row detected (row 6 is exact copy of row 5).
    - Calling drop=True with keep=False drops EVERY copy, shrinking 8 rows to 6 rows.
    """
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_data_tools(session)
    check_duplicates_tool = next(t for t in data_tools if t.name == "check_duplicates_tool")

    # Audit without dropping
    res_audit = check_duplicates_tool.invoke({"drop": False})
    assert res_audit["total_documents_before"] == 8
    assert res_audit["duplicate_count"] == 1
    assert res_audit["total_documents_after"] == 8

    # Apply drop=True with keep=False
    res_drop = check_duplicates_tool.invoke({"drop": True})
    assert res_drop["total_documents_before"] == 8
    assert res_drop["duplicate_count"] == 1
    assert res_drop["total_documents_after"] == 6
    assert len(session.dataframe) == 6


def test_sample_data_tool_known_answer_13():
    """Test sample_data_tool against known answer #13 in TEST_FIXTURE.md.

    Known Answer #13 specs:
    - n=4, random_state=42 returns exactly rows [1, 5, 0, 7].
    """
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_data_tools(session)
    sample_data_tool = next(t for t in data_tools if t.name == "sample_data_tool")

    res = sample_data_tool.invoke({"n": 4, "random_state": 42})

    assert res["total_documents_before"] == 8
    assert res["sampled_count"] == 4
    assert res["sampled_indices"] == [1, 5, 0, 7]


def test_describe_data_tool_known_answer_18():
    """Test describe_data_tool against known answer #18 in TEST_FIXTURE.md."""
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_data_tools(session)
    describe_data_tool = next(t for t in data_tools if t.name == "describe_data_tool")

    res = describe_data_tool.invoke({})

    # Assert character length list
    assert res["text_lengths"] == [29, 23, 17, 30, 24, 18, 18, 0]

    # Assert overall describe stats
    overall = res["overall_describe"]
    assert overall["count"] == 8
    assert np.isclose(overall["mean"], 19.875, atol=1e-3)
    assert np.isclose(overall["std"], 9.4330, atol=1e-3)
    assert overall["min"] == 0
    assert overall["max"] == 30

    # Assert per-category describe stats
    per_cat = res["per_category_describe"]
    assert "catA" in per_cat
    assert "catB" in per_cat
    assert per_cat["catA"]["count"] == 4
    assert np.isclose(per_cat["catA"]["mean"], 17.25, atol=1e-3)
    assert per_cat["catB"]["count"] == 4
    assert np.isclose(per_cat["catB"]["mean"], 22.5, atol=1e-3)

    # Assert mandatory boxplot figure is set on session
    assert session.pending_figure is not None


def test_tokenize_tool_known_answer_10():
    """Test tokenize_tool against known answer #10 in TEST_FIXTURE.md."""
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_data_tools(session)
    tokenize_tool = next(t for t in data_tools if t.name == "tokenize_tool")

    res = tokenize_tool.invoke({"text_column": "text"})

    expected_unigrams = [
        ["always", "alpha", "alpha", "alpha", "beta"],
        ["always", "alpha", "alpha", "beta"],
        ["always", "alpha", "beta"],
        ["always", "gamma", "gamma", "gamma", "delta"],
        ["always", "gamma", "gamma", "delta"],
        ["always", "gamma", "delta"],
        ["always", "gamma", "delta"],
        [],  # Empty document edge case
    ]

    assert res["total_documents"] == 8
    assert session.dataframe["unigrams"].tolist() == expected_unigrams
    assert session.artifacts["unigrams"] == expected_unigrams
    # Explicitly check empty document case at index 7
    assert session.dataframe["unigrams"].iloc[7] == []
