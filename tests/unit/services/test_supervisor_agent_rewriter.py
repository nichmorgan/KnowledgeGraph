"""Tests for response rewriter — Concern #4: KG Internals Exposed (TDD RED phase)."""

import pytest
from unittest.mock import MagicMock

from app.services.supervisor_agent import SupervisorAgentService


@pytest.fixture
def supervisor_agent_with_rewriter(mock_user_service, mock_graphrag):
    hint_agent = MagicMock()
    rewrite_agent = MagicMock()
    return SupervisorAgentService(
        user_service=mock_user_service,
        graph_rag=mock_graphrag,
        hint_agent=hint_agent,
        rewrite_agent=rewrite_agent,
    )


@pytest.fixture
def _happy_path_mocks(mock_user_service, mock_graphrag, user_trajectory):
    mock_user = MagicMock()
    mock_user.id = "u1"
    mock_user_service.get_user.side_effect = lambda uid: mock_user
    mock_user_service.add_trajectory_entry.side_effect = lambda *a: user_trajectory

    mock_rag_result = MagicMock()
    mock_rag_result.answer = (
        "Based on the knowledge graph node 'angr_framework', "
        "the concept graph shows a PREREQUISITE_FOR relationship "
        "to 'symbolic_execution'."
    )
    item = MagicMock()
    item.content = "{'name': 'angr_framework'}"
    item.metadata = {"score": 0.9}
    mock_rag_result.retriever_result.items = [item]
    mock_graphrag.search.side_effect = lambda *a, **kw: mock_rag_result

    t1 = MagicMock()
    t1.id = "t1"
    mock_user_service.get_user_trajectory_by_query_exact_match.side_effect = (
        lambda *a, **kw: [t1]
    )
    mock_user_service.get_user_trajectory_by_query_similarity.side_effect = (
        lambda *a, **kw: []
    )
    t_hist = MagicMock()
    t_hist.query = "how to run the code"
    mock_user_service.get_user_trajectory.side_effect = lambda *a, **kw: [t_hist]
    return mock_user_service, mock_graphrag


def test_rewriter_strips_kg_terms(supervisor_agent_with_rewriter, _happy_path_mocks):
    rewritten = (
        "Angr is a Python framework for binary analysis. "
        "Before using it, you should understand symbolic execution."
    )
    svc = supervisor_agent_with_rewriter
    mock_out = MagicMock()
    mock_out.output = rewritten
    svc._SupervisorAgentService__rewrite_agent.run_sync.return_value = mock_out
    mock_hint_out = MagicMock()
    mock_hint_out.output = ""
    svc._SupervisorAgentService__hint_agent.run_sync.return_value = mock_hint_out

    res = svc.retrieve_context("u1", "what is angr?", "c1")
    assert res is not None
    assert res.answer == rewritten
    for term in ("knowledge graph", "node", "concept graph", "retrieved nodes"):
        assert term.lower() not in res.answer.lower()


def test_raw_answer_preserved_in_trajectory(
    supervisor_agent_with_rewriter, _happy_path_mocks
):
    svc = supervisor_agent_with_rewriter
    raw = (
        "Based on the knowledge graph node 'angr_framework', "
        "the concept graph shows a PREREQUISITE_FOR relationship "
        "to 'symbolic_execution'."
    )
    rewritten = "X is related to Y in this context."

    mock_out = MagicMock()
    mock_out.output = rewritten
    svc._SupervisorAgentService__rewrite_agent.run_sync.return_value = mock_out
    mock_hint_out = MagicMock()
    mock_hint_out.output = ""
    svc._SupervisorAgentService__hint_agent.run_sync.return_value = mock_hint_out

    captured = {}

    def _capture_traj(*a):
        captured["traj"] = a[1] if len(a) > 1 else a[0]
        return MagicMock()

    _happy_path_mocks[0].add_trajectory_entry.side_effect = _capture_traj
    svc.retrieve_context("u1", "q", "c1")

    traj = captured.get("traj")
    assert traj is not None, "Trajectory was not captured"
    assert hasattr(traj, "raw_answer"), "Trajectory missing raw_answer field"
    assert traj.raw_answer == raw


def test_rewriter_called_before_returning_result(
    supervisor_agent_with_rewriter, _happy_path_mocks
):
    rewritten = "This is a clean, student-friendly answer."
    svc = supervisor_agent_with_rewriter
    mock_out = MagicMock()
    mock_out.output = rewritten
    svc._SupervisorAgentService__rewrite_agent.run_sync.return_value = mock_out
    mock_hint_out = MagicMock()
    mock_hint_out.output = ""
    svc._SupervisorAgentService__hint_agent.run_sync.return_value = mock_hint_out

    res = svc.retrieve_context("u1", "explain", "c1")
    assert res is not None
    assert res.answer == rewritten
    svc._SupervisorAgentService__rewrite_agent.run_sync.assert_called_once()


def test_rewriter_not_called_when_no_answer(
    supervisor_agent_with_rewriter, mock_user_service, mock_graphrag
):
    svc = supervisor_agent_with_rewriter
    mock_user = MagicMock()
    mock_user.id = "u1"
    mock_user_service.get_user.side_effect = lambda uid: mock_user

    mock_rag_result = MagicMock()
    mock_rag_result.answer = ""
    mock_rag_result.retriever_result.items = []
    mock_graphrag.search.side_effect = lambda *a, **kw: mock_rag_result

    mock_user_service.get_user_trajectory_by_query_exact_match.side_effect = (
        lambda *a, **kw: []
    )
    mock_user_service.get_user_trajectory_by_query_similarity.side_effect = (
        lambda *a, **kw: []
    )
    mock_user_service.get_user_trajectory.side_effect = lambda *a, **kw: []
    mock_user_service.add_trajectory_entry.side_effect = lambda *a: MagicMock()
    mock_hint_out = MagicMock()
    mock_hint_out.output = ""
    svc._SupervisorAgentService__hint_agent.run_sync.return_value = mock_hint_out

    res = svc.retrieve_context("u1", "q", "c1")
    assert res is not None
    assert res.answer == ""
    svc._SupervisorAgentService__rewrite_agent.run_sync.assert_not_called()


def test_backward_compat_no_rewrite_agent(
    mock_user_service, mock_graphrag, user_trajectory
):
    svc = SupervisorAgentService(
        user_service=mock_user_service,
        graph_rag=mock_graphrag,
        hint_agent=MagicMock(),
    )
    mock_user = MagicMock()
    mock_user.id = "u1"
    mock_user_service.get_user.side_effect = lambda uid: mock_user
    mock_user_service.add_trajectory_entry.side_effect = lambda *a: user_trajectory

    raw_answer = "Knowledge graph node 'X' shows concept 'Y'."
    mock_rag_result = MagicMock()
    mock_rag_result.answer = raw_answer
    mock_rag_result.retriever_result.items = []
    mock_graphrag.search.side_effect = lambda *a, **kw: mock_rag_result

    mock_user_service.get_user_trajectory_by_query_exact_match.side_effect = (
        lambda *a, **kw: []
    )
    mock_user_service.get_user_trajectory_by_query_similarity.side_effect = (
        lambda *a, **kw: []
    )
    mock_user_service.get_user_trajectory.side_effect = lambda *a, **kw: []
    mock_hint_out = MagicMock()
    mock_hint_out.output = ""
    svc._SupervisorAgentService__hint_agent.run_sync.return_value = mock_hint_out

    res = svc.retrieve_context("u1", "q", "c1")
    assert res is not None
    assert res.answer == raw_answer
