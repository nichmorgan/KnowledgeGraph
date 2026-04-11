"""Tests for confidence threshold fallback — Concern #3a (TDD RED phase)."""

import logging

import pytest
from neo4j_graphrag.generation import GraphRAG
from unittest.mock import MagicMock

from app.services.supervisor_agent import SupervisorAgentService


def _build_rag_result(answer: str, score: float):
    """Create a mock RAG result with a single retriever item at the given score."""
    mock_result = MagicMock()
    mock_result.answer = answer
    item = MagicMock()
    item.content = "{'name': 'TestNode'}"
    item.metadata = {"score": score}
    mock_result.retriever_result.items = [item]
    return mock_result


@pytest.fixture
def supervisor_agent_with_threshold(mock_user_service, mock_graphrag):
    return SupervisorAgentService(
        user_service=mock_user_service,
        graph_rag=mock_graphrag,
        hint_agent=MagicMock(),
        confidence_threshold=0.6,
    )


@pytest.fixture
def supervisor_agent_with_rewrite(mock_user_service, mock_graphrag):
    return SupervisorAgentService(
        user_service=mock_user_service,
        graph_rag=mock_graphrag,
        hint_agent=MagicMock(),
        rewrite_agent=MagicMock(),
        confidence_threshold=0.6,
    )


@pytest.fixture
def supervisor_agent_with_rewrite_and_content_rag(
    mock_user_service, mock_graphrag, mocker
):
    mock_content_rag = mocker.MagicMock(spec=GraphRAG)
    return SupervisorAgentService(
        user_service=mock_user_service,
        graph_rag=mock_graphrag,
        hint_agent=MagicMock(),
        rewrite_agent=MagicMock(),
        content_rag=mock_content_rag,
        confidence_threshold=0.6,
    )


@pytest.fixture
def _full_user_setup(mock_user_service, user_trajectory):
    """Sets up user + trajectory mocks for tests that complete the full retrieve_context flow."""
    mock_user = MagicMock()
    mock_user.id = "u1"
    mock_user_service.get_user.side_effect = lambda uid: mock_user
    mock_user_service.add_trajectory_entry.side_effect = lambda *a: user_trajectory
    mock_user_service.get_user_trajectory_by_query_exact_match.side_effect = (
        lambda *a, **kw: []
    )
    mock_user_service.get_user_trajectory_by_query_similarity.side_effect = (
        lambda *a, **kw: []
    )
    mock_user_service.get_user_trajectory.side_effect = lambda *a, **kw: []
    return mock_user_service


@pytest.fixture
def _happy_path_setup(mock_user_service, mock_graphrag, user_trajectory):
    """Common setup: valid user, RAG result with a retriever item, no repeated queries."""
    mock_user = MagicMock()
    mock_user.id = "u1"
    mock_user_service.get_user.side_effect = lambda uid: mock_user
    mock_user_service.add_trajectory_entry.side_effect = lambda *a: user_trajectory

    mock_rag_result = MagicMock()
    mock_rag_result.answer = "Normal answer about the topic"

    item = MagicMock()
    item.content = "{'name': 'TestNode'}"

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
    t_hist.query = "unrelated past query"
    mock_user_service.get_user_trajectory.side_effect = lambda *a, **kw: [t_hist]

    return item, mock_rag_result


def test_scores_above_threshold_returns_normal_answer(
    supervisor_agent_with_threshold, _happy_path_setup
):
    """When scores[0] >= confidence_threshold, normal answer is returned."""
    item, _ = _happy_path_setup
    item.metadata = {"score": 0.85}

    svc = supervisor_agent_with_threshold
    svc._SupervisorAgentService__hint_agent.run_sync.return_value = MagicMock(output="")

    res = svc.retrieve_context("u1", "what is symbolic execution?", "c1")

    assert res is not None
    assert res.answer == "Normal answer about the topic"


def test_scores_below_threshold_returns_fallback(
    supervisor_agent_with_threshold, _happy_path_setup
):
    """When scores[0] < confidence_threshold, fallback message is returned instead."""
    item, mock_rag_result = _happy_path_setup
    item.metadata = {"score": 0.3}

    svc = supervisor_agent_with_threshold
    svc._SupervisorAgentService__hint_agent.run_sync.return_value = MagicMock(output="")

    res = svc.retrieve_context("u1", "what is symbolic execution?", "c1")

    assert res is not None
    assert res.answer == SupervisorAgentService.RESPONSE_FALLBACK


def test_low_confidence_query_is_logged(
    supervisor_agent_with_threshold, _happy_path_setup, caplog
):
    """Low-confidence queries are logged for instructor review."""
    item, mock_rag_result = _happy_path_setup
    item.metadata = {"score": 0.1}

    svc = supervisor_agent_with_threshold
    svc._SupervisorAgentService__hint_agent.run_sync.return_value = MagicMock(output="")

    with caplog.at_level(logging.WARNING):
        res = svc.retrieve_context("u1", "obscure question", "c1")

    assert res is not None
    assert res.answer == SupervisorAgentService.RESPONSE_FALLBACK
    assert any("Low confidence" in record.message for record in caplog.records)


def test_score_exactly_at_threshold_returns_normal_answer(
    supervisor_agent_with_threshold, _happy_path_setup
):
    """When scores[0] == confidence_threshold exactly, normal answer is returned."""
    item, _ = _happy_path_setup
    item.metadata = {"score": 0.6}

    svc = supervisor_agent_with_threshold
    svc._SupervisorAgentService__hint_agent.run_sync.return_value = MagicMock(output="")

    res = svc.retrieve_context("u1", "test query", "c1")

    assert res is not None
    assert res.answer == "Normal answer about the topic"


def test_no_scores_returns_normal_answer(
    supervisor_agent_with_threshold, mock_user_service, mock_graphrag, user_trajectory
):
    """When retriever returns no items (empty scores), normal answer is returned."""
    mock_user = MagicMock()
    mock_user.id = "u1"
    mock_user_service.get_user.side_effect = lambda uid: mock_user
    mock_user_service.add_trajectory_entry.side_effect = lambda *a: user_trajectory

    mock_rag_result = MagicMock()
    mock_rag_result.answer = "Answer with no retriever results"
    mock_rag_result.retriever_result.items = []
    mock_graphrag.search.side_effect = lambda *a, **kw: mock_rag_result

    mock_user_service.get_user_trajectory_by_query_exact_match.side_effect = (
        lambda *a, **kw: []
    )
    mock_user_service.get_user_trajectory_by_query_similarity.side_effect = (
        lambda *a, **kw: []
    )
    mock_user_service.get_user_trajectory.side_effect = lambda *a, **kw: []

    svc = supervisor_agent_with_threshold
    svc._SupervisorAgentService__hint_agent.run_sync.return_value = MagicMock(output="")

    res = svc.retrieve_context("u1", "query", "c1")

    assert res is not None
    assert res.answer == "Answer with no retriever results"


# ---------------------------------------------------------------------------
# 3-stage retry pipeline tests
# ---------------------------------------------------------------------------


def test_stage1_rewrite_succeeds_returns_non_fallback(
    supervisor_agent_with_rewrite, _full_user_setup, mock_graphrag
):
    """Stage 1: when the rewritten query scores above threshold, a valid answer is returned."""
    mock_graphrag.search.side_effect = [
        _build_rag_result("Low-confidence answer", 0.3),
        _build_rag_result("Precise rewritten answer", 0.85),
    ]

    svc = supervisor_agent_with_rewrite
    mock_rewrite_out = MagicMock()
    mock_rewrite_out.output = "precise technical query for KG retrieval"
    svc._SupervisorAgentService__rewrite_agent.run_sync.return_value = mock_rewrite_out
    svc._SupervisorAgentService__hint_agent.run_sync.return_value = MagicMock(output="")

    res = svc.retrieve_context("u1", "how does it work?", "c1")

    assert res is not None
    assert res.answer != SupervisorAgentService.RESPONSE_FALLBACK


def test_stage2_content_rag_succeeds_returns_non_fallback(
    supervisor_agent_with_rewrite_and_content_rag, _full_user_setup, mock_graphrag
):
    """Stage 2: when rewrite is still low-confidence but content_rag scores above threshold, a valid answer is returned."""
    mock_graphrag.search.side_effect = [
        _build_rag_result("Original low answer", 0.3),
        _build_rag_result("Rewrite also low", 0.4),
    ]
    content_rag = supervisor_agent_with_rewrite_and_content_rag._SupervisorAgentService__content_rag
    content_rag.search.side_effect = lambda *a, **kw: _build_rag_result(
        "Content chunk answer", 0.9
    )

    svc = supervisor_agent_with_rewrite_and_content_rag
    mock_rewrite_out = MagicMock()
    mock_rewrite_out.output = "rewritten technical query"
    svc._SupervisorAgentService__rewrite_agent.run_sync.return_value = mock_rewrite_out
    svc._SupervisorAgentService__hint_agent.run_sync.return_value = MagicMock(output="")

    res = svc.retrieve_context("u1", "how does it work?", "c1")

    assert res is not None
    assert res.answer != SupervisorAgentService.RESPONSE_FALLBACK


def test_stage3_all_attempts_fail_returns_fallback(
    supervisor_agent_with_rewrite_and_content_rag, mock_user_service, mock_graphrag
):
    """Stage 3: when all retrieval attempts remain below threshold, fallback is returned."""
    mock_user = MagicMock()
    mock_user.id = "u1"
    mock_user_service.get_user.side_effect = lambda uid: mock_user

    mock_graphrag.search.side_effect = [
        _build_rag_result("Low answer", 0.2),
        _build_rag_result("Rewrite also low", 0.25),
    ]
    content_rag = supervisor_agent_with_rewrite_and_content_rag._SupervisorAgentService__content_rag
    content_rag.search.side_effect = lambda *a, **kw: _build_rag_result(
        "Content RAG also low", 0.3
    )

    svc = supervisor_agent_with_rewrite_and_content_rag
    mock_rewrite_out = MagicMock()
    mock_rewrite_out.output = "rewritten query that still misses"
    svc._SupervisorAgentService__rewrite_agent.run_sync.return_value = mock_rewrite_out

    res = svc.retrieve_context("u1", "completely obscure query", "c1")

    assert res is not None
    assert res.answer == SupervisorAgentService.RESPONSE_FALLBACK


def test_rewrite_low_without_content_rag_returns_fallback(
    supervisor_agent_with_rewrite, mock_user_service, mock_graphrag
):
    """When rewrite is still low-confidence and no content_rag is configured, fallback is returned."""
    mock_user = MagicMock()
    mock_user.id = "u1"
    mock_user_service.get_user.side_effect = lambda uid: mock_user

    mock_graphrag.search.side_effect = [
        _build_rag_result("Low original answer", 0.3),
        _build_rag_result("Low rewritten answer", 0.4),
    ]

    svc = supervisor_agent_with_rewrite
    mock_rewrite_out = MagicMock()
    mock_rewrite_out.output = "rewritten query still below threshold"
    svc._SupervisorAgentService__rewrite_agent.run_sync.return_value = mock_rewrite_out

    res = svc.retrieve_context("u1", "unclear question", "c1")

    assert res is not None
    assert res.answer == SupervisorAgentService.RESPONSE_FALLBACK
