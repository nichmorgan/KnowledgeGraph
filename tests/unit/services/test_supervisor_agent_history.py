"""Tests for conversation history threading into rewrite/hint context."""

import pytest
from neo4j_graphrag.types import LLMMessage

from app.services.supervisor_agent import SupervisorAgentService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_history(*pairs: tuple[str, str]) -> list[LLMMessage]:
    """Build a list of LLM messages from (role, content) pairs."""
    return [LLMMessage(role=role, content=content) for role, content in pairs]


# ---------------------------------------------------------------------------
# __format_history
# ---------------------------------------------------------------------------


class TestFormatHistory:
    _fmt = staticmethod(SupervisorAgentService._SupervisorAgentService__format_history)

    def test_none_returns_empty_string(self):
        assert self._fmt(None) == ""

    def test_empty_list_returns_empty_string(self):
        assert self._fmt([]) == ""

    def test_user_role_labeled_as_student(self):
        msgs = _make_history(("user", "What is recursion?"))
        result = self._fmt(msgs)
        assert result == "Student: What is recursion?"

    def test_assistant_role_labeled_as_tutor(self):
        msgs = _make_history(("assistant", "Recursion is a function calling itself."))
        result = self._fmt(msgs)
        assert result == "Tutor: Recursion is a function calling itself."

    def test_multiple_messages_joined_by_newlines(self):
        msgs = _make_history(
            ("user", "What is recursion?"),
            ("assistant", "It is a function calling itself."),
            ("user", "Can you give an example?"),
        )
        result = self._fmt(msgs)
        expected = (
            "Student: What is recursion?\n"
            "Tutor: It is a function calling itself.\n"
            "Student: Can you give an example?"
        )
        assert result == expected

    def test_unknown_role_defaults_to_tutor_label(self):
        # Non-user roles fall through to the "Tutor" label; content must still appear
        msg = LLMMessage(role="system", content="System prompt here.")
        result = self._fmt([msg])
        assert "System prompt here." in result
        assert result.startswith("Tutor:")


# ---------------------------------------------------------------------------
# __rewrite_query_for_retrieval + message_history
# ---------------------------------------------------------------------------


@pytest.fixture
def svc_with_rewriter(mocker, mock_user_service, mock_graphrag):
    return SupervisorAgentService(
        user_service=mock_user_service,
        graph_rag=mock_graphrag,
        hint_agent=mocker.MagicMock(),
        rewrite_agent=mocker.MagicMock(),
    )


class TestRewriteQueryHistory:
    """Call __rewrite_query_for_retrieval directly to inspect the prompt."""

    def test_history_context_included_in_rewrite_prompt(
        self, mocker, svc_with_rewriter
    ):
        mock_out = mocker.MagicMock()
        mock_out.output = "rewritten query"
        svc_with_rewriter._SupervisorAgentService__rewrite_agent.run_sync.return_value = mock_out

        history = _make_history(
            ("user", "What is a binary tree?"),
            ("assistant", "A tree where each node has at most two children."),
        )

        svc_with_rewriter._SupervisorAgentService__rewrite_query_for_retrieval(
            "and traversal?", history
        )

        call_args = (
            svc_with_rewriter._SupervisorAgentService__rewrite_agent.run_sync.call_args
        )
        prompt = call_args[0][0]
        assert "Recent conversation:" in prompt
        assert "Student: What is a binary tree?" in prompt
        assert "Tutor: A tree where each node" in prompt

    def test_no_history_omits_context_block_in_rewrite_prompt(
        self, mocker, svc_with_rewriter
    ):
        mock_out = mocker.MagicMock()
        mock_out.output = "rewritten query"
        svc_with_rewriter._SupervisorAgentService__rewrite_agent.run_sync.return_value = mock_out

        svc_with_rewriter._SupervisorAgentService__rewrite_query_for_retrieval(
            "and traversal?", None
        )

        call_args = (
            svc_with_rewriter._SupervisorAgentService__rewrite_agent.run_sync.call_args
        )
        prompt = call_args[0][0]
        assert "Recent conversation:" not in prompt

    def test_returns_original_query_when_no_rewrite_agent(
        self, mocker, mock_user_service, mock_graphrag
    ):
        svc = SupervisorAgentService(
            user_service=mock_user_service,
            graph_rag=mock_graphrag,
            hint_agent=mocker.MagicMock(),
        )
        history = _make_history(("user", "previous question"))
        result = svc._SupervisorAgentService__rewrite_query_for_retrieval(
            "my query", history
        )
        assert result == "my query"


# ---------------------------------------------------------------------------
# __generate_hint + message_history
# ---------------------------------------------------------------------------


class TestGenerateHintHistory:
    """Call __generate_hint directly to inspect hint prompts."""

    def _make_svc(
        self,
        mocker,
        mock_user_service,
        mock_graphrag,
        *,
        hint_by_similarity_threshold=2,
    ):
        return SupervisorAgentService(
            user_service=mock_user_service,
            graph_rag=mock_graphrag,
            hint_agent=mocker.MagicMock(),
            hint_by_similarity_threshold=hint_by_similarity_threshold,
        )

    def test_repeated_query_hint_includes_history(
        self, mocker, mock_user_service, mock_graphrag
    ):
        svc = self._make_svc(mocker, mock_user_service, mock_graphrag)
        hint_out = mocker.MagicMock()
        hint_out.output = "Here is your hint."
        svc._SupervisorAgentService__hint_agent.run_sync.return_value = hint_out

        history = _make_history(
            ("user", "How does sorting work?"),
            ("assistant", "It reorders elements based on a comparator."),
        )

        # query_repeat_count >= hint_by_similarity_threshold triggers repeated query hint
        svc._SupervisorAgentService__generate_hint(
            "explain sorting",
            2,
            retrieved_nodes=["SortingNode"],
            user_id="u1",
            message_history=history,
        )

        call_args = svc._SupervisorAgentService__hint_agent.run_sync.call_args
        prompt = call_args[0][0]
        assert "Recent conversation:" in prompt
        assert "Student: How does sorting work?" in prompt

    def test_repeated_query_hint_no_history_omits_block(
        self, mocker, mock_user_service, mock_graphrag
    ):
        svc = self._make_svc(mocker, mock_user_service, mock_graphrag)
        hint_out = mocker.MagicMock()
        hint_out.output = "Here is your hint."
        svc._SupervisorAgentService__hint_agent.run_sync.return_value = hint_out

        svc._SupervisorAgentService__generate_hint(
            "explain sorting",
            2,
            retrieved_nodes=["SortingNode"],
            user_id="u1",
            message_history=None,
        )

        call_args = svc._SupervisorAgentService__hint_agent.run_sync.call_args
        prompt = call_args[0][0]
        assert "Recent conversation:" not in prompt

    def test_procedural_impasse_hint_includes_history(
        self, mocker, mock_user_service, mock_graphrag
    ):
        svc = SupervisorAgentService(
            user_service=mock_user_service,
            graph_rag=mock_graphrag,
            hint_agent=mocker.MagicMock(),
            hint_procedural_history_limit=2,
        )
        # Prior procedural history
        t1, t2 = mocker.MagicMock(), mocker.MagicMock()
        t1.query = "how to run the script"
        t2.query = "how to fix the error"
        mock_user_service.get_user_trajectory.side_effect = lambda *a, **kw: [t1, t2]

        hint_out = mocker.MagicMock()
        hint_out.output = "Think about the concept."
        svc._SupervisorAgentService__hint_agent.run_sync.return_value = hint_out

        history = _make_history(
            ("user", "how to run the script"),
            ("assistant", "You can use python main.py."),
        )

        # repeat_count=0 so repeated-query branch is skipped; procedural impasse takes over
        svc._SupervisorAgentService__generate_hint(
            "how to execute this code",
            0,
            retrieved_nodes=["CodeNode"],
            user_id="u1",
            message_history=history,
        )

        call_args = svc._SupervisorAgentService__hint_agent.run_sync.call_args
        prompt = call_args[0][0]
        assert "Recent conversation:" in prompt
        assert "Student: how to run the script" in prompt

    def test_procedural_impasse_hint_no_history_omits_block(
        self, mocker, mock_user_service, mock_graphrag
    ):
        svc = SupervisorAgentService(
            user_service=mock_user_service,
            graph_rag=mock_graphrag,
            hint_agent=mocker.MagicMock(),
            hint_procedural_history_limit=2,
        )
        t1, t2 = mocker.MagicMock(), mocker.MagicMock()
        t1.query = "how to run the script"
        t2.query = "how to fix the error"
        mock_user_service.get_user_trajectory.side_effect = lambda *a, **kw: [t1, t2]

        hint_out = mocker.MagicMock()
        hint_out.output = "Think about the concept."
        svc._SupervisorAgentService__hint_agent.run_sync.return_value = hint_out

        svc._SupervisorAgentService__generate_hint(
            "how to execute this code",
            0,
            retrieved_nodes=["CodeNode"],
            user_id="u1",
            message_history=None,
        )

        call_args = svc._SupervisorAgentService__hint_agent.run_sync.call_args
        prompt = call_args[0][0]
        assert "Recent conversation:" not in prompt


# ---------------------------------------------------------------------------
# __rewrite_response + message_history
# ---------------------------------------------------------------------------


class TestRewriteResponseHistory:
    """Call __rewrite_response directly to inspect the prompt."""

    def test_history_included_in_rewrite_response_prompt(
        self, mocker, svc_with_rewriter
    ):
        mock_out = mocker.MagicMock()
        mock_out.output = "Clean student-friendly answer."
        svc_with_rewriter._SupervisorAgentService__rewrite_agent.run_sync.return_value = mock_out

        history = _make_history(
            ("user", "What is a stack?"),
            ("assistant", "A LIFO data structure."),
        )

        svc_with_rewriter._SupervisorAgentService__rewrite_response(
            "Raw answer about stack.", "how do I use it?", history
        )

        call_args = (
            svc_with_rewriter._SupervisorAgentService__rewrite_agent.run_sync.call_args
        )
        prompt = call_args[0][0]
        assert "Recent conversation" in prompt
        assert "Student: What is a stack?" in prompt
        assert "Tutor: A LIFO data structure." in prompt

    def test_no_history_omits_context_block_in_rewrite_response(
        self, mocker, svc_with_rewriter
    ):
        mock_out = mocker.MagicMock()
        mock_out.output = "Clean student-friendly answer."
        svc_with_rewriter._SupervisorAgentService__rewrite_agent.run_sync.return_value = mock_out

        svc_with_rewriter._SupervisorAgentService__rewrite_response(
            "Raw answer.", "what is a stack?", None
        )

        call_args = (
            svc_with_rewriter._SupervisorAgentService__rewrite_agent.run_sync.call_args
        )
        prompt = call_args[0][0]
        assert "Recent conversation" not in prompt

    def test_rewritten_answer_returned_to_caller(self, mocker, svc_with_rewriter):
        expected = "Contextually-aware rewritten response."
        mock_out = mocker.MagicMock()
        mock_out.output = expected
        svc_with_rewriter._SupervisorAgentService__rewrite_agent.run_sync.return_value = mock_out

        history = _make_history(("user", "Prior question"))
        result = svc_with_rewriter._SupervisorAgentService__rewrite_response(
            "Raw answer.", "follow-up question", history
        )
        assert result == expected

    def test_empty_raw_answer_bypasses_rewrite(self, mocker, svc_with_rewriter):
        result = svc_with_rewriter._SupervisorAgentService__rewrite_response(
            "", "q", None
        )
        svc_with_rewriter._SupervisorAgentService__rewrite_agent.run_sync.assert_not_called()
        assert result == ""
