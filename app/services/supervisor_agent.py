import logging
import re
import time
from pydantic import BaseModel

from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.types import LLMMessage
from pydantic_ai import Agent

from app import models
from enum import StrEnum

from .user import UserService


class SupervisorResult(BaseModel):
    answer: str
    hint_text: str | None = None
    hint_reason: str | None = None


class Intent(StrEnum):
    DEFINITION = "definition"
    PROCEDURAL = "procedural"
    TROUBLESHOOTING = "troubleshooting"
    EXAMPLE_REQUEST = "example_request"
    CONTEXT_REQUEST = "context_request"


class SupervisorAgentService:
    RESPONSE_FALLBACK = "I couldn't find relevant information to answer your question. Please try rephrasing."

    INTENT_KEYWORDS: dict[Intent, list[str]] = {
        Intent.DEFINITION: [
            "what is",
            "what are",
            "explain",
            "define",
            "definition",
            "meaning of",
            "describe",
        ],
        Intent.PROCEDURAL: [
            "how to",
            "how do",
            "steps to",
            "implement",
            "procedure",
            "how can i",
            "what steps",
        ],
        Intent.TROUBLESHOOTING: [
            "error in",
            "not working",
            "fix",
            "bug",
            "issue",
            "broken",
            "fails",
            "exception",
            "debug",
        ],
        Intent.EXAMPLE_REQUEST: [
            "example of",
            "show me",
            "example",
            "sample",
            "demo",
            "give me an example",
        ],
    }

    INTENT_TO_PREFERRED_TYPES: dict[Intent, list[models.KnowledgeType]] = {
        Intent.DEFINITION: [models.KnowledgeType.CONCEPTUAL],
        Intent.PROCEDURAL: [models.KnowledgeType.PROCEDURAL],
        Intent.TROUBLESHOOTING: [models.KnowledgeType.PROCEDURAL],
        Intent.EXAMPLE_REQUEST: [models.KnowledgeType.ASSESSMENT],
        Intent.CONTEXT_REQUEST: [
            models.KnowledgeType.CONCEPTUAL,
            models.KnowledgeType.PROCEDURAL,
            models.KnowledgeType.ASSESSMENT,
        ],
    }

    DIFFICULTY_ORDER: dict[models.KnowledgeDifficulty, int] = {
        models.KnowledgeDifficulty.EASY: 0,
        models.KnowledgeDifficulty.MEDIUM: 1,
        models.KnowledgeDifficulty.HARD: 2,
    }

    def __init__(
        self,
        *,
        user_service: UserService,
        graph_rag: GraphRAG,
        hint_agent: Agent,
        rewrite_agent: Agent | None = None,
        content_rag: GraphRAG | None = None,
        top_k: int = 5,
        similarity_threshold: float = 0.85,
        hint_by_similarity_threshold: int = 2,
        hint_procedural_history_limit: int = 3,
        confidence_threshold: float = 0.6,
        procedural_keywords: list[str] = [
            "run",
            "fix",
            "load",
            "execute",
            "implement",
            "solve",
            "compile",
            "test",
            "error",
            "code",
        ],
    ):
        self.__user_service = user_service
        self.__graph_rag = graph_rag
        self.__content_rag = content_rag
        self.__hint_agent = hint_agent
        self.__rewrite_agent = rewrite_agent
        self.__top_k = top_k
        self.__similarity_threshold = similarity_threshold
        self.__hint_by_similarity_threshold = hint_by_similarity_threshold
        self.__hint_procedural_history_limit = hint_procedural_history_limit
        self.__procedural_keywords = procedural_keywords
        self.__confidence_threshold = confidence_threshold
        self.__logger = logging.getLogger(__name__)

    @staticmethod
    def __format_history(message_history: list[LLMMessage] | None) -> str:
        """Render the last N exchanges as plain text for inclusion in LLM prompts."""
        if not message_history:
            return ""
        lines = []
        for msg in message_history:
            role = "Student" if msg.role == "user" else "Tutor"
            lines.append(f"{role}: {msg.content}")
        return "\n".join(lines)

    def __rewrite_query_for_retrieval(
        self, query: str, message_history: list[LLMMessage] | None = None
    ) -> str:
        """Rephrase an informal student question into precise technical terms for KG retrieval."""
        if not self.__rewrite_agent:
            return query
        history_block = self.__format_history(message_history)
        context = f"\n\nRecent conversation:\n{history_block}" if history_block else ""
        prompt = (
            "Rephrase the following student question into precise technical terms "
            "suitable for searching a knowledge graph with domain-specific node labels. "
            "Return only the rephrased query with no explanation."
            f"{context}\n\n"
            f"Student question: {query}"
        )
        rephrased = self.__rewrite_agent.run_sync(prompt).output.strip()
        self.__logger.info(f"Query rewritten for retrieval: '{query}' → '{rephrased}'")
        return rephrased

    def __retrieve_node_metadata(
        self,
        query: str,
        message_history: list[LLMMessage] | None = None,
        *,
        graph_rag: GraphRAG | None = None,
    ):
        rag = graph_rag or self.__graph_rag
        start_time = time.time()
        result = rag.search(
            query,
            message_history=message_history,
            retriever_config={"top_k": self.__top_k},
            return_context=True,
            response_fallback=self.RESPONSE_FALLBACK,
        )
        response_time_sec = round(time.time() - start_time, 2)
        self.__logger.info(f"Context retrieved in {response_time_sec} seconds")

        self.__logger.info("Parsing retriever results for node metadata...")
        retrieved_nodes, scores = [], []
        if retriever_result := result.retriever_result:
            try:
                for item in retriever_result.items:
                    node_name = "Unknown"
                    if isinstance(item.content, str):
                        match = re.search(r"'name': '([^']+)'", item.content)
                        if match:
                            node_name = match.group(1)
                    retrieved_nodes.append(node_name)
                    if isinstance(item.metadata, dict):
                        scores.append(item.metadata.get("score"))
            except Exception as e:
                self.__logger.warning(f"Failed to parse retriever result metadata: {e}")

        self.__logger.info(f"Retrieved nodes: {len(retrieved_nodes)}")
        return result, retrieved_nodes, scores, response_time_sec

    def __classify_intent(self, query: str) -> str:
        """Classify student query intent into categories.

        Returns one of: definition, procedural, troubleshooting, example_request, or context_request.
        """
        q_lower = query.lower().strip()
        for intent, keywords in self.INTENT_KEYWORDS.items():
            if any(kw in q_lower for kw in keywords):
                return intent
        return Intent.CONTEXT_REQUEST

    def __get_interaction_type(self, query: str):
        q_lower = query.lower()
        keywords_mapping = {
            "hint_request": ["hint", "help", "assist", "clue", "suggest"],
            "code_request": ["code", "script", "function", "write", "implement"],
            "concept_request": [
                "concept",
                "explain",
                "definition",
                "understand",
                "what is",
            ],
        }

        for interaction_type, keywords in keywords_mapping.items():
            if any(word in q_lower for word in keywords):
                return interaction_type
        return Intent.CONTEXT_REQUEST

    def __get_similar_trajectory_queries(self, query: str, *, user_id: str):
        trajectories: list[str] = []

        # TODO: Why not only use the query_similarity as a exact match would be always present in the list?
        for item in self.__user_service.get_user_trajectory_by_query_exact_match(
            user_id, query
        ):
            if item.id and item.id not in trajectories:
                trajectories.append(item.id)

        for item in self.__user_service.get_user_trajectory_by_query_similarity(
            user_id, query, threshold=self.__similarity_threshold
        ):
            if item.id and item.id not in trajectories:
                trajectories.append(item.id)

        return list(trajectories)

    def __generate_hint(
        self,
        query: str,
        query_repeat_count: int,
        *,
        retrieved_nodes: list[str],
        user_id: str,
        message_history: list[LLMMessage] | None = None,
    ):
        hint_triggered = query_repeat_count >= self.__hint_by_similarity_threshold
        hint_reason = None
        hint_text = None
        history_block = self.__format_history(message_history)
        history_context = f"\n\nRecent conversation:\n{history_block}" if history_block else ""

        if hint_triggered:
            hint_reason = "Repeated query (possible confusion)"
            hint_prompt = (
                f"Provide a short, encouraging hint to help the student progress on: '{query}'. "
                f"Focus on conceptual reinforcement rather than giving the answer directly. "
                f"Context nodes: {retrieved_nodes[:3]}"
                f"{history_context}"
            )
            hint_text = self.__hint_agent.run_sync(hint_prompt).output.strip()
            self.__logger.info(f"Hint triggered: {hint_reason} -> {hint_text}")
        else:
            last_trajectories = self.__user_service.get_user_trajectory(
                user_id, limit=self.__hint_procedural_history_limit
            )
            recent_queries = [item.query.lower() for item in last_trajectories]
            recent_queries.append(query.lower())

            def is_procedural(q):
                return any(k in q for k in self.__procedural_keywords)

            if all(is_procedural(q) for q in recent_queries + [query.lower()]):
                hint_triggered = True
                hint_reason = "Procedural impasse (stuck on how-to steps)"
                hint_prompt = (
                    f"The student has been asking several procedural questions in a row. "
                    f"Generate a reflective, conceptual hint encouraging them to focus on the underlying idea of '{query}'. "
                    f"Do not reveal exact code; instead, suggest understanding the concept that supports this step."
                    f"{history_context}"
                )
                hint_text = self.__hint_agent.run_sync(hint_prompt).output.strip()
                self.__logger.info(f"Hint triggered: {hint_reason} -> {hint_text}")

        return hint_triggered, hint_reason, hint_text

    def __rewrite_response(
        self,
        raw_answer: str,
        query: str,
        message_history: list[LLMMessage] | None = None,
    ) -> str:
        if not self.__rewrite_agent or not raw_answer:
            return raw_answer

        history_block = self.__format_history(message_history)
        history_context = (
            f"\n\nRecent conversation (use this to tailor the tone and depth of your answer):\n{history_block}"
            if history_block
            else ""
        )
        prompt = (
            "You are a teaching assistant. Rewrite the answer below so it reads as a "
            "direct, natural response to the student's question.\n\n"
            "Rules:\n"
            "- Never mention: nodes, graph, knowledge graph, concept node, procedural node, "
            "assessment node, retrieval, scores, labels, edges, PREREQUISITE_FOR, DEPENDS_ON, "
            "EXTENDS_TO, ENABLES, or any other graph relationship names.\n"
            "- Never reference internal system structure or metadata.\n"
            "- Write as if you are a human tutor speaking directly to the student.\n"
            "- Keep all factual content accurate — only change the framing and language.\n"
            "- If the student has been asking about related topics, acknowledge that continuity naturally.\n"
            "- Be concise and clear.\n"
            f"{history_context}\n\n"
            f"Student question: {query}\n\n"
            f"Raw answer: {raw_answer}"
        )
        result = self.__rewrite_agent.run_sync(prompt)
        return result.output.strip()

    def __rerank_results(
        self,
        items: list,
        intent: str,
        user_id: str,
    ) -> list:
        """Re-rank retriever results based on intent, difficulty, and pedagogical sequence.

        Uses KG node type metadata from Neo4j to determine node types.
        Returns the same items, reordered by pedagogical relevance.
        """
        if not items:
            return items

        preferred_types = self.INTENT_TO_PREFERRED_TYPES.get(
            intent,
            [
                models.KnowledgeType.CONCEPTUAL,
                models.KnowledgeType.PROCEDURAL,
                models.KnowledgeType.ASSESSMENT,
            ],
        )

        def _extract_node_type(item) -> str:
            metadata = item.metadata or {}
            node_type = metadata.get("node_type", "")
            if node_type:
                return node_type.lower()
            item_content = item.content if isinstance(item.content, str) else ""
            for t in models.KnowledgeType:
                if f"'type': '{t}'" in item_content or f'"type": "{t}"' in item_content:
                    return t
            return ""

        def _extract_difficulty(item) -> str:
            metadata = item.metadata or {}
            difficulty = metadata.get("difficulty", "")
            if difficulty:
                return difficulty.lower()
            item_content = item.content if isinstance(item.content, str) else ""
            for d in models.KnowledgeDifficulty:
                if (
                    f"'difficulty': '{d}'" in item_content
                    or f'"difficulty": "{d}"' in item_content
                ):
                    return d
            return models.KnowledgeDifficulty.MEDIUM

        def _extract_score(item) -> float:
            metadata = item.metadata or {}
            return metadata.get("score", 0.0) or 0.0

        def _rerank_key(item) -> tuple:
            node_type = _extract_node_type(item)
            difficulty = _extract_difficulty(item)
            score = _extract_score(item)

            # 1. Intent match: lower is better (0 = best match)
            if node_type in preferred_types:
                type_rank = preferred_types.index(node_type)
            else:
                type_rank = len(preferred_types)

            # 2. Difficulty: lower is easier
            difficulty_rank = self.DIFFICULTY_ORDER.get(difficulty, 1)

            # 3. Pedagogical sequence: conceptual -> procedural -> assessment
            pedagogical_order = {
                models.KnowledgeType.CONCEPTUAL: 0,
                models.KnowledgeType.PROCEDURAL: 1,
                models.KnowledgeType.ASSESSMENT: 2,
            }
            pedagogical_rank = pedagogical_order.get(node_type, 3)

            return (type_rank, difficulty_rank, pedagogical_rank, -score)

        return sorted(items, key=_rerank_key)

    def retrieve_context(
        self,
        user_id: str,
        query: str,
        course_id: str,
        message_history: list[LLMMessage] | None = None,
    ):
        try:
            self.__logger.info(f"Loading user {user_id} state...")
            user = self.__user_service.get_user(user_id)
            if not user:
                self.__logger.warning(f"User {user_id} not found!")
                return None
            if not user.id:
                self.__logger.warning(f"User {user_id} has no ID!")
                return None

            self.__logger.info("Retrieving node metadata...")
            rag_result, retrieved_nodes, scores, response_time_sec = (
                self.__retrieve_node_metadata(query, message_history)
            )
            node_entry_count = len(retrieved_nodes)

            # Check confidence threshold; if low, rewrite query and retry
            if (
                scores
                and scores[0] is not None
                and scores[0] < self.__confidence_threshold
            ):
                self.__logger.warning(
                    f"Low confidence (score={scores[0]:.2f} < {self.__confidence_threshold:.2f}): '{query}'"
                )
                rewritten_query = self.__rewrite_query_for_retrieval(query, message_history)
                if rewritten_query != query:
                    rr, rn, rs, rt = self.__retrieve_node_metadata(rewritten_query, message_history)
                    if rs and rs[0] is not None and rs[0] >= self.__confidence_threshold:
                        self.__logger.info(f"Rewritten query succeeded (score={rs[0]:.2f})")
                        query = rewritten_query
                        rag_result, retrieved_nodes, scores, response_time_sec = rr, rn, rs, rt
                        node_entry_count = len(retrieved_nodes)
                    elif self.__content_rag:
                        self.__logger.info("Trajectory RAG low confidence; trying content chunk RAG...")
                        cr, cn, cs, ct = self.__retrieve_node_metadata(
                            rewritten_query, message_history, graph_rag=self.__content_rag
                        )
                        if cs and cs[0] is not None and cs[0] >= self.__confidence_threshold:
                            self.__logger.info(f"Content chunk RAG succeeded (score={cs[0]:.2f})")
                            query = rewritten_query
                            rag_result, retrieved_nodes, scores, response_time_sec = cr, cn, cs, ct
                            node_entry_count = len(retrieved_nodes)
                        else:
                            self.__logger.warning("All retrieval attempts below confidence threshold.")
                            return SupervisorResult(answer=self.RESPONSE_FALLBACK, hint_text=None, hint_reason=None)
                    else:
                        self.__logger.warning("Rewritten query still below confidence threshold.")
                        return SupervisorResult(answer=self.RESPONSE_FALLBACK, hint_text=None, hint_reason=None)
                else:
                    return SupervisorResult(answer=self.RESPONSE_FALLBACK, hint_text=None, hint_reason=None)

            self.__logger.info("Determining interaction type...")
            interaction_type = self.__classify_intent(query)

            # Re-rank retriever results based on intent, difficulty, and pedagogical sequence
            if rag_result.retriever_result and rag_result.retriever_result.items:
                rag_result.retriever_result.items = self.__rerank_results(
                    rag_result.retriever_result.items,
                    interaction_type,
                    user.id,
                )
                # Recompute node names and scores after re-ranking
                retrieved_nodes = []
                scores = []
                for item in rag_result.retriever_result.items:
                    node_name = "Unknown"
                    if isinstance(item.content, str):
                        match = re.search(r"'name': '([^']+)'", item.content)
                        if match:
                            node_name = match.group(1)
                    retrieved_nodes.append(node_name)
                    if isinstance(item.metadata, dict):
                        scores.append(item.metadata.get("score"))

            self.__logger.info("Computing node count and repeat count...")
            similar_trajectory_ids = self.__get_similar_trajectory_queries(
                query, user_id=user.id
            )
            query_repeat_count = len(similar_trajectory_ids)

            hint_triggered, hint_reason, hint_text = self.__generate_hint(
                query,
                query_repeat_count,
                retrieved_nodes=retrieved_nodes,
                user_id=user.id,
                message_history=message_history,
            )

            raw_answer = rag_result.answer if rag_result else ""
            rewritten_answer = self.__rewrite_response(raw_answer, query, message_history)

            new_trajectory = models.UserTrajectory(
                user_id=user.id,
                query=query,
                retrieved_nodes=retrieved_nodes,
                scores=scores,
                interaction_type=interaction_type,
                query_repeat_count=query_repeat_count + 1,
                node_entry_count=node_entry_count,
                response_time_sec=response_time_sec,
                hint_triggered=hint_triggered,
                hint_reason=hint_reason,
                hint_text=hint_text,
                raw_answer=raw_answer,
                course_id=course_id,
            )
            self.__user_service.add_trajectory_entry(user_id, new_trajectory)
            self.__logger.info(
                f"Context retrieval logged. ({interaction_type}, {node_entry_count} nodes, {response_time_sec}s)"
            )
            return SupervisorResult(
                answer=rewritten_answer,
                hint_text=hint_text,
                hint_reason=hint_reason,
            )

        except Exception as e:
            self.__logger.error(f"Error occurred while retrieving context: {e}")
            return None
