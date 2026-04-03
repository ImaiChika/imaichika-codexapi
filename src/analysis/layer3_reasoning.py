from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from src.analysis.layer3_parts import (
    ReasoningContextMixin,
    ReasoningDecisionMixin,
    ReasoningReportMixin,
)
from src.config import RAG_MAX_DOC_CHARS, RAG_TOP_K, VECTOR_COLLECTION, VECTOR_DB_DIR
from src.models.llm_wrapper import QwenWrapper
from src.storage.vector_store import VectorStore


class ReasoningLayer(ReasoningContextMixin, ReasoningDecisionMixin, ReasoningReportMixin):
    """
    Layer3: hybrid retrieval + LLM reasoning.

    这里保留 Layer3 的统一对外入口，内部实现则拆到了 `layer3_parts/` 中。
    这样既不改变三层架构，也能把“上下文构建 / 推理纠偏 / 报告渲染”解耦开，
    后续阅读、答辩讲解和继续扩展都会更轻松。

    RAG retrieval strategy:
    1) semantic recall over historical embeddings
    2) recent-window recall for dialogue continuity
    3) same-user recall for persona continuity
    4) keyword-overlap recall for topic continuity
    """

    def __init__(
        self,
        max_memory_size: int = 3000,
        recent_window: int = 16,
        semantic_top_k: int = 12,
        context_top_k: int = 6,
        max_context_chars: int = 1400,
    ):
        from src.models.embedding import EmbeddingEngine

        # 基础能力组件：嵌入、LLM、向量库。
        self.embedder = EmbeddingEngine()
        self.llm = QwenWrapper()
        self.vector_store = VectorStore(VECTOR_DB_DIR, VECTOR_COLLECTION)

        # 这些参数直接决定 Layer3 的检索范围和提示词上下文长度。
        self.max_memory_size = max_memory_size
        self.recent_window = recent_window
        self.semantic_top_k = semantic_top_k
        self.context_top_k = context_top_k
        self.max_context_chars = max_context_chars
        self.rag_top_k = RAG_TOP_K
        self.rag_max_doc_chars = RAG_MAX_DOC_CHARS

        # 运行期记忆池与索引，完全沿用原实现。
        self.memory_pool: List[Dict] = []
        self.user_index: Dict[str, List[int]] = defaultdict(list)
        self.keyword_index: Dict[str, List[int]] = defaultdict(list)
