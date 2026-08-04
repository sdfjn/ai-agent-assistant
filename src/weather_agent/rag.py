"""
RAG 知识库引擎 —— 基于 ChromaDB + 中文 Embedding。

LangChain 规范：
  使用 LangChain 的 TextLoader、RecursiveCharacterTextSplitter、
  Chroma 向量存储和 OpenAI/HuggingFace Embeddings，
  将私有知识库接入 Agent。

同时导出 search_knowledge 工具供 Agent 使用。
"""

import os
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings

from .config import settings


class RAGEngine:
    """
    RAG 知识库引擎。

    负责：
      1. 加载 knowledge/ 目录下的文档
      2. 中文分块
      3. Embedding 向量化
      4. 存入 ChromaDB
      5. 提供语义检索接口
    """

    def __init__(self):
        self._embedding_model = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
        )
        self._vectorstore: Chroma | None = None

    @property
    def vectorstore(self) -> Chroma | None:
        return self._vectorstore

    def load_documents(self, force_reload: bool = False) -> int:
        """
        加载 knowledge/ 目录下所有 .txt 和 .md 文件。

        流程：加载 → 分块 → Embedding → 存入 ChromaDB

        Args:
            force_reload: 是否强制重建向量库（删除旧数据）

        Returns:
            文档块总数
        """
        persist_dir = settings.chroma_persist_dir

        # 强制重建：删除旧的向量库
        if force_reload and os.path.exists(persist_dir):
            import shutil
            shutil.rmtree(persist_dir)

        knowledge_dir = settings.knowledge_dir
        if not os.path.isdir(knowledge_dir):
            self._vectorstore = None
            return 0

        # 收集所有文档
        all_docs = []
        for file_path in Path(knowledge_dir).glob("*"):
            if file_path.suffix in (".txt", ".md"):
                try:
                    loader = TextLoader(str(file_path), encoding="utf-8")
                    all_docs.extend(loader.load())
                except Exception:
                    continue

        if not all_docs:
            self._vectorstore = None
            return 0

        # 中文友好的分块策略
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""],
        )
        chunks = text_splitter.split_documents(all_docs)

        if self._vectorstore is None:
            self._vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self._embedding_model,
                persist_directory=persist_dir,
            )
        else:
            self._vectorstore.add_documents(chunks)

        return len(chunks)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """
        语义检索。

        Args:
            query: 搜索查询
            top_k: 返回结果数量

        Returns:
            [{"content": "...", "score": 0.95}, ...]
        """
        if self._vectorstore is None:
            return []

        docs = self._vectorstore.similarity_search_with_relevance_scores(
            query, k=top_k
        )

        results = []
        for doc, score in docs:
            results.append({
                "content": doc.page_content,
                "score": round(score, 4),
            })
        return results

    def get_stats(self) -> dict:
        """获取知识库统计信息。"""
        if self._vectorstore is None:
            return {
                "status": "not_loaded",
                "collection_count": 0,
                "embedding_model": settings.embedding_model,
            }

        collection = self._vectorstore._collection
        return {
            "status": "ready",
            "collection_count": collection.count(),
            "embedding_model": settings.embedding_model,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
        }


# ─── 全局单例 ──────────────────────────────────────────

rag_engine = RAGEngine()


# ═══════════════════════════════════════════════════════════
# 工具 3：知识库搜索（作为 LangChain Tool 导出）
# ═══════════════════════════════════════════════════════════


@tool
def search_knowledge(query: str) -> str:
    """
    从私有知识库中搜索信息。
    当用户询问专业知识、概念解释、技术原理等需要查资料的问题时使用此工具。
    例如：用户问"什么是RAG？"、"Transformer怎么工作的？"、"解释一下机器学习"。

    Args:
        query: 搜索查询，用完整的问题或关键词都可以
    """
    try:
        if rag_engine.vectorstore is None:
            rag_engine.load_documents()

        results = rag_engine.search(query, top_k=3)
        if not results:
            return "知识库中没有找到相关信息。"

        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"[来源{i}，相关度{r['score']:.0%}] {r['content'][:300]}")
        return "\n\n".join(lines)
    except Exception as e:
        return f"知识库搜索失败: {str(e)}"
