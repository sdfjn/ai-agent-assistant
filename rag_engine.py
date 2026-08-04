"""
================================================================================
RAG 引擎 — 检索增强生成
================================================================================

RAG = Retrieval-Augmented Generation（检索增强生成）

通俗理解：
  用户问一个问题 → 先去"私有知识库"里搜相关文档 → 把搜到的内容
  和问题一起发给 LLM → LLM 基于这些内容回答

为什么要 RAG？
  - LLM 的知识有截止日期（训练数据只到某个时间点）
  - LLM 不知道你公司的内部文档
  - LLM 会"幻觉"（编造不存在的事实）
  - RAG 让 LLM 基于你指定的文档回答，减少幻觉

技术流程（5步）：
  ① 文档加载   → 读取 txt/md/pdf 等文件
  ② 文本分块   → 长文档切成小块（每块约 500 字）
  ③ 向量嵌入   → 每块文字转成数学向量（Embedding）
  ④ 向量存储   → 存入 Chroma 向量数据库
  ⑤ 相似检索   → 用户提问时，找最相关的文档块

所用技术：
  - LangChain: 文档加载器、文本分割器、向量存储封装
  - ChromaDB: 轻量级向量数据库（数据存本地，无需单独服务）
  - sentence-transformers: 免费本地 Embedding 模型（all-MiniLM-L6-v2）
================================================================================
"""

import os
from pathlib import Path

# ─── LangChain 组件 ────────────────────────────────────────
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ─── 配置 ──────────────────────────────────────────────────
# 知识库文档目录
KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"

# 向量数据库持久化目录（数据存在这里，重启不丢失）
CHROMA_DB_DIR = Path(__file__).parent / "chroma_db"

# Embedding 模型
# shibing624/text2vec-base-chinese: 中文专优化模型，对中文文本检索效果更好
# all-MiniLM-L6-v2: 轻量级英文优化模型（80MB），中文也能用但稍差
EMBEDDING_MODEL = "shibing624/text2vec-base-chinese"

# 文本分块参数
CHUNK_SIZE = 500      # 每块最多 500 个字符
CHUNK_OVERLAP = 50    # 相邻块之间重叠 50 字符（防止一句话被切断）


class RAGEngine:
    """
    RAG 引擎：管理文档的加载、分块、向量化和检索。

    使用方式：
        engine = RAGEngine()
        engine.load_documents()       # 首次或文档更新后调用
        results = engine.search("什么是机器学习")  # 搜索
    """

    def __init__(self):
        # 创建 Embedding 模型（免费、本地运行）
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},  # 用 CPU（有 GPU 可改成 "cuda"）
            encode_kwargs={"normalize_embeddings": True},  # 归一化，提升检索精度
        )

        # Chroma 向量数据库实例（懒加载，调用 load_documents 后才建）
        self.vectorstore: Chroma | None = None

    # ─── 步骤1+2：加载文档 + 文本分块 ────────────────────────

    def load_documents(self, force_reload: bool = False) -> int:
        """
        加载 knowledge/ 目录下的所有文档，分块后存入向量数据库。

        参数：
            force_reload: True = 强制重新加载（文档更新后使用）
        返回：
            文档块总数
        """
        # 如果向量库已存在且不强制重载，直接复用
        if not force_reload and self._vectorstore_exists():
            self.vectorstore = Chroma(
                persist_directory=str(CHROMA_DB_DIR),
                embedding_function=self.embeddings,
            )
            return self.vectorstore._collection.count()

        # ─── 步骤1：加载文档 ─────────────────────────────
        if not KNOWLEDGE_DIR.exists():
            raise FileNotFoundError(
                f"知识库目录不存在: {KNOWLEDGE_DIR}\n"
                f"请创建该目录并放入 .txt 或 .md 文件"
            )

        documents = []
        # 加载 .txt 文件
        txt_files = list(KNOWLEDGE_DIR.glob("*.txt"))
        for f in txt_files:
            loader = TextLoader(str(f), encoding="utf-8")
            documents.extend(loader.load())

        # 加载 .md 文件
        md_files = list(KNOWLEDGE_DIR.glob("*.md"))
        for f in md_files:
            loader = TextLoader(str(f), encoding="utf-8")
            documents.extend(loader.load())

        if not documents:
            raise ValueError(
                f"知识库目录 {KNOWLEDGE_DIR} 中没有找到 .txt 或 .md 文件"
            )

        print(f"[RAG] 加载了 {len(documents)} 个文档")

        # ─── 步骤2：文本分块 ─────────────────────────────
        # RecursiveCharacterTextSplitter：按段落→句子→字符的优先级切分
        # overlap 保证关键信息不会恰好被切断在边界上
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        )
        chunks = text_splitter.split_documents(documents)
        print(f"[RAG] 切分为 {len(chunks)} 个文档块（chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}）")

        # ─── 步骤3+4：向量嵌入 + 存入 Chroma ──────────────
        # Chroma 自动完成：对每个 chunk 调 embedding 模型 → 存向量
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=str(CHROMA_DB_DIR),
        )
        print(f"[RAG] 向量化完成，已存入 {CHROMA_DB_DIR}")

        return len(chunks)

    # ─── 步骤5：相似检索 ────────────────────────────────────

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """
        根据用户问题，从知识库中检索最相关的文档块。

        参数：
            query: 用户问题
            top_k: 返回最相似的前 K 个文档块
        返回：
            [{"content": "...", "source": "ai_knowledge.txt", "score": 0.92}, ...]
        """
        if self.vectorstore is None:
            raise RuntimeError("请先调用 load_documents() 加载知识库")

        # Chroma 内部：把 query 也转成向量 → 余弦相似度计算 → 返回最相似的 K 个
        docs = self.vectorstore.similarity_search_with_score(query, k=top_k)

        results = []
        for doc, score in docs:
            # score 越小越相似（Chroma 用距离，0 = 完全匹配）
            similarity = 1.0 / (1.0 + score)  # 转换为 0~1，越大越相似
            results.append({
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "score": round(similarity, 3),
            })
        return results

    # ─── 辅助方法 ────────────────────────────────────────────

    def _vectorstore_exists(self) -> bool:
        """检查 chroma_db 目录中是否已有向量数据"""
        return CHROMA_DB_DIR.exists() and any(CHROMA_DB_DIR.iterdir())

    def get_stats(self) -> dict:
        """获取知识库统计信息"""
        if self.vectorstore is None:
            return {"status": "未加载", "chunks": 0}
        return {
            "status": "已加载",
            "chunks": self.vectorstore._collection.count(),
            "embedding_model": EMBEDDING_MODEL,
            "storage_dir": str(CHROMA_DB_DIR),
        }


# ─── 全局单例 ──────────────────────────────────────────────
# 整个应用共用一个 RAG 引擎实例（避免重复加载模型）


rag_engine = RAGEngine()