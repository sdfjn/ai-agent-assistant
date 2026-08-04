"""
FastAPI 应用入口 —— Web 服务和 API 路由。

LangChain 规范：
  应用层（FastAPI）与业务层（Agent/Tools/RAG）完全分离。
  main.py 只负责 HTTP 路由和 SSE 连接，不包含业务逻辑。
"""

import json
import os

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from .config import settings
from .schemas import ChatRequest, RAGSearchRequest, HealthResponse
from .agent import stream_agent
from .rag import rag_engine, search_knowledge

# ═══════════════════════════════════════════════════════════
# FastAPI 应用初始化
# ═══════════════════════════════════════════════════════════

app = FastAPI(
    title="Weather AI Agent",
    description="基于 LangChain + LangGraph 的天气助手 Agent",
    version="3.0.0",
)

# ─── 静态文件 ──────────────────────────────────────────

_static_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "static",
)
if os.path.exists(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")


@app.get("/")
async def index():
    """聊天界面首页。"""
    index_path = os.path.join(_static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "前端文件未找到", "docs": "/docs"}


# ═══════════════════════════════════════════════════════════
# API 路由
# ═══════════════════════════════════════════════════════════


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    对话接口 —— SSE 流式返回 Agent 的完整推理过程。

    事件类型：
      - thinking:     Agent 正在思考
      - tool_call:    Agent 调用了工具
      - tool_result:  工具返回结果
      - done:         最终回答（含 history）
      - error:        错误信息
    """

    async def event_stream():
        async for event in stream_agent(
            user_message=request.message,
            history=request.history,
            rag_tool=search_knowledge,
        ):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/rag/search")
async def rag_search(request: RAGSearchRequest):
    """
    直接搜索知识库 —— 不经过 Agent，用于调试和验证。

    帮助验证文档加载、分块和检索效果。
    """
    try:
        if rag_engine.vectorstore is None:
            rag_engine.load_documents()
        results = rag_engine.search(request.query, top_k=request.top_k)
        return {"status": "ok", "results": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/rag/stats")
async def rag_stats():
    """知识库统计信息：文档块数、Embedding 模型等。"""
    return rag_engine.get_stats()


@app.get("/api/health")
async def health() -> HealthResponse:
    """健康检查。"""
    return HealthResponse(
        status="ok",
        model=settings.llm_model,
        provider=settings.llm_provider,
        rag_enabled=rag_engine.vectorstore is not None,
    )


@app.get("/api/tools")
async def list_tools():
    """
    列出当前 Agent 可用的所有工具。

    直接从 @tool 装饰的函数中提取名称和描述，
    不需要手动维护工具列表。
    """
    from .tools import BUILTIN_TOOLS

    tools_info = [
        {"name": t.name, "description": t.description}
        for t in [*BUILTIN_TOOLS, search_knowledge]
    ]
    return tools_info


# ═══════════════════════════════════════════════════════════
# 启动入口
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("[Weather AI Agent v3.0] 启动中...")
    print(f"   模型: {settings.llm_model}")
    print(f"   提供商: {settings.llm_provider}")
    print(f"   前端: http://localhost:{settings.port}")
    print(f"   接口文档: http://localhost:{settings.port}/docs")
    print(f"   API (对话):   POST /api/chat")
    print(f"   API (RAG搜索): POST /api/rag/search")
    print(f"   API (知识库):  GET  /api/rag/stats")
    print("=" * 60)

    if not settings.is_configured:
        print("[WARN] 未检测到 API Key！")
        print("   请复制 .env.example 为 .env 并填入你的 API Key")
        print("=" * 60)

    # 启动时初始化 RAG 知识库
    print("[RAG] 正在初始化知识库...")
    try:
        count = rag_engine.load_documents()
        print(f"[RAG] 知识库就绪，共 {count} 个文档块")
    except Exception as e:
        print(f"[RAG] 初始化失败（知识库搜索功能不可用）: {e}")
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=settings.port)
