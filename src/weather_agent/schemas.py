"""
API 数据模型 —— Pydantic 请求/响应 Schema。

LangChain 项目中，业务 Schema 与 API 逻辑分离。
"""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """聊天请求体。"""
    message: str = Field(..., description="用户消息")
    history: list[dict] = Field(default_factory=list, description="对话历史")


class RAGSearchRequest(BaseModel):
    """知识库检索请求体。"""
    query: str = Field(..., description="搜索查询")
    top_k: int = Field(default=3, ge=1, le=20, description="返回结果数")


class HealthResponse(BaseModel):
    """健康检查响应。"""
    status: str
    model: str
    provider: str
    rag_enabled: bool
