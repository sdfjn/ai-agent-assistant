"""
应用配置 —— 使用 pydantic-settings 管理环境变量。

所有配置项都从 .env 文件加载，提供类型校验和默认值。
"""

import os
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    """应用全局配置。

    通过 .env 文件配置，支持 Anthropic 和 OpenAI 兼容接口。
    如果同时填写了两个 API Key，优先使用 Anthropic。
    """

    # ── LLM 配置 ──
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_base_url: str = Field(default="", alias="OPENAI_BASE_URL")
    llm_model: str = Field(default="claude-haiku-4-5-20251001", alias="LLM_MODEL")

    # ── 服务配置 ──
    port: int = Field(default=8000, alias="PORT")

    # ── RAG 配置 ──
    knowledge_dir: str = Field(
        default_factory=lambda: os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)
            ))),
            "knowledge",
        ),
    )
    chroma_persist_dir: str = Field(
        default_factory=lambda: os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)
            ))),
            "chroma_db",
        ),
    )
    embedding_model: str = "shibing624/text2vec-base-chinese"
    chunk_size: int = 500
    chunk_overlap: int = 50

    @property
    def llm_provider(self) -> str:
        """推断当前使用的 LLM 提供商。"""
        if self.anthropic_api_key:
            return "anthropic"
        if self.openai_api_key:
            return "openai"
        return "none"

    @property
    def is_configured(self) -> bool:
        """检查是否至少配置了一个 API Key。"""
        return bool(self.anthropic_api_key or self.openai_api_key)


# 全局单例
settings = Settings()
