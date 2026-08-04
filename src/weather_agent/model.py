"""
LLM 模型初始化 —— 使用 LangChain 的 init_chat_model。

LangChain 规范：模型初始化集中管理，
通过统一的工厂函数创建，支持自动推断 provider。
"""

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from .config import settings


def create_llm(temperature: float = 0.0) -> BaseChatModel:
    """
    创建 LLM 实例。

    根据环境变量自动选择 Anthropic 或 OpenAI 兼容接口。
    使用 init_chat_model() 自动推断 provider，
    无需手动判断 if/elif。

    Returns:
        BaseChatModel 实例（ChatAnthropic 或 ChatOpenAI）。
    """
    if not settings.is_configured:
        raise RuntimeError(
            "未配置 API Key！请在 .env 中设置 ANTHROPIC_API_KEY 或 OPENAI_API_KEY"
        )

    # init_chat_model 根据 api_key 参数自动推断使用哪个 provider
    # 支持 Anthropic 原生接口和 OpenAI 兼容接口（含国产模型）
    if settings.anthropic_api_key:
        return init_chat_model(
            settings.llm_model,
            model_provider="anthropic",
            api_key=settings.anthropic_api_key,
            temperature=temperature,
            max_tokens=1024,
        )
    else:
        return init_chat_model(
            settings.llm_model,
            model_provider="openai",
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url or None,
            temperature=temperature,
        )


def bind_tools(llm: BaseChatModel, tools: list) -> BaseChatModel:
    """
    将工具绑定到 LLM。

    .bind_tools() 是 LangChain 的标准做法：
    把每个工具（@tool 装饰的函数）的 JSON Schema
    注入到每次 API 调用中，让模型能感知可用工具。

    Args:
        llm: 通过 create_llm() 创建的模型实例
        tools: 由 @tool 装饰器创建的工具列表

    Returns:
        绑定了工具的 LLM 实例
    """
    return llm.bind_tools(tools)
