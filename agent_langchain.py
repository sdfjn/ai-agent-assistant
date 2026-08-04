"""
================================================================================
LangChain Agent — 用 LangChain 框架重构的 ReAct Agent
================================================================================

为什么学这个？
  手写 ReAct（main.py）→ 理解底层原理
  LangChain Agent（本文件）→ 生产级标准写法，JD 里 80% 的岗位点名要求

LangChain Agent 的核心组件（本文件的 4 个关键概念）：

  ① Tool（工具）
     = Python 函数 + 描述（告诉 LLM 怎么用）
     LangChain 用 @tool 装饰器创建，自动生成 JSON Schema

  ② LLM（大模型）
     = ChatOpenAI / ChatAnthropic
     LangChain 统一封装，自带 .bind_tools() 绑定工具

  ③ Agent（智能体）= prompt + LLM + 输出解析器
     LangChain 用 create_tool_calling_agent() 创建

  ④ AgentExecutor（执行器）= Agent + 循环控制
     负责：执行工具 → 喂回结果 → 循环直到 LLM 给出最终答案
     这封装的就是我们手写的 ReAct for 循环

和手写版的对比：
  ┌────────────────────────┬──────────────────────────┐
  │  手写 ReAct (main.py)   │  LangChain Agent (本文件)  │
  ├────────────────────────┼──────────────────────────┤
  │  手动循环 for turn      │  AgentExecutor 自动循环    │
  │  手动拼接 tool_call msg │  框架自动处理消息格式       │
  │  手动解析 tool_use 块   │  框架自动解析              │
  │  手动管理 token 上限    │  框架内置 callback 处理     │
  │  约 100 行代码          │  约 30 行代码              │
  └────────────────────────┴──────────────────────────┘
================================================================================
"""

import json
import os
from datetime import datetime
from typing import Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")


# ============================================================================
# ① 工具定义（用 LangChain 的 @tool 装饰器）
# ============================================================================
# @tool 装饰器自动把函数名→工具名、docstring→description、参数类型→JSON Schema
# 不需要手动写 TOOLS 列表和 JSON Schema 了！

from langchain.tools import tool


@tool
def get_weather(city: str) -> str:
    """
    查询指定城市的实时天气信息，包括温度、风速、天气状况。
    当用户询问天气、气温、会不会下雨等问题时使用此工具。
    """
    # 步骤1：城市名 → 经纬度
    geo_url = "https://geocoding-api.open-meteo.com/v1/search"
    geo_resp = httpx.get(geo_url, params={"name": city, "count": 1, "language": "zh"}, timeout=10)
    geo_data = geo_resp.json()

    if not geo_data.get("results"):
        return f"找不到城市: {city}"

    r = geo_data["results"][0]

    # 步骤2：经纬度 → 天气
    weather_url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": r["latitude"], "longitude": r["longitude"],
        "current_weather": True, "timezone": "auto",
    }
    w_resp = httpx.get(weather_url, params=params, timeout=10)
    cw = w_resp.json().get("current_weather", {})

    # 步骤3：天气码转描述
    weather_codes = {
        0: "晴天", 1: "大部晴朗", 2: "多云", 3: "阴天",
        45: "有雾", 48: "雾凇",
        51: "小雨", 53: "中雨", 55: "大雨",
        61: "小阵雨", 63: "中阵雨", 65: "大阵雨",
        71: "小雪", 73: "中雪", 75: "大雪",
        80: "小阵雨", 81: "中阵雨", 82: "大阵雨",
        95: "雷暴", 96: "雷暴+冰雹", 99: "强雷暴+冰雹",
    }
    weather = weather_codes.get(cw.get("weathercode", 0), "未知")

    return json.dumps({
        "城市": f"{r.get('name', city)}, {r.get('country', '')}",
        "温度": f"{cw.get('temperature', 'N/A')}°C",
        "风速": f"{cw.get('windspeed', 'N/A')} km/h",
        "天气": weather,
        "观测时间": cw.get("time", "N/A"),
    }, ensure_ascii=False)


@tool
def get_time() -> str:
    """获取当前日期和时间。当用户询问现在几点、今天几号、星期几时使用此工具。"""
    now = datetime.now()
    weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    return f"{now.strftime('%Y-%m-%d %H:%M:%S')} {weekdays[now.weekday()]}"


# ============================================================================
# ② 创建 LLM 实例（LangChain 统一封装）
# ============================================================================

def create_llm(temperature: float = 0.0):
    """
    根据配置创建对应的 LLM 实例。

    LangChain 的 ChatOpenAI / ChatAnthropic 是对 SDK 的封装，
    好处是：它们都实现了统一的 BaseChatModel 接口，
    可以无缝切换，不需要改 Agent 代码。
    """
    if ANTHROPIC_API_KEY:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=LLM_MODEL,
            api_key=ANTHROPIC_API_KEY,
            temperature=temperature,
            max_tokens=1024,
        )
    elif OPENAI_API_KEY:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=LLM_MODEL,
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL if OPENAI_BASE_URL else None,
            temperature=temperature,
        )
    else:
        raise RuntimeError("请设置 ANTHROPIC_API_KEY 或 OPENAI_API_KEY")


# ============================================================================
# ③+④ 创建 Agent + Executor
# ============================================================================

SYSTEM_PROMPT = """你是一个实用的 AI 助手，名叫"小A"。

你有以下工具可用：
- get_weather: 查询城市天气
- get_time: 获取当前时间
- search_knowledge: 从私有知识库中搜索信息（当用户问专业知识时使用）

规则：
1. 用户问天气时，必须调用 get_weather，绝对不能编造
2. 用户问专业知识时，优先使用 search_knowledge 搜索知识库
3. 回答简洁友好，用中文回复"""


def get_agent_executor(rag_tool=None):
    """
    获取 Agent 执行器。

    这是 LangChain 的"标准配方"：
      LLM.bind_tools(tools) → create_tool_calling_agent() → AgentExecutor
    """
    from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

    # 组装工具列表
    tools = [get_weather, get_time]
    if rag_tool:
        tools.append(rag_tool)

    # 创建 LLM
    llm = create_llm()

    # ─── 关键步骤：bind_tools ──────────────────────────
    # .bind_tools() 把工具的 JSON Schema 注入到每次 API 调用中
    # 相当于我们手写时把 TOOLS 列表传给 call_llm()
    llm_with_tools = llm.bind_tools(tools)

    # ─── 创建 Prompt 模板 ──────────────────────────────
    # MessagesPlaceholder 是 LangChain 的特殊占位符：
    #   "chat_history" → 对话历史（自动管理）
    #   "agent_scratchpad" → Agent 中间思考过程（tool_call 记录自动放这里）
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # ─── 创建 Agent ────────────────────────────────────
    # create_tool_calling_agent: LangChain 内置的 tool-calling agent 配方
    # 内部自动处理：ReAct 格式的消息拼接、tool_call 解析
    agent = create_tool_calling_agent(llm_with_tools, tools, prompt)

    # ─── 创建 Executor ────────────────────────────────
    # AgentExecutor 封装了我们手写的 for turn in range(5) 循环：
    #   - max_iterations: 等同于 MAX_TURNS
    #   - verbose: 打印详细日志（调试用）
    #   - handle_parsing_errors: 解析失败时自动重试
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        max_iterations=5,
        verbose=False,
        handle_parsing_errors=True,
    )

    return executor


# ============================================================================
# 运行 Agent（流式版本）
# ============================================================================

async def run_langchain_agent(
    user_message: str,
    history: list[dict] = None,
    stream: bool = True,
    rag_tool=None,
):
    """
    运行 LangChain Agent（异步生成器，SSE 流式输出）。

    和 main.py 里的 run_agent() 接口一致，可以无缝替换。
    区别是内部用 LangChain 的 AgentExecutor 代替手写 for 循环。
    """
    from langchain_core.messages import HumanMessage, AIMessage

    if stream:
        yield {"type": "thinking", "content": "Agent (LangChain) 正在思考..."}

    # 获取 executor（传入 RAG 工具）
    executor = get_agent_executor(rag_tool=rag_tool)

    # 转换历史消息格式
    chat_history = []
    if history:
        for h in history:
            if h["role"] == "user":
                chat_history.append(HumanMessage(content=h["content"]))
            elif h["role"] == "assistant":
                chat_history.append(AIMessage(content=h["content"]))

    try:
        # ─── 核心：一步调用，LangChain 内部完成 ReAct 循环 ───
        # 如果需要看中间步骤，可以用 astream_events() 代替 ainvoke()
        result = await executor.ainvoke({
            "input": user_message,
            "chat_history": chat_history,
        })

        output = result.get("output", "抱歉，处理出错了。")

        yield {
            "type": "done",
            "content": output,
            "history": [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": output},
            ],
        }

    except Exception as e:
        yield {"type": "error", "content": f"Agent 执行失败: {str(e)}"}


async def run_langchain_agent_streaming(
    user_message: str,
    history: list[dict] = None,
    rag_tool=None,
):
    """
    流式版本：展示 LangChain Agent 的每一步中间过程。

    用 astream_events() 替代 ainvoke()，可以捕获到：
      - on_chat_model_start: LLM 开始推理
      - on_tool_start: 开始执行工具
      - on_tool_end: 工具执行完毕
      - on_chat_model_stream: LLM 逐 token 输出
    """
    from langchain_core.messages import HumanMessage, AIMessage

    yield {"type": "thinking", "content": "Agent (LangChain) 正在思考..."}

    executor = get_agent_executor(rag_tool=rag_tool)

    chat_history = []
    if history:
        for h in history:
            if h["role"] == "user":
                chat_history.append(HumanMessage(content=h["content"]))
            elif h["role"] == "assistant":
                chat_history.append(AIMessage(content=h["content"]))

    try:
        final_output = ""

        # astream_events: LangChain 的"透明模式"
        # 每个内部事件都会抛出来，前端能展示完整推理链
        async for event in executor.astream_events(
            {"input": user_message, "chat_history": chat_history},
            version="v2",
        ):
            kind = event["event"]

            if kind == "on_tool_start":
                yield {
                    "type": "tool_call",
                    "name": event["name"],
                    "args": event["data"].get("input", {}),
                }

            elif kind == "on_tool_end":
                yield {
                    "type": "tool_result",
                    "name": event["name"],
                    "content": str(event["data"].get("output", "")),
                }

            elif kind == "on_chat_model_stream":
                chunk = event["data"].get("chunk", {})
                # LangChain 的 AIMessageChunk 结构
                if hasattr(chunk, "content") and chunk.content:
                    final_output += chunk.content

        if final_output:
            yield {
                "type": "done",
                "content": final_output,
                "history": [
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": final_output},
                ],
            }
        else:
            # 如果没有流式 token，用 ainvoke 兜底
            result = await executor.ainvoke({
                "input": user_message,
                "chat_history": chat_history,
            })
            output = result.get("output", "抱歉，处理出错了。")
            yield {
                "type": "done",
                "content": output,
                "history": [
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": output},
                ],
            }

    except Exception as e:
        yield {"type": "error", "content": f"Agent 执行失败: {str(e)}"}
