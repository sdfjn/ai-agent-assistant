"""
================================================================================
AI Agent Assistant — 带工具调用的对话式 Agent
================================================================================

核心架构（从外到里）：
  用户 → [FastAPI] → [Agent 引擎] → [LLM API]
                          ↓
                     [工具执行器]
                          ↓
                   [天气API / 其他工具]

Agent 的工作循环（ReAct 模式）：
  1. 用户发消息
  2. Agent 把「消息 + 可用工具列表」发给 LLM
  3. LLM 决定：直接回复文本？还是调用工具？
  4. 如果要调工具 → Agent 执行工具 → 把结果喂回 LLM → 回到步骤3
  5. 如果是文本 → 返回给用户

运行方式：
  pip install -r requirements.txt
  cp .env.example .env   # 填入你的API Key
  python main.py
  然后打开 http://localhost:8000
================================================================================
"""

import json
import os
from datetime import datetime
from typing import Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ─── 加载环境变量 ─────────────────────────────────────────────
load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")  # 国产模型可改 base_url
LLM_MODEL = os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001")


# ============================================================================
# 第一部分：工具定义
# ============================================================================
# 每个工具就是一个 Python 函数 + 一个 JSON Schema 描述。
# JSON Schema 告诉 LLM："我有什么工具、参数是什么、怎么用它"。
# 这就是 Agent 的"手"——没有工具，Agent 只会聊天；有了工具，Agent 能干活。


def geocode_city(city_name: str) -> dict:
    """
    城市名 → 经纬度（调用 Open-Meteo 免费地理编码API）
    """
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": city_name, "count": 1, "language": "zh"}
    resp = httpx.get(url, params=params, timeout=10)
    data = resp.json()

    if data.get("results"):
        r = data["results"][0]
        return {
            "city": r.get("name", city_name),
            "country": r.get("country", ""),
            "latitude": r["latitude"],
            "longitude": r["longitude"],
        }
    return {"error": f"找不到城市: {city_name}"}


def get_weather(city: str) -> dict:
    """
    查询指定城市的实时天气（调用 Open-Meteo 免费天气API，无需Key）
    返回：温度、风速、天气状况等
    """
    # 第1步：地名转经纬度
    geo = geocode_city(city)
    if "error" in geo:
        return geo

    # 第2步：用经纬度查询天气
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": geo["latitude"],
        "longitude": geo["longitude"],
        "current_weather": True,
        "timezone": "auto",
    }
    resp = httpx.get(url, params=params, timeout=10)
    data = resp.json()
    cw = data.get("current_weather", {})

    # 第3步：整理成易读的格式
    weather_code_map = {
        0: "☀️ 晴天", 1: "🌤️ 大部晴朗", 2: "⛅ 多云",
        3: "☁️ 阴天", 45: "🌫️ 有雾", 48: "🌫️ 雾凇",
        51: "🌧️ 小雨", 53: "🌧️ 中雨", 55: "🌧️ 大雨",
        61: "🌧️ 小阵雨", 63: "🌧️ 中阵雨", 65: "🌧️ 大阵雨",
        71: "🌨️ 小雪", 73: "🌨️ 中雪", 75: "🌨️ 大雪",
        80: "🌧️ 小阵雨", 81: "🌧️ 中阵雨", 82: "🌧️ 大阵雨",
        95: "⛈️ 雷暴", 96: "⛈️ 雷暴+冰雹", 99: "⛈️ 强雷暴+冰雹",
    }
    weather_desc = weather_code_map.get(cw.get("weathercode", 0), "未知")

    return {
        "city": geo["city"],
        "country": geo["country"],
        "temperature": f"{cw.get('temperature', 'N/A')}°C",
        "wind_speed": f"{cw.get('windspeed', 'N/A')} km/h",
        "weather": weather_desc,
        "time": cw.get("time", "N/A"),
    }


# ─── 工具的 JSON Schema 描述 ──────────────────────────────────
# 这是给 LLM 看的"说明书"，告诉它：
#   1. 工具叫什么名字
#   2. 有什么作用
#   3. 需要什么参数
# LLM 看完后就能自主决定：这个工具能不能帮用户解决问题

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的实时天气信息，包括温度、风速、天气状况。当用户询问天气、气温、会不会下雨等问题时使用此工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，例如 '北京'、'上海'、'Tokyo'、'London'",
                    }
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "获取当前日期和时间。当用户询问现在几点、今天几号、星期几时使用此工具。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]

# 工具名称 → 实际执行函数的映射表
# Agent 收到 LLM 的工具调用请求后，在这里找到对应的函数并执行
TOOL_EXECUTORS = {
    "get_weather": lambda args: get_weather(args["city"]),
    "get_time": lambda args: {
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "weekday": ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"][datetime.now().weekday()],
    },
}


# ============================================================================
# 第二部分：LLM 适配层
# ============================================================================
# 支持两种 LLM 调用方式：
#   A) Anthropic Claude（原生工具调用支持最好）
#   B) OpenAI 兼容接口（OpenAI / DeepSeek / 通义千问 / 豆包 等）
#
# 这里展示了两种接入方式的差异：
#   - Anthropic: 用 Messages API，工具定义在 tool_config 参数里
#   - OpenAI: 用 Chat Completions API，工具定义在 tools 参数里
#   两种返回格式不同，需要分别解析

SYSTEM_PROMPT = """你是一个实用的 AI 助手，名叫"小A"。

你有以下能力：
- 💬 日常对话：回答各种问题
- 🌤️ 查天气：使用 get_weather 工具查询城市天气
- ⏰ 查时间：使用 get_time 工具获取当前时间
- 📚 搜知识库：使用 search_knowledge 工具从私有知识库中搜索专业知识

规则：
1. 用户问天气时，必须调用 get_weather 工具获取真实数据，**绝对不要编造天气信息**
2. 用户问时间时，调用 get_time 工具获取真实时间
3. 用户问专业知识（如机器学习、Transformer、RAG 等技术概念）时，使用 search_knowledge 搜索知识库
4. 回答简洁友好，用中文回复
5. 天气查询前先告诉用户"正在查询XX的天气..."
6. 拿到天气数据后，用口语化的方式告诉用户"""


def call_llm_anthropic(messages: list, tools: list) -> dict:
    """调用 Anthropic Claude API（原生工具调用）

    Anthropic 和 OpenAI 的消息格式差异很大，这里做完整转换：

    OpenAI 格式 (内部存储) → Anthropic 格式 (API调用)

    主要差异：
    1. Anthropic 用 system 参数，不用 system role
    2. Anthropic 的 assistant tool_use 是 content block，不是独立的 tool_calls 字段
    3. Anthropic 的 tool result 是 user message 里的 tool_result content block
    """
    import anthropic

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # 步骤1：转换工具定义
    anthropic_tools = []
    for t in tools:
        func = t["function"]
        anthropic_tools.append({
            "name": func["name"],
            "description": func["description"],
            "input_schema": {
                "type": "object",
                "properties": func["parameters"]["properties"],
                "required": func["parameters"].get("required", []),
            },
        })

    # 步骤2：转换消息列表
    system_msg = ""
    anthropic_messages = []

    for m in messages:
        role = m["role"]

        if role == "system":
            # system prompt 合并到一个字符串（Anthropic 的 system 是独立参数）
            system_msg += m.get("content", "")
            continue

        if role == "user":
            anthropic_messages.append({
                "role": "user",
                "content": m.get("content", ""),
            })

        elif role == "assistant" and m.get("tool_calls"):
            # OpenAI: {"role":"assistant","tool_calls":[...]}
            # → Anthropic: {"role":"assistant","content":[{"type":"tool_use",...}]}
            content_blocks = []
            # 如果有文本内容，先加 text block
            if m.get("content"):
                content_blocks.append({"type": "text", "text": m["content"]})
            # 再加 tool_use blocks
            for tc in m["tool_calls"]:
                args = tc["function"]["arguments"]
                if isinstance(args, str):
                    args = json.loads(args)
                content_blocks.append({
                    "type": "tool_use",
                    "id": tc.get("id", "call_1"),
                    "name": tc["function"]["name"],
                    "input": args,
                })
            anthropic_messages.append({
                "role": "assistant",
                "content": content_blocks,
            })

        elif role == "tool":
            # OpenAI: {"role":"tool","tool_call_id":"xxx","content":"..."}
            # → Anthropic: {"role":"user","content":[{"type":"tool_result","tool_use_id":"xxx","content":"..."}]}
            anthropic_messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": m.get("tool_call_id", "call_1"),
                    "content": m.get("content", ""),
                }],
            })

        elif role == "assistant":
            # 普通文本回复
            anthropic_messages.append({
                "role": "assistant",
                "content": m.get("content", ""),
            })

    # 步骤3：调用 Anthropic API
    response = client.messages.create(
        model=LLM_MODEL,
        max_tokens=1024,
        system=system_msg if system_msg else None,
        messages=anthropic_messages,
        tools=anthropic_tools,
    )

    return _parse_anthropic_response(response)


def _parse_anthropic_response(response) -> dict:
    """解析 Anthropic 返回：判断是文本回复还是工具调用"""
    for block in response.content:
        if block.type == "tool_use":
            return {
                "type": "tool_call",
                "tool_name": block.name,
                "arguments": block.input,
                "tool_id": block.id,
            }

    # 没有 tool_use，合并所有文本
    text = "".join(
        block.text for block in response.content if block.type == "text"
    )
    return {"type": "text", "content": text}


def call_llm_openai(messages: list, tools: list) -> dict:
    """调用 OpenAI 兼容 API"""
    from openai import OpenAI

    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL if OPENAI_BASE_URL else None,
    )

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto",  # 让模型自己决定是否调用工具
    )

    return _parse_openai_response(response)


def _parse_openai_response(response) -> dict:
    """解析 OpenAI 返回"""
    choice = response.choices[0]
    msg = choice.message

    if msg.tool_calls:
        tc = msg.tool_calls[0]
        return {
            "type": "tool_call",
            "tool_name": tc.function.name,
            "arguments": json.loads(tc.function.arguments),
            "tool_id": tc.id,
        }
    else:
        return {"type": "text", "content": msg.content or ""}


def call_llm(messages: list, tools: list) -> dict:
    """
    统一的 LLM 调用入口
    自动根据配置选择 Anthropic 或 OpenAI 接口
    """
    if ANTHROPIC_API_KEY:
        return call_llm_anthropic(messages, tools)
    elif OPENAI_API_KEY:
        return call_llm_openai(messages, tools)
    else:
        raise RuntimeError("请设置 ANTHROPIC_API_KEY 或 OPENAI_API_KEY 环境变量")


# ============================================================================
# 第三部分：Agent 引擎（ReAct 循环）
# ============================================================================
# 这是整个项目最核心的部分 —— Agent 的"大脑"。
#
# ReAct = Reasoning（推理） + Acting（行动）
# 流程：
#   ┌──────────────────────────────────────────┐
#   │  1. 用户发消息                             │
#   │  2. 组装消息（系统提示 + 历史 + 新消息）      │
#   │  3. 调用 LLM，让它决定下一步                  │
#   │  4. LLM 返回 tool_call? ───→ 执行工具 ───→ 跳回步骤3  │
#   │  5. LLM 返回 text? ───→ 结束，返回给用户       │
#   └──────────────────────────────────────────┘
#
# 关键设计：
#   - max_turns：限制最多循环 5 轮，防止死循环
#   - 每步通过 yield 返回，实现流式展示思考过程


async def run_agent(
    user_message: str,
    history: list[dict],
    stream: bool = True,
):
    """
    运行 Agent 主循环（异步生成器，支持流式输出）

    参数：
        user_message: 用户输入
        history: 之前的对话历史
        stream: 是否流式输出

    yield:
        {"type": "thinking", "content": "..."}   # Agent 正在思考
        {"type": "tool_call", "name": "...", "args": {...}}  # 正在调用工具
        {"type": "tool_result", "content": "..."}  # 工具执行结果
        {"type": "done", "content": "..."}  # 最终回复
        {"type": "error", "content": "..."}  # 出错了
    """
    # 组装初始消息列表
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": user_message},
    ]

    MAX_TURNS = 5  # 最多循环5轮，防止无限循环

    for turn in range(MAX_TURNS):
        # ─── 步骤1: 通知前端"正在思考" ───
        if stream:
            yield {
                "type": "thinking",
                "content": f"Agent 正在思考..." if turn == 0 else f"Agent 正在分析工具结果...",
            }

        # ─── 步骤2: 调用 LLM ───
        try:
            result = call_llm(messages, TOOLS)
        except Exception as e:
            yield {"type": "error", "content": f"LLM 调用失败: {str(e)}"}
            return

        # ─── 步骤3: 判断 LLM 的决策 ───
        if result["type"] == "text":
            # LLM 决定直接回复文本 → 对话结束
            assistant_msg = result["content"]
            yield {
                "type": "done",
                "content": assistant_msg,
                "history": [
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": assistant_msg},
                ],
            }
            return

        elif result["type"] == "tool_call":
            tool_name = result["tool_name"]
            tool_args = result["arguments"]

            # 通知前端：正在调用工具
            if stream:
                yield {
                    "type": "tool_call",
                    "name": tool_name,
                    "args": tool_args,
                }

            # 执行工具
            executor = TOOL_EXECUTORS.get(tool_name)
            if executor:
                try:
                    tool_result = executor(tool_args)
                except Exception as e:
                    tool_result = {"error": f"工具执行失败: {str(e)}"}
            else:
                tool_result = {"error": f"未知工具: {tool_name}"}

            # 通知前端：工具执行完毕
            if stream:
                yield {
                    "type": "tool_result",
                    "name": tool_name,
                    "content": json.dumps(tool_result, ensure_ascii=False),
                }

            # 把 LLM 的工具调用请求 和 工具执行结果 都加入对话
            # 这样 LLM 在下一轮就能看到"自己调了什么工具、工具返回了什么"
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": result.get("tool_id", "call_1"),
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(tool_args, ensure_ascii=False),
                    },
                }],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": result.get("tool_id", "call_1"),
                "content": json.dumps(tool_result, ensure_ascii=False),
            })

            # 循环回到步骤1，让 LLM 基于工具结果继续思考
            continue

    # 超过最大轮数还没结束
    yield {"type": "error", "content": "Agent 思考轮数超过上限，请简化您的问题"}


# ============================================================================
# 第四部分：FastAPI 应用
# ============================================================================

app = FastAPI(
    title="AI Agent Assistant",
    description="带工具调用 + LangChain + RAG 的对话式 AI Agent",
    version="2.0.0",
)


# ─── 请求/响应模型 ─────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []  # [{"role": "user/assistant", "content": "..."}]
    mode: str = "native"      # 🆕 "native" = 手写ReAct | "langchain" = LangChain Agent


# ========================================================================
# 🆕 RAG 知识库搜索工具（供 Agent 调用）
# ========================================================================
# 这是一个"桥接"工具：用 LangChain 的 @tool 定义，但和手写 Agent 也兼容。
# Agent 调 search_knowledge("什么是机器学习") → RAG 引擎检索 → 返回文档片段

from langchain.tools import tool as lc_tool

@lc_tool
def search_knowledge(query: str) -> str:
    """
    从私有知识库中搜索信息。当用户询问专业知识、概念解释、技术原理等需要查资料的问题时使用此工具。
    例如：用户问"什么是RAG？"、"Transformer怎么工作的？"、"解释一下机器学习"。
    """
    from rag_engine import rag_engine as re

    try:
        if re.vectorstore is None:
            re.load_documents()
        results = re.search(query, top_k=3)
        if not results:
            return "知识库中没有找到相关信息。"

        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"[来源{i}，相关度{r['score']:.0%}] {r['content'][:300]}")
        return "\n\n".join(lines)
    except Exception as e:
        return f"知识库搜索失败: {str(e)}"


def _get_rag_tool_for_native():
    """
    为手写 Agent 生成 search_knowledge 工具的 JSON Schema 定义。
    手写 Agent 用 OpenAI 格式的 TOOLS 列表，需要手动写 Schema。
    """
    return {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "从私有知识库中搜索信息。当用户询问专业知识、概念解释、技术原理时使用。例如：什么是RAG、Transformer原理、机器学习分类等。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索查询，用完整的问题或关键词都可以",
                    }
                },
                "required": ["query"],
            },
        },
    }


# 注册 RAG 工具到执行器
_rag_tool_schema = _get_rag_tool_for_native()
TOOLS.append(_rag_tool_schema)
TOOL_EXECUTORS["search_knowledge"] = lambda args: search_knowledge.invoke(args)


# ─── 对话接口（手写 ReAct，保留原版） ─────────────────────────

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    手写 ReAct Agent（教学版）—— SSE 流式返回每一步思考过程。
    保留原实现，方便对比学习 LangChain 版。
    """
    async def generate():
        async for event in run_agent(
            user_message=request.message,
            history=request.history,
            stream=True,
        ):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ─── 🆕 LangChain Agent 对话接口 ──────────────────────────────

@app.post("/api/chat/langchain")
async def chat_langchain(request: ChatRequest):
    """
    LangChain 版 Agent —— 使用 AgentExecutor 代替手写 ReAct 循环。
    内部用 astream_events 捕获中间步骤，实现与手写版一致的流式效果。
    """
    from agent_langchain import run_langchain_agent_streaming

    async def generate():
        async for event in run_langchain_agent_streaming(
            user_message=request.message,
            history=request.history,
            rag_tool=search_knowledge,
        ):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ─── 🆕 RAG 知识库接口 ────────────────────────────────────────

class RAGSearchRequest(BaseModel):
    query: str
    top_k: int = 3


@app.post("/api/rag/search")
async def rag_search(request: RAGSearchRequest):
    """
    直接搜索知识库（不经过 Agent，直接调 RAG 引擎）。
    用于测试知识库内容是否正确加载、检索是否准确。
    """
    from rag_engine import rag_engine as re

    try:
        if re.vectorstore is None:
            re.load_documents()
        results = re.search(request.query, top_k=request.top_k)
        return {"status": "ok", "results": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/rag/stats")
async def rag_stats():
    """获取知识库统计信息"""
    from rag_engine import rag_engine as re
    return re.get_stats()


# ─── 健康检查 ──────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "model": LLM_MODEL,
        "rag_enabled": True,
        "langchain_enabled": True,
    }


# ─── 可用工具列表 ──────────────────────────────────────────────

@app.get("/api/tools")
async def list_tools():
    """返回当前 Agent 可用的所有工具（含 RAG 搜索）"""
    tools_info = []
    for t in TOOLS:
        tools_info.append({
            "name": t["function"]["name"],
            "description": t["function"]["description"],
        })
    return tools_info


# ─── 前端页面 ──────────────────────────────────────────────────

import os as _os
_static_dir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "static")
if _os.path.exists(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")


@app.get("/")
async def index():
    """聊天界面首页"""
    return FileResponse(_os.path.join(_static_dir, "index.html"))


# ============================================================================
# 第五部分：启动入口
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))

    print("=" * 60)
    print("[AI Agent Assistant v2.0] 启动中...")
    print(f"   模型: {LLM_MODEL}")
    print(f"   前端: http://localhost:{port}")
    print(f"   API (手写Agent):  POST /api/chat")
    print(f"   API (LangChain):  POST /api/chat/langchain")
    print(f"   API (RAG搜索):    POST /api/rag/search")
    print(f"   API (知识库状态): GET  /api/rag/stats")
    print("=" * 60)

    if not ANTHROPIC_API_KEY and not OPENAI_API_KEY:
        print("[WARN] 未检测到 API Key！")
        print("   请复制 .env.example 为 .env 并填入你的 API Key")
        print("=" * 60)

    # ─── 🆕 启动时初始化 RAG 知识库 ──────────────────────
    print("[RAG] 正在初始化知识库...")
    try:
        from rag_engine import rag_engine as re
        count = re.load_documents()
        print(f"[RAG] 知识库就绪，共 {count} 个文档块")
    except Exception as e:
        print(f"[RAG] 初始化失败（知识库搜索功能不可用）: {e}")
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=port)
