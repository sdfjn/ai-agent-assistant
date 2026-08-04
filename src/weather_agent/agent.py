"""
Agent 图定义 —— 使用 LangGraph 的 create_react_agent。

LangChain 现代规范：
  使用 LangGraph 的预构建 ReAct Agent 替代旧的 AgentExecutor。
  create_react_agent 内部自动完成：
    1. 构建 StateGraph（agent 节点 + tools 节点）
    2. 条件边判断（继续调用工具 or 结束）
    3. 消息状态管理和 append 逻辑

对比旧项目中的 agent_langchain.py：
  旧：create_tool_calling_agent() + AgentExecutor + astream_events
  新：create_react_agent() + astream_events（代码更少，更可控）
"""

import json
from typing import AsyncIterator

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage

from .config import settings
from .model import create_llm
from .prompts import SYSTEM_PROMPT
from .tools import BUILTIN_TOOLS
from .rag import search_knowledge


def _assemble_tools(rag_tool=None) -> list:
    """组装工具列表：内置工具 + 可选的 RAG 工具。"""
    tools = list(BUILTIN_TOOLS)
    if rag_tool is not None:
        tools.append(rag_tool)
    return tools


def create_agent(rag_tool=None):
    """
    创建 LangGraph ReAct Agent。

    Args:
        rag_tool: 可选的 RAG 知识库搜索工具

    Returns:
        编译好的 LangGraph 图（Runnable），可直接 .ainvoke() 或 .astream_events()
    """
    llm = create_llm()
    tools = _assemble_tools(rag_tool)

    # create_react_agent 是 LangGraph 的预构建配方:
    #   - 自动创建 agent 节点（LLM + 工具绑定）
    #   - 自动创建 tools 节点（工具执行）
    #   - 自动添加条件边（tool_calls → tools，否则 → END）
    # state_modifier 会作为 system message 注入每条消息之前
    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=SYSTEM_PROMPT,
    )


def _convert_history(history: list[dict]) -> list:
    """
    将前端传来的对话历史转换为 LangChain Message 对象。

    Args:
        history: [{"role": "user/assistant", "content": "..."}, ...]

    Returns:
        [HumanMessage(...), AIMessage(...), ...]
    """
    messages = []
    for h in (history or []):
        role = h.get("role", "")
        content = h.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages


async def stream_agent(
    user_message: str,
    history: list[dict] | None = None,
    rag_tool=None,
) -> AsyncIterator[dict]:
    """
    流式运行 Agent，通过 SSE 推送每一步事件。

    使用 LangGraph 的 astream_events v2 API，
    可以捕获 LLM 推理、工具调用、工具返回等所有中间状态。

    LangGraph 的 astream_events 事件类型：
      - on_chat_model_start:   LLM 开始推理
      - on_chat_model_stream:   LLM 逐 token 输出
      - on_chat_model_end:      LLM 推理结束
      - on_tool_start:          工具开始执行
      - on_tool_end:            工具执行完毕

    Yields:
        {"type": "thinking", "content": "..."}
        {"type": "tool_call", "name": "...", "args": {...}}
        {"type": "tool_result", "name": "...", "content": "..."}
        {"type": "done", "content": "...", "history": [...]}
        {"type": "error", "content": "..."}
    """
    try:
        agent = create_agent(rag_tool=rag_tool)
    except Exception as e:
        yield {"type": "error", "content": f"Agent 初始化失败: {str(e)}"}
        return

    # 组装输入消息
    chat_history = _convert_history(history)
    messages = [*chat_history, HumanMessage(content=user_message)]

    yield {"type": "thinking", "content": "Agent 正在思考..."}

    try:
        final_output = ""
        tool_call_seen: set[str] = set()  # 避免重复推送同一个工具调用

        async for event in agent.astream_events(
            {"messages": messages},
            version="v2",
        ):
            kind = event["event"]

            # ── 工具开始执行 ──
            if kind == "on_tool_start":
                tool_name = event.get("name", "unknown")
                tool_input = event["data"].get("input", {})
                # LangGraph 可能的格式：{args: {...}} 或直接 {...}
                if isinstance(tool_input, dict) and "args" in tool_input:
                    tool_input = tool_input["args"]

                event_id = f"{tool_name}:{json.dumps(tool_input, sort_keys=True, default=str)}"
                if event_id not in tool_call_seen:
                    tool_call_seen.add(event_id)
                    yield {
                        "type": "tool_call",
                        "name": tool_name,
                        "args": tool_input,
                    }

            # ── 工具执行完毕 ──
            elif kind == "on_tool_end":
                tool_name = event.get("name", "unknown")
                output = event["data"].get("output", "")

                # LangGraph 的 tool 输出可能是 ToolMessage
                if hasattr(output, "content"):
                    output = output.content
                elif not isinstance(output, str):
                    output = str(output)

                yield {
                    "type": "tool_result",
                    "name": tool_name,
                    "content": output,
                }

            # ── LLM 流式输出 token ──
            elif kind == "on_chat_model_stream":
                chunk = event["data"].get("chunk", {})
                if hasattr(chunk, "content") and chunk.content:
                    # AIMessageChunk 的 content 可能是 str 或 list
                    content = chunk.content
                    if isinstance(content, str):
                        final_output += content

        # ── 返回最终结果 ──
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
            # 没有流式 token（如纯工具调用后无文本），用 ainvoke 兜底
            result = await agent.ainvoke({"messages": messages})
            output_messages = result.get("messages", [])
            if output_messages:
                last_msg = output_messages[-1]
                if hasattr(last_msg, "content"):
                    final_output = last_msg.content
                else:
                    final_output = str(last_msg)

            yield {
                "type": "done",
                "content": final_output or "处理完成。",
                "history": [
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": final_output or "处理完成。"},
                ],
            }

    except Exception as e:
        yield {"type": "error", "content": f"Agent 执行失败: {str(e)}"}
