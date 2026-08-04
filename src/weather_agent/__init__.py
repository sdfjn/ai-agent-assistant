"""
Weather AI Agent — 基于 LangChain + LangGraph 的天气助手。

采用 LangChain 标准项目结构：
  config   → 配置管理
  model    → LLM 初始化
  prompts  → 提示词模板
  tools    → 工具定义（@tool）
  rag      → RAG 知识库引擎
  agent    → LangGraph Agent 图
  schemas  → API 数据模型
  main     → FastAPI 应用入口
"""
