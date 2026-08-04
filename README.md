# Weather AI Agent

基于 **LangChain + LangGraph** 的 AI 天气助手，展示现代 LangChain 项目标准结构与开发规范。

支持 Anthropic 及 OpenAI 兼容接口，可查询实时天气、当前时间，并结合本地 RAG 知识库回答专业问题。

## 功能

- 🌤️ **天气查询** — Open-Meteo 免费 API，全球城市实时天气
- ⏰ **时间查询** — 当前日期、时间、星期
- 📚 **RAG 知识库** — ChromaDB + 中文 Embedding，语义检索私有文档
- 💬 **SSE 流式对话** — 前端实时展示 Agent 思考过程与工具调用

## 架构

```text
用户浏览器 (static/index.html)
        │ POST /api/chat (SSE)
        ▼
FastAPI (src/weather_agent/main.py)
        │
        ▼
LangGraph Agent (src/weather_agent/agent.py)
  ├── LLM (src/weather_agent/model.py)  ← init_chat_model()
  ├── 工具 (src/weather_agent/tools.py)  ← @tool 装饰器
  └── RAG  (src/weather_agent/rag.py)    ← ChromaDB 向量检索
```

**一次典型调用**：用户提问 → LLM 判断 → 调用工具 → 结果回填 → LLM 生成回答 → SSE 推送给前端。

## 项目结构

```text
ai-agent/
├── src/weather_agent/          # 主包（LangChain 标准拆分）
│   ├── main.py                 # FastAPI 应用入口 & API 路由
│   ├── config.py               # 配置管理（pydantic-settings）
│   ├── model.py                # LLM 初始化（init_chat_model）
│   ├── prompts.py              # 系统提示词
│   ├── tools.py                # 工具定义（@tool 装饰器）
│   ├── rag.py                  # RAG 引擎 & search_knowledge 工具
│   ├── agent.py                # LangGraph create_react_agent
│   └── schemas.py              # Pydantic 请求/响应模型
├── static/
│   └── index.html              # 聊天界面
├── knowledge/                  # 私有知识库（.txt / .md）
├── requirements.txt
├── Dockerfile
├── .env.example
└── .gitignore
```

## 快速开始

Python 3.11+，PowerShell：

```powershell
cd ai-agent
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
# 编辑 .env，填写 API Key
python -m src.weather_agent.main
```

打开 http://localhost:8000。接口文档位于 http://localhost:8000/docs。

## 配置

不要同时填写两个 API Key；如果两者都存在，优先使用 Anthropic。

**Anthropic**：

```dotenv
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=
OPENAI_BASE_URL=
LLM_MODEL=claude-haiku-4-5-20251001
PORT=8000
```

**OpenAI 兼容接口**（DeepSeek / 通义千问 / 豆包等）：

```dotenv
ANTHROPIC_API_KEY=
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
PORT=8000
```

使用 OpenAI 官方接口时 `OPENAI_BASE_URL` 留空。

## API

| 方法 | 路径 | 用途 |
|---|---|---|
| `POST` | `/api/chat` | Agent 对话，SSE 流式响应 |
| `POST` | `/api/rag/search` | 测试知识库检索 |
| `GET` | `/api/rag/stats` | 知识库统计信息 |
| `GET` | `/api/tools` | 已注册工具列表 |
| `GET` | `/api/health` | 健康检查 |

聊天请求：

```json
{"message": "北京现在天气怎么样？", "history": []}
```

## 更新知识库

1. 将 UTF-8 编码的 `.txt` 或 `.md` 文件放入 `knowledge/`
2. 删除 `chroma_db/` 后重启，程序会自动重建
3. 用 `POST /api/rag/search` 验证检索效果

默认分块 500 字符、重叠 50 字符，可在 `config.py` 中调整。

## 技术栈

| 组件 | 选型 |
|---|---|
| Web 框架 | FastAPI + SSE |
| Agent 引擎 | LangGraph (`create_react_agent`) |
| LLM 统一 | `init_chat_model()` |
| 工具定义 | `@tool` 装饰器 |
| 配置管理 | pydantic-settings |
| 向量数据库 | ChromaDB |
| Embedding | `shibing624/text2vec-base-chinese` |
| 天气数据 | Open-Meteo（免费，无需 Key） |

## License

MIT
