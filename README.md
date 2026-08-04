# AI Agent Assistant

一个用于学习 AI Agent 工作原理的中文项目。网页端可以与大模型对话，模型还能按需调用天气、时间和本地知识库工具。

项目保留两套 Agent：

- **Native**：手写工具调用循环，用来理解底层原理。
- **LangChain**：使用 `AgentExecutor` 完成同样的工作，用来学习框架封装。

> 这是教学项目，未经认证、限流、审计和错误恢复改造，请勿直接部署到公网。

## 功能与架构

- 支持 Anthropic 及 OpenAI 兼容接口
- 使用 Open-Meteo 查询天气
- 使用本地 Embedding 和 ChromaDB 检索私有知识库
- 通过 SSE 展示思考状态、工具调用、结果和最终回答

```text
浏览器 static/index.html
        │ POST + SSE
        ▼
FastAPI main.py
        ├── Native Agent：main.py/run_agent
        └── LangChain Agent：agent_langchain.py
                    │
          天气 / 时间 / 知识库工具
                              │
                              ▼
                  rag_engine.py → ChromaDB
                              ▲
                       knowledge/*.{txt,md}
```

一次工具调用会经历：用户提问 → LLM 选择工具 → Python 执行工具 → 结果加入消息列表 → LLM 生成最终回答。

## 项目结构

```text
ai-agent/
├── main.py                 # FastAPI、Native Agent、路由和工具注册
├── agent_langchain.py      # LangChain Agent
├── rag_engine.py           # 分块、Embedding 和向量检索
├── static/index.html       # 聊天界面
├── knowledge/              # 私有知识库（.txt/.md）
├── chroma_db/              # 可重新生成的本地向量库
├── requirements.txt
├── .env.example
└── .gitignore
```

## 快速开始

推荐 Python 3.11。PowerShell 中执行：

```powershell
cd F:\modelbushu\ai-agent
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
# 编辑 .env，只配置一个服务商
python main.py
```

打开 <http://localhost:8000>。接口文档位于 <http://localhost:8000/docs>。

## 配置

不要同时填写两个 API Key；如果两者都存在，程序会优先使用 Anthropic。

Anthropic：

```dotenv
ANTHROPIC_API_KEY=你的密钥
OPENAI_API_KEY=
OPENAI_BASE_URL=
LLM_MODEL=你的模型ID
PORT=8000
```

OpenAI 兼容接口：

```dotenv
ANTHROPIC_API_KEY=
OPENAI_API_KEY=你的密钥
OPENAI_BASE_URL=https://你的服务商/v1
LLM_MODEL=你的模型ID
PORT=8000
```

使用 OpenAI 官方接口时 `OPENAI_BASE_URL` 留空。模型 ID 和 Base URL 以服务商文档为准。RAG 当前固定使用本地模型 `shibing624/text2vec-base-chinese` 和 CPU，可在 `rag_engine.py` 中修改。

## API

| 方法 | 路径 | 用途 |
|---|---|---|
| `POST` | `/api/chat` | Native Agent，SSE 响应 |
| `POST` | `/api/chat/langchain` | LangChain Agent，SSE 响应 |
| `POST` | `/api/rag/search` | 测试知识库检索 |
| `GET` | `/api/rag/stats` | 知识库状态 |
| `GET` | `/api/tools` | 已注册工具 |
| `GET` | `/api/health` | 健康检查 |

聊天请求：

```json
{"message":"北京现在天气怎么样？","history":[]}
```

RAG 检索请求：

```json
{"query":"什么是 RAG？","top_k":3}
```

## 更新知识库

1. 将 UTF-8 编码的 `.txt` 或 `.md` 放入 `knowledge/`。
2. 删除 `chroma_db/` 后重启，或调用 `load_documents(force_reload=True)`。
3. 使用 `/api/rag/search` 检查结果。

默认分块大小为 500 字符、重叠 50 字符。修改分块参数或 Embedding 模型后必须重建向量库。

## 学习顺序

1. 阅读 `main.py` 的 `TOOLS` 和 `TOOL_EXECUTORS`，理解函数与 JSON Schema。
2. 阅读 `run_agent()`，理解“模型决策 → 工具执行 → 结果回填”的循环。
3. 对照 `agent_langchain.py`，理解 `@tool`、Prompt 和 `AgentExecutor`。
4. 阅读 `rag_engine.py`，理解分块、Embedding、向量存储和 Top-K 检索。
5. 阅读前端 `sendMessage()`，理解浏览器如何消费 SSE。

推荐练习：新增安全的计算器工具；加入自己的 Markdown 文档；比较两种 Agent 模式。

## 生产化前需要补充

- 身份认证、权限和限流
- 工具参数校验、超时、重试与熔断
- 对话持久化、上下文裁剪和费用统计
- Prompt 注入防护和工具隔离
- 日志、监控和自动化测试
- 异步 HTTP 客户端及并发控制
