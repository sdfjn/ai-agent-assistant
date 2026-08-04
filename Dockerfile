# ============================================
# AI Agent Assistant Docker 镜像
# ============================================
# 构建：docker build -t ai-agent .
# 运行：docker run -p 8000:8000 --env-file .env ai-agent

FROM python:3.11-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY main.py agent_langchain.py rag_engine.py ./
COPY static/ static/
COPY knowledge/ knowledge/

# 暴露端口（与 .env 中 PORT 一致）
EXPOSE 8000

# 启动
CMD ["python", "main.py"]
