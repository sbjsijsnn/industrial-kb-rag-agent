"""层5: FastAPI 后端。uvicorn src.api:app --reload 启动, /docs 看交互式文档。"""
import time

from fastapi import FastAPI
from pydantic import BaseModel

from src.agent import build_agent
from src.rag_chain import answer
from src.schemas import diagnose

app = FastAPI(title="工业设备智能运维知识库", version="0.1.0")
_agent = None  # 懒加载, 避免启动时就加载模型


class ChatRequest(BaseModel):
    question: str
    mode: str = "rerank"          # vector / hybrid / rerank
    session_id: str = "default"   # agent 模式的多轮会话 id


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    """RAG 问答(层1): 检索 + 生成 + 引用。"""
    t0 = time.time()
    result = answer(req.question, mode=req.mode)
    result["latency_ms"] = int((time.time() - t0) * 1000)
    return result


@app.post("/diagnose")
def diagnose_fault(req: ChatRequest):
    """结构化诊断(层2): 返回 Pydantic 校验过的 JSON。"""
    return diagnose(req.question).model_dump()


@app.post("/agent")
def agent_chat(req: ChatRequest):
    """Agent 问答(层3): 自主调用 手册检索/故障码/维修记录 三个工具。"""
    global _agent
    if _agent is None:
        _agent = build_agent()
    t0 = time.time()
    result = _agent.invoke(
        {"messages": [{"role": "user", "content": req.question}]},
        config={"configurable": {"thread_id": req.session_id}, "recursion_limit": 15},
    )
    return {
        "answer": result["messages"][-1].content,
        "latency_ms": int((time.time() - t0) * 1000),
    }
