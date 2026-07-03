"""Gradio 前端: python -m src.ui 启动, 浏览器访问 http://localhost:7860"""
import gradio as gr

from src.agent import build_agent
from src.rag_chain import answer

_agent = None


def rag_respond(message, history):
    hist = []
    for user, bot in history:
        hist += [{"role": "user", "content": user},
                 {"role": "assistant", "content": bot}]
    result = answer(message, history=hist)
    text = result["answer"]
    if result["sources"]:
        text += "\n\n📚 引用来源:\n" + "\n".join(
            f'[{s["index"]}] {s["source"]} 第{s["page"]}页' for s in result["sources"]
        )
    return text


def agent_respond(message, history):
    global _agent
    if _agent is None:
        _agent = build_agent()
    result = _agent.invoke(
        {"messages": [{"role": "user", "content": message}]},
        config={"configurable": {"thread_id": "gradio"}, "recursion_limit": 15},
    )
    return result["messages"][-1].content


with gr.Blocks(title="工业设备智能运维助手") as demo:
    gr.Markdown("# 🏭 工业设备智能运维知识库\n基于 RAG + Agent 的设备手册问答系统")
    with gr.Tab("📖 知识库问答 (RAG)"):
        gr.ChatInterface(rag_respond, examples=[
            "S7-1200 CPU 的 ERROR 灯红色常亮怎么处理?",
            "PLC 程序下载失败有哪些可能原因?",
        ])
    with gr.Tab("🤖 运维 Agent (工具调用)"):
        gr.ChatInterface(agent_respond, examples=[
            "E203 是什么故障? 这台设备之前出过类似问题吗?",
            "查一下 S7-1200 的维修历史",
        ])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
