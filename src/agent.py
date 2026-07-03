"""层3: LangGraph ReAct Agent。模型自主决定调用哪些工具、调用几次。

三个工具:
    search_manual      -> 包装 RAG 检索 (层1)
    query_fault_code   -> 查故障代码库 (SQLite, scripts/init_fault_db.py 初始化)
    get_repair_history -> 查历史维修记录 (SQLite)

vs 纯 RAG 的区别: 问"E203 是什么故障, 之前修过吗" 这种复合问题,
纯 RAG 只能检索一次; Agent 会先查故障码表、再查维修记录、综合后回答。
"""
import sqlite3

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from src import config
from src.retrieval import retrieve


@tool
def search_manual(query: str) -> str:
    """在设备手册知识库中检索相关内容。输入自然语言问题, 返回带来源的手册片段。"""
    chunks = retrieve(query, mode="rerank")
    if not chunks:
        return "知识库中没有找到相关内容。"
    return "\n\n".join(
        f'[{c["source"]} 第{c["page"]}页] {c["text"]}' for c in chunks
    )


@tool
def query_fault_code(code: str) -> str:
    """根据故障代码(如 E203、F0001)查询故障码数据库, 返回故障含义和处理建议。"""
    conn = sqlite3.connect(config.FAULT_DB)
    row = conn.execute(
        "SELECT device, meaning, action FROM fault_codes WHERE code = ?", (code.upper(),)
    ).fetchone()
    conn.close()
    if not row:
        return f"故障码 {code} 不在数据库中, 建议用 search_manual 检索手册。"
    return f"设备: {row[0]}\n含义: {row[1]}\n处理建议: {row[2]}"


@tool
def get_repair_history(device: str) -> str:
    """查询某设备的历史维修记录。输入设备名(如 S7-1200), 返回最近的维修历史。"""
    conn = sqlite3.connect(config.FAULT_DB)
    rows = conn.execute(
        "SELECT date, fault, solution FROM repair_history "
        "WHERE device LIKE ? ORDER BY date DESC LIMIT 5", (f"%{device}%",)
    ).fetchall()
    conn.close()
    if not rows:
        return f"没有 {device} 的维修记录。"
    return "\n".join(f"{r[0]} | 故障: {r[1]} | 处理: {r[2]}" for r in rows)


SYSTEM_PROMPT = """你是工厂设备运维 Agent。收到问题后自主规划:
- 故障代码类问题 -> 先 query_fault_code, 查不到再 search_manual
- 操作/参数类问题 -> search_manual
- 涉及"以前/历史/上次" -> get_repair_history
- 复合问题分步调用多个工具, 综合结果后回答
回答必须注明信息来源(工具名或手册页码)。资料不足就说不知道, 不许编造。"""


def build_agent():
    llm = ChatOpenAI(
        model=config.LLM_MODEL, api_key=config.LLM_API_KEY,
        base_url=config.LLM_BASE_URL, temperature=0,
    )
    return create_react_agent(
        llm,
        tools=[search_manual, query_fault_code, get_repair_history],
        prompt=SYSTEM_PROMPT,
        checkpointer=MemorySaver(),  # 多轮记忆: 同 thread_id 自动带上下文
    )


def run(question: str, thread_id: str = "cli") -> str:
    agent = build_agent()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"configurable": {"thread_id": thread_id}, "recursion_limit": 15},
    )
    return result["messages"][-1].content


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "E203 是什么故障? 这台设备之前出过类似问题吗?"
    print(run(q))
