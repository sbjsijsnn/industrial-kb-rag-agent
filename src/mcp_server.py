"""MCP Server: 把知识库三工具用 MCP 协议服务化。

意义 (对比直接 import):
    - 解耦: 任何 MCP 客户端 (Claude Desktop / 其它 Agent 系统 / IDE) 都能即插即用,
      不需要 import 本项目代码 — "AI 工具的 USB-C 接口"
    - 本项目的姊妹项目 industrial-ops-agents (多智能体运维系统) 就是通过
      这个 server 消费知识库的, 见 DECISIONS.md #14

用法:
    python -m src.mcp_server          # stdio 传输 (被 MCP 客户端拉起)
"""
import sqlite3
import sys

from mcp.server.fastmcp import FastMCP

from src import config

mcp = FastMCP("industrial-kb")


@mcp.tool()
def search_manual(query: str) -> str:
    """在工业设备手册知识库中检索。输入自然语言问题(如"S7-1200 的 ERROR 灯红色常亮怎么处理"),
    返回带手册名和页码的原文片段。适用于操作步骤、参数含义、故障处理等问题。"""
    from src.retrieval import retrieve  # 延迟导入: 只在真正用到时加载重模型
    chunks = retrieve(query, mode="rerank")
    if not chunks:
        return "知识库中没有找到相关内容。"
    return "\n\n".join(
        f'[{c["source"]} 第{c["page"]}页] {c["text"]}' for c in chunks
    )


@mcp.tool()
def query_fault_code(code: str) -> str:
    """根据设备故障代码(如 E203、F0070、SRVO-023)查询故障码数据库,
    返回故障含义和处理建议。查不到时会提示改用 search_manual。"""
    conn = sqlite3.connect(config.FAULT_DB)
    row = conn.execute(
        "SELECT device, meaning, action FROM fault_codes WHERE code = ?", (code.upper(),)
    ).fetchone()
    conn.close()
    if not row:
        return f"故障码 {code} 不在数据库中, 建议用 search_manual 检索手册。"
    return f"设备: {row[0]}\n含义: {row[1]}\n处理建议: {row[2]}"


@mcp.tool()
def get_repair_history(device: str) -> str:
    """查询某设备的历史维修记录。输入设备名(如 S7-1200、G120),返回最近 5 条维修历史,
    包含日期、故障现象和处理方案。用于判断是否为复发性故障。"""
    conn = sqlite3.connect(config.FAULT_DB)
    rows = conn.execute(
        "SELECT date, fault, solution FROM repair_history "
        "WHERE device LIKE ? ORDER BY date DESC LIMIT 5", (f"%{device}%",)
    ).fetchall()
    conn.close()
    if not rows:
        return f"没有 {device} 的维修记录。"
    return "\n".join(f"{r[0]} | 故障: {r[1]} | 处理: {r[2]}" for r in rows)


def _warmup():
    """启动时预热检索链路(embedding/reranker/BM25), 否则客户端第一次调用
    search_manual 要等 20-30s 模型加载, 容易触发 MCP 客户端超时。
    (教训来自评测脚本: 冷启动会污染首次调用, 见 STUDY_GUIDE 踩坑 #4)"""
    try:
        from src.retrieval import retrieve
        retrieve("预热", mode="rerank")
        print("[mcp-server] warmup done", file=sys.stderr)
    except Exception as e:  # Milvus 未启动等情况: 服务照常起, 工具调用时再报错
        print(f"[mcp-server] warmup failed (continuing): {e}", file=sys.stderr)


if __name__ == "__main__":
    _warmup()
    mcp.run()  # 默认 stdio 传输
