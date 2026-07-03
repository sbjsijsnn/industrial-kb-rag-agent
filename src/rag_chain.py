"""RAG 生成链: 检索 -> 拼上下文 -> LLM 生成带引用的答案。含拒答(幻觉抑制)。"""
from functools import lru_cache

from openai import OpenAI

from src import config
from src.retrieval import retrieve

SYSTEM_PROMPT = """你是工业设备运维专家助手。严格遵守以下规则:
1. 只根据【参考资料】回答, 禁止编造资料中没有的内容。
2. 每个论断后面标注引用编号, 如 [1]、[2]。
3. 如果参考资料与问题无关, 直接回答: "知识库中未找到相关资料, 建议查阅设备原厂手册或联系厂家技术支持。"
4. 回答面向一线工程师: 直接给步骤和结论, 不废话。"""

ANSWER_TEMPLATE = """【参考资料】
{context}

【问题】{question}

请基于参考资料回答, 并标注引用编号。"""


@lru_cache(maxsize=1)
def _llm() -> OpenAI:
    return OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)


def format_context(chunks: list[dict]) -> str:
    return "\n\n".join(
        f'[{i + 1}] (来源: {c["source"]} 第{c["page"]}页)\n{c["text"]}'
        for i, c in enumerate(chunks)
    )


def answer(question: str, history: list[dict] | None = None, mode: str = "rerank") -> dict:
    """主入口。返回 {answer, sources, refused}。history 为 [{role, content}] 多轮上下文。"""
    chunks = retrieve(question, mode=mode)

    # 拒答: 重排分数太低 = 知识库里没有相关内容, 强行回答必然幻觉
    best = max((c.get("rerank_score", 1.0) for c in chunks), default=0.0)
    if not chunks or best < config.RERANK_SCORE_THRESHOLD:
        return {
            "answer": "知识库中未找到相关资料, 建议查阅设备原厂手册或联系厂家技术支持。",
            "sources": [], "refused": True,
        }

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages += history[-6:]  # 只带最近 3 轮, 控制 token 成本
    messages.append({
        "role": "user",
        "content": ANSWER_TEMPLATE.format(context=format_context(chunks), question=question),
    })
    resp = _llm().chat.completions.create(
        model=config.LLM_MODEL, messages=messages, temperature=0.2,
    )
    return {
        "answer": resp.choices[0].message.content,
        "sources": [{"index": i + 1, "source": c["source"], "page": c["page"]}
                    for i, c in enumerate(chunks)],
        "refused": False,
    }


if __name__ == "__main__":
    import json
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "S7-1200 CPU 指示灯红色常亮是什么故障"
    result = answer(q)
    print(result["answer"])
    print("\n--- 引用 ---")
    print(json.dumps(result["sources"], ensure_ascii=False, indent=2))
