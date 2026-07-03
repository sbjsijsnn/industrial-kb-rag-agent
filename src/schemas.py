"""层2: Pydantic 结构化输出。LLM 返回验证过的对象而非自由文本。

为什么: 下游系统(工单/报表)要消费诊断结果, 解析自由文本用正则太脆;
structured output 让模型直接吐 JSON 且强制 schema 校验, 不合法会自动重试。
"""
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src import config
from src.retrieval import retrieve


class FaultDiagnosis(BaseModel):
    """一次设备故障诊断的结构化结果。"""
    fault_summary: str = Field(description="故障现象一句话总结")
    possible_causes: list[str] = Field(description="可能原因, 按概率从高到低")
    severity: str = Field(description="严重程度: 低 / 中 / 高")
    repair_steps: list[str] = Field(description="建议的排查/维修步骤, 按执行顺序")
    need_manufacturer: bool = Field(description="是否需要联系原厂支持")
    citations: list[str] = Field(description="引用来源, 格式: 文件名 第N页")


def diagnose(fault_description: str) -> FaultDiagnosis:
    """输入故障描述, 返回结构化诊断。"""
    chunks = retrieve(fault_description, mode="rerank")
    context = "\n\n".join(
        f'(来源: {c["source"]} 第{c["page"]}页) {c["text"]}' for c in chunks
    )
    llm = ChatOpenAI(
        model=config.LLM_MODEL, api_key=config.LLM_API_KEY,
        base_url=config.LLM_BASE_URL, temperature=0,
    ).with_structured_output(FaultDiagnosis)
    return llm.invoke(
        f"根据以下参考资料, 对故障进行结构化诊断。资料没提到的不要编。\n\n"
        f"【参考资料】\n{context}\n\n【故障描述】{fault_description}"
    )


if __name__ == "__main__":
    import sys
    desc = sys.argv[1] if len(sys.argv) > 1 else "S7-1200 的 ERROR 灯闪烁, 程序无法下载"
    result = diagnose(desc)
    print(result.model_dump_json(indent=2))
