"""层4: 评测脚本。对比三种检索模式的召回率 — 差值就是简历上的量化指标。

用法:
    1. 手动编写 src/eval/testset.jsonl (30-50 条), 每行:
       {"question": "...", "expected_source": "手册文件名.pdf", "expected_page": 12}
       (expected_page 可选; 标注"这个问题的答案在哪份手册哪一页")
    2. python -m src.eval.run_eval

输出: 每种模式的 Top-3 命中率 + 平均延迟。
"""
import json
import time
from pathlib import Path

from src.retrieval import retrieve

TESTSET = Path(__file__).parent / "testset.jsonl"


def hit(chunks: list[dict], expected_source: str, expected_page: int | None) -> bool:
    """命中判定: Top-N 里有没有来自正确手册(和页码±1)的 chunk。"""
    for c in chunks:
        if expected_source.lower() not in c["source"].lower():
            continue
        if expected_page is None or abs(c["page"] - expected_page) <= 1:
            return True
    return False


def main():
    if not TESTSET.exists():
        print(f"先创建测试集: {TESTSET}")
        print('每行: {"question": "...", "expected_source": "xx.pdf", "expected_page": 12}')
        return
    cases = [json.loads(line) for line in open(TESTSET, encoding="utf-8") if line.strip()]
    print(f"测试集: {len(cases)} 条\n")

    # 预热: 首次调用要加载 embedding/reranker 模型和构建 BM25 索引 (~10s),
    # 不预热的话这段时间会被算进第一条查询, 污染平均延迟
    print("[warmup] 加载模型与索引...")
    retrieve("预热查询", mode="rerank")

    print(f'{"mode":<10}{"Top-3命中率":<14}{"平均延迟(ms)":<12}')
    print("-" * 36)
    for mode in ["vector", "hybrid", "rerank"]:
        hits, total_ms = 0, 0.0
        for case in cases:
            t0 = time.time()
            chunks = retrieve(case["question"], mode=mode)
            total_ms += (time.time() - t0) * 1000
            if hit(chunks, case["expected_source"], case.get("expected_page")):
                hits += 1
        rate = hits / len(cases) * 100
        print(f'{mode:<10}{rate:>8.1f}%     {total_ms / len(cases):>8.0f}')

    print("\n把这张表填进 README 的量化指标区 (rerank vs vector 的差值就是 Reranker 的提升)")


if __name__ == "__main__":
    main()
