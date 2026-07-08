"""检索层: 向量检索 + BM25 关键词检索 -> RRF 融合 -> BGE-Reranker 重排。

三种模式可对比 (评测脚本会分别测, 差值就是简历上的量化指标):
    vector  : 纯向量
    hybrid  : 向量 + BM25 + RRF
    rerank  : hybrid 之后再过 Reranker (默认, 效果最好)
"""
import json
from functools import lru_cache

import jieba
from pymilvus import MilvusClient
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from src import config


def _local_or_remote(model_id: str) -> str:
    """模型已有本地缓存时直接返回缓存路径, 彻底跳过 HuggingFace Hub 的联网检查。

    背景: SentenceTransformer(模型名) 每次冷启动都会向 Hub 发 HEAD 请求校验文件,
    镜像站不稳时直接加载失败 — 但模型明明在本地。snapshot_download(local_files_only=True)
    纯本地解析缓存, 不发任何网络请求; 没有缓存时才回退到模型名(联网下载)。
    注意不能用 HF_HUB_OFFLINE=1: 缓存中"本就不存在"的可选文件在离线模式下会抛错而非跳过。
    """
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(model_id, local_files_only=True)
    except Exception:
        return model_id


@lru_cache(maxsize=1)
def _encoder() -> SentenceTransformer:
    return SentenceTransformer(_local_or_remote(config.EMBEDDING_MODEL))


@lru_cache(maxsize=1)
def _reranker() -> CrossEncoder:
    return CrossEncoder(_local_or_remote(config.RERANKER_MODEL))


@lru_cache(maxsize=1)
def _milvus() -> MilvusClient:
    return MilvusClient(uri=config.MILVUS_URI)


@lru_cache(maxsize=1)
def _bm25_index():
    """从 chunks.jsonl 构建 BM25。中文必须先分词(jieba), 否则整句算一个 token。"""
    chunks = [json.loads(line) for line in open(config.CHUNKS_FILE, encoding="utf-8")]
    tokenized = [list(jieba.cut(c["text"])) for c in chunks]
    return BM25Okapi(tokenized), chunks


def _chunk_key(c: dict) -> str:
    return f'{c["source"]}#p{c["page"]}#{c["text"][:40]}'


def vector_search(query: str, top_k: int = config.VECTOR_TOP_K) -> list[dict]:
    qv = _encoder().encode([query], normalize_embeddings=True)[0].tolist()
    res = _milvus().search(
        config.COLLECTION_NAME, data=[qv], limit=top_k,
        output_fields=["text", "source", "page"],
    )
    return [
        {"text": h["entity"]["text"], "source": h["entity"]["source"],
         "page": h["entity"]["page"], "score": h["distance"]}
        for h in res[0]
    ]


def bm25_search(query: str, top_k: int = config.BM25_TOP_K) -> list[dict]:
    bm25, chunks = _bm25_index()
    scores = bm25.get_scores(list(jieba.cut(query)))
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [{**chunks[i], "score": float(scores[i])} for i in top_idx if scores[i] > 0]


def rrf_fuse(result_lists: list[list[dict]], k: int = config.RRF_K) -> list[dict]:
    """Reciprocal Rank Fusion: score = Σ 1/(k + rank)。只看排名不看原始分数,
    所以不需要归一化向量分和 BM25 分 — 这是选 RRF 而非加权求和的原因。"""
    fused: dict[str, dict] = {}
    for results in result_lists:
        for rank, item in enumerate(results):
            key = _chunk_key(item)
            if key not in fused:
                fused[key] = {**item, "rrf_score": 0.0}
            fused[key]["rrf_score"] += 1.0 / (k + rank + 1)
    return sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)


def rerank(query: str, candidates: list[dict], top_n: int = config.RERANK_TOP_N) -> list[dict]:
    """Cross-Encoder 逐对打分 (query, chunk)。比向量检索准得多但慢, 所以只重排候选集。"""
    if not candidates:
        return []
    scores = _reranker().predict([(query, c["text"]) for c in candidates])
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return ranked[:top_n]


def retrieve(query: str, mode: str = "rerank") -> list[dict]:
    """统一入口。mode: vector / hybrid / rerank"""
    if mode == "vector":
        return vector_search(query)[:config.RERANK_TOP_N]
    fused = rrf_fuse([vector_search(query), bm25_search(query)])
    if mode == "hybrid":
        return fused[:config.RERANK_TOP_N]
    return rerank(query, fused[:config.VECTOR_TOP_K + config.BM25_TOP_K])


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "S7-1200 通信故障怎么排查"
    for mode in ["vector", "hybrid", "rerank"]:
        print(f"\n===== mode={mode} =====")
        for r in retrieve(q, mode):
            print(f'  [{r["source"]} p{r["page"]}] {r["text"][:60]}...')
