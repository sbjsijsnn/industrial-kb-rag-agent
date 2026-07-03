"""数据摄入管道: PDF -> 分块 -> embedding -> Milvus 入库 + chunks.jsonl (给 BM25 用)。

用法:
    python -m src.ingest              # 处理 data/raw 下所有 PDF
    python -m src.ingest --drop       # 先清空集合再重建
"""
import argparse
import json
import sys

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymilvus import DataType, MilvusClient
from sentence_transformers import SentenceTransformer

from src import config


def load_and_split() -> list[dict]:
    """加载 data/raw 下所有 PDF, 切分为 chunk 列表。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", ";", " ", ""],  # 中文文档优先按段/句切
    )
    chunks = []
    pdfs = sorted(config.DATA_RAW.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"data/raw 下没有 PDF, 先放几份设备手册进去: {config.DATA_RAW}")
    for pdf in pdfs:
        print(f"[load] {pdf.name}")
        docs = PyPDFLoader(str(pdf)).load()
        for piece in splitter.split_documents(docs):
            text = piece.page_content.strip()
            if len(text) < 30:  # 过滤目录页/页眉页脚碎片
                continue
            chunks.append({
                "text": text,
                "source": pdf.name,
                "page": piece.metadata.get("page", -1) + 1,  # 转 1-based 页码
            })
    print(f"[split] 共 {len(chunks)} 个 chunk (chunk_size={config.CHUNK_SIZE})")
    return chunks


def build_collection(client: MilvusClient, drop: bool):
    if drop and client.has_collection(config.COLLECTION_NAME):
        client.drop_collection(config.COLLECTION_NAME)
    if client.has_collection(config.COLLECTION_NAME):
        return
    schema = client.create_schema(auto_id=True)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("text", DataType.VARCHAR, max_length=4000)
    schema.add_field("source", DataType.VARCHAR, max_length=256)
    schema.add_field("page", DataType.INT64)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=config.EMBEDDING_DIM)
    index_params = client.prepare_index_params()
    # HNSW: 内存型图索引, 百万级以下召回率/延迟折衷最好 (vs IVF_FLAT), 见 DECISIONS.md
    index_params.add_index("vector", index_type="HNSW", metric_type="COSINE",
                           params={"M": 16, "efConstruction": 200})
    client.create_collection(config.COLLECTION_NAME, schema=schema, index_params=index_params)
    print(f"[milvus] 集合 {config.COLLECTION_NAME} 已创建 (HNSW/COSINE)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drop", action="store_true", help="清空集合重建")
    args = parser.parse_args()

    chunks = load_and_split()

    # 保存 chunks.jsonl — BM25 检索和评测都要用同一份语料
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    with open(config.CHUNKS_FILE, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"[save] {config.CHUNKS_FILE}")

    print(f"[embed] 加载 {config.EMBEDDING_MODEL} (首次运行会下载模型)...")
    encoder = SentenceTransformer(config.EMBEDDING_MODEL)
    vectors = encoder.encode(
        [c["text"] for c in chunks],
        batch_size=32, show_progress_bar=True, normalize_embeddings=True,
    )

    client = MilvusClient(uri=config.MILVUS_URI)
    build_collection(client, drop=args.drop)
    rows = [{**c, "vector": v.tolist()} for c, v in zip(chunks, vectors)]
    for i in range(0, len(rows), 500):
        client.insert(config.COLLECTION_NAME, rows[i:i + 500])
    print(f"[milvus] 已插入 {len(rows)} 条")


if __name__ == "__main__":
    main()
