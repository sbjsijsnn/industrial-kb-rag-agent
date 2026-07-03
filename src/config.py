"""集中配置。所有可调参数都在这里,方便实验对比(写进 DECISIONS.md)。"""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# ---------- LLM (OpenAI 兼容接口: DeepSeek / Qwen / OpenAI 改 base_url 即可) ----------
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

# ---------- Embedding / Reranker (HuggingFace 本地推理) ----------
# 国内下载慢时设置环境变量 HF_ENDPOINT=https://hf-mirror.com
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
EMBEDDING_DIM = 1024  # bge-large-zh-v1.5 输出维度
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")

# ---------- Milvus ----------
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
COLLECTION_NAME = "industrial_manuals"

# ---------- 分块策略 (实验后的折衷值, 见 DECISIONS.md) ----------
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# ---------- 检索参数 ----------
VECTOR_TOP_K = 10      # 向量检索召回数
BM25_TOP_K = 10        # BM25 召回数
RERANK_TOP_N = 3       # 重排后送入 LLM 的条数
RRF_K = 60             # RRF 融合常数(论文默认 60)
RERANK_SCORE_THRESHOLD = 0.1  # 低于此分数视为"检索不到", 触发拒答

# ---------- 数据路径 ----------
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
CHUNKS_FILE = DATA_PROCESSED / "chunks.jsonl"
FAULT_DB = DATA_PROCESSED / "fault_codes.db"
