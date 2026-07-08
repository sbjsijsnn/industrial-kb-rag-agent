# 技术决策记录 (DECISIONS.md)

> 记录每个技术选型的理由和放弃的备选方案。面试被问"为什么用 X 不用 Y"时,这里就是答案。

## 1. 向量库: Milvus (而非 FAISS / Chroma / Pinecone)

- **FAISS**: 只是索引库,没有持久化/过滤/服务化,生产环境要自己包一层 — 适合实验不适合部署
- **Chroma**: 轻量好上手,但企业级特性(分区/权限/亿级扩展)弱
- **Pinecone**: 云服务,数据出境 — 工业客户的手册是敏感资产,必须私有化部署
- **Milvus** ✅: 开源可私有化、HNSW/IVF 索引可选、支持标量过滤(按设备型号过滤)、
  国内生态好(中文文档/社区)、招聘 JD 出现率最高

## 2. 索引: HNSW (而非 IVF_FLAT)

- 数据量 < 100 万时 HNSW 的召回率-延迟曲线优于 IVF
- 参数: M=16, efConstruction=200 (Milvus 官方推荐的均衡值)
- 代价: 内存占用比 IVF 高 — 本项目 chunk 数量级 1 万,可忽略
- 如果扩展到亿级: 换 DiskANN 或 IVF_PQ 量化压缩

## 3. chunk_size = 500, overlap = 100

- 尝试过 300 / 500 / 1000:
  - 300: 单个 chunk 上下文不完整,故障处理步骤常被切断
  - 1000: 检索粒度太粗,一个 chunk 混多个主题,Reranker 分数区分度下降
  - 500 ✅: 手册"一个小节"大致这个长度,语义完整
- overlap 100 (20%) 防止关键句正好被切在边界
- 分隔符优先级: 段落 > 换行 > 中文句号 — 中文文档不能用默认英文分隔符

## 4. 混合检索 + RRF (而非纯向量 / 加权求和)

- 纯向量的问题: 工业型号/故障码(如 "S7-1200"、"F0001")是**字面匹配需求**,
  embedding 对这类 token 语义表征弱,常召回错误型号的手册
- BM25 关键词通道正好补上; 中文必须先 jieba 分词, 否则整句是一个 token
- 融合用 **RRF** 而非加权求和: 向量分(余弦 0-1)和 BM25 分(无上界)量纲不同,
  加权求和要调归一化和权重两个超参; RRF 只看排名,零调参
- k=60 用论文默认值

## 5. Reranker: BGE-reranker-base (而非 large / 不用)

- 为什么要 Reranker: 双塔(bi-encoder)检索是"分别编码再比较",精度天花板低;
  Cross-Encoder 把 (query, chunk) 拼一起过模型,交互充分,精度高但慢 —
  所以只对召回的 ~20 条重排,不做全库扫描
- base vs large: large 精度略高但慢 3 倍、显存要求高。笔记本开发用 base,
  部署到 GPU 服务器可一行配置换 large
- 实测提升: 见 README 指标表 (rerank vs vector 差值)

## 6. Embedding: bge-large-zh-v1.5 (而非 OpenAI text-embedding-3)

- 手册是中文为主, bge-zh 系列在 C-MTEB 中文榜长期领先
- 本地推理零 API 成本 — 入库 1 万 chunk 若用 OpenAI API 又慢又花钱
- 私有化: 敏感文档不出内网
- 备选 M3E: bge 效果和社区活跃度都更好

## 7. LLM: DeepSeek API (OpenAI 兼容封装)

- 便宜(约 GPT-4 的 1/20)、中文强、国内直连
- 用 OpenAI SDK + base_url 抽象, 换 Qwen/GPT/Claude 只改 .env 三行
- temperature=0.2: 运维问答要稳定,不要创造性

## 8. 拒答阈值 (rerank_score < 0.1 拒答)

- 工业场景幻觉代价高(按错误步骤操作可能损坏设备)
- Reranker 分数是天然的"相关度置信度": 全部候选都低分 = 知识库没有这个内容
- 阈值 0.1 是初始值,应该用测试集里的"知识库外问题"校准(见 eval)

## 9. Agent: LangGraph create_react_agent (而非自写循环 / LangChain AgentExecutor)

- LangChain 旧版 AgentExecutor 已被官方标记为 legacy,新项目一律 LangGraph
- create_react_agent 自带: 工具调用循环、MemorySaver 多轮记忆、recursion_limit 防死循环
- recursion_limit=15: 三个工具的复合问题一般 4-6 步,15 是安全上限

## 10. BM25 语料和向量语料同源 (chunks.jsonl)

- 两条检索通道必须索引**同一份 chunk**,否则 RRF 融合时 key 对不上
- ingest 时同时写 Milvus 和 chunks.jsonl,保证一致性

## 11. 语料: 真实公开手册, 但英文版暂不入库

- 语料 = 西门子 S7-1200 系统手册中文版(1156 页, cache.industry.siemens.com 公开直链)
  + 三菱 FX3U 硬件手册中文版(8 页) — 后者页数少,主要作用是**跨品牌干扰项**,
  逼检索层区分"哪个品牌的手册"
- 同一手册的英文版(28MB)已下载但**不入库**: embedding 用的 bge-large-zh 是中文优化模型,
  英文 chunk 会拉低整体检索质量。多语种支持是 roadmap(换 bge-m3)
- 曾下载 FX3UC 中文手册(338 页),但该 PDF 字体无 Unicode cmap,pypdf 提取出 GBK 乱码 —
  **入库前必须抽样验证文本提取质量**,乱码文档需要 OCR(v1 不做,记入 roadmap)

## 12. 测试集: 逐题锚定真实手册页码 (而非拍脑袋编问题)

- 构建方法: 先全文扫描手册定位主题页(关键词出现≥2次) → 人工读该页原文 →
  只对"答案确实在这一页"的内容出题 → 记录 expected_page
- 这样 Top-3 命中判定(source + page±1)才是真判定; 编造的问题可能语料里根本没答案,
  测出来的召回率无意义
- 拒答测试集单独一份(testset_refusal.jsonl): 语料外品牌/无关问题,用于校准拒答阈值

## 13. Embedding 跑 CPU (暂不装 CUDA torch)

- 本机 GTX 1660 Ti 6GB 可以跑,但 pip 装 CUDA 版 torch 要多下 ~2.5GB
- 语料 ~7k chunk 量级, CPU batch 编码一次性 20-40 分钟,可接受(入库是低频操作)
- 如果语料涨到 10 万 chunk 再装 CUDA 版 — **不为一次性任务提前优化**

## 14. MCP Server: 知识库三工具服务化 (src/mcp_server.py)

- 把 search_manual / query_fault_code / get_repair_history 用 MCP 协议暴露,
  姊妹项目 industrial-ops-agents (多智能体运维系统) 作为 MCP 客户端消费
- 传输 stdio: 客户端拉起本 server 子进程, JSON-RPC 走标准输入输出 —
  **所以 server 的一切日志必须打到 stderr, stdout 是协议通道**
- 启动即预热检索链路: 否则客户端首次调用要等 20-30s 模型加载, 易触发超时

## 15. 模型加载: 本地缓存优先 (_local_or_remote)

- 问题: SentenceTransformer(模型名) 冷启动时会向 HuggingFace Hub 发校验请求,
  镜像站不稳时**明明模型在本地却加载失败**
- 修复: snapshot_download(local_files_only=True) 纯本地解析缓存路径, 零网络请求;
  无缓存时回退模型名联网下载
- 弯路: 先试了 HF_HUB_OFFLINE=1 — 结果缓存中"本就不存在"的可选文件(modules.json)
  在离线模式下抛错而非跳过, 更糟。**离线开关不是免费的加速器**

## 未来优化 (面试可聊的 roadmap)

- [ ] 父子块检索: 小块检索、大块生成,兼顾精度和上下文
- [ ] 查询改写 (HyDE): 口语化提问先改写成手册风格再检索
- [ ] 语义缓存: Redis + 向量相似度,高频问题直接回缓存
- [ ] 表格解析: 手册中的参数表用 unstructured 单独提取
- [ ] MCP Server: 把三个工具用 MCP 协议暴露,支持 Claude/千问等任意客户端接入
- [ ] 多语种 embedding (bge-m3): 让英文手册也能入库
- [ ] OCR 管道: 处理字体无 Unicode 映射的老 PDF (如 FX3UC 手册)
