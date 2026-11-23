"""第 2/3 步：建索引 + 混合检索。

为什么要“向量 + BM25”混合：
- 向量（bge-m3）懂语义：“怎么升级固件” 能召回写着 “OTA / firmware update” 的代码。
- BM25 懂精确词：UDS 里 0x22、DID F190、函数名 parseDifop 这种标识符，向量模型经常对不准。
- 两路结果用 RRF（倒数排名融合）合并：不需要把两种分数归一化到同一尺度，鲁棒、零调参。
"""
import json
import re
from pathlib import Path

from .ingest import Chunk, load_chunks

IDENT = re.compile(r"0x[0-9a-fA-F]+|[A-Za-z_][A-Za-z0-9_]*|\d+")
CJK = re.compile(r"[\u4e00-\u9fff]+")
CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
RRF_K = 60
# 提问套话在问题里出现、在代码里几乎不出现：不切掉的话，“怎么/么编”这种二元组会把中文文档整体抬高
CJK_STOP = re.compile(r"为什么|是怎么|怎么样|如何|怎么|怎样|什么|哪些|哪里|是否|可以|一下|的|了|吗|呢|是|和|在")


def tokenize(text):
    """代码友好分词：保留完整标识符，同时拆 camelCase / snake_case；中文去套话后用二元组。"""
    tokens = []
    for word in IDENT.findall(text):
        low = word.lower()
        tokens.append(low)
        if low.startswith("0x"):
            continue
        parts = [p.lower() for piece in word.split("_") for p in CAMEL.findall(piece)]
        if len(parts) > 1:
            tokens.extend(parts)
    for run in CJK.findall(text):
        for piece in CJK_STOP.split(run):
            tokens.extend(piece[i:i + 2] for i in range(max(1, len(piece) - 1)) if piece)
    return tokens


def route_projects(query, aliases):
    """问题里点名了项目就只在该项目里搜：最便宜、最确定的召回提升手段（先路由，再检索）。"""
    q = query.lower()
    return {p for p, names in aliases.items() if any(n.lower() in q for n in [p, *names])}


def rrf(rankings, k=RRF_K):
    """rankings: 多路有序 id 列表。分数 = Σ 1/(k + 名次)。"""
    scores = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda cid: scores[cid], reverse=True)


class Embedder:
    def __init__(self, model_path, device="cuda"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(str(model_path), device=device)
        if device == "cuda":
            self.model.half()  # FP16 推理：显存减半、速度翻倍，检索精度几乎不变

    def encode(self, texts, batch_size=32):
        return self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=len(texts) > 256)


def build_index(index_dir, embed_model, max_seq_len=1024):
    import faiss
    import numpy as np

    index_dir = Path(index_dir)
    chunks = load_chunks(index_dir / "chunks.jsonl")
    embedder = Embedder(embed_model)
    embedder.model.max_seq_length = max_seq_len  # 块 ≤3000 字符，1024 token 足够；越长越慢
    vectors = embedder.encode([c.embed_text() for c in chunks]).astype(np.float32)
    index = faiss.IndexFlatIP(vectors.shape[1])  # 已归一化，内积 = 余弦；几万块以内暴力检索 <10ms
    index.add(vectors)
    faiss.write_index(index, str(index_dir / "vectors.faiss"))
    (index_dir / "manifest.json").write_text(json.dumps(
        {"embed_model": str(embed_model), "chunks": len(chunks), "dim": int(vectors.shape[1])}, indent=1), encoding="utf-8")
    return len(chunks)


class Retriever:
    def __init__(self, index_dir, embedder=None, use_vectors=True):
        from rank_bm25 import BM25Okapi

        self.index_dir = Path(index_dir)
        self.chunks = load_chunks(self.index_dir / "chunks.jsonl")
        self.by_id = {c.id: c for c in self.chunks}
        self.files = json.loads((self.index_dir / "files.json").read_text(encoding="utf-8"))
        projects = self.index_dir / "projects.json"
        self.aliases = json.loads(projects.read_text(encoding="utf-8")) if projects.exists() else {}
        self.route = True
        self.bm25 = BM25Okapi([tokenize(c.embed_text()) for c in self.chunks])
        self.index = self.embedder = None
        if use_vectors:
            import faiss
            manifest = json.loads((self.index_dir / "manifest.json").read_text(encoding="utf-8"))
            self.index = faiss.read_index(str(self.index_dir / "vectors.faiss"))
            self.embedder = embedder or Embedder(manifest["embed_model"])

    def _bm25_rank(self, query, n, scope):
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.chunks[i].id for i in order if scores[i] > 0 and self._in(i, scope)][:n]

    def _vector_rank(self, query, n, scope):
        vec = self.embedder.encode([query])
        _, idx = self.index.search(vec, n * 8 if scope else n)  # 过滤前多取，过滤后仍有 n 条
        return [self.chunks[i].id for i in idx[0] if i >= 0 and self._in(i, scope)][:n]

    def _in(self, i, scope):
        return not scope or self.chunks[i].project.lower() in scope

    def search(self, query, k=8, project=None, pool=50):
        scope = {project.lower()} if project else {p.lower() for p in route_projects(query, self.aliases)} if self.route else set()
        # 套路：先在每一路内部按项目过滤，再做 RRF。若先融合后过滤，深层名次的“两路都沾边”的块会挤掉单路第一名
        rankings = [self._bm25_rank(query, pool, scope)]
        if self.index is not None:
            rankings.append(self._vector_rank(query, pool, scope))
        return dedupe([self.by_id[cid] for cid in rrf(rankings)])[:k]

    def read(self, ref, start, end, max_lines=150):
        """read 工具：只允许读入库白名单里的文件，并限制行数，防止 Agent 越权读盘/塞爆上下文。"""
        path = self.files.get(ref)
        if path is None:
            return None
        start = max(1, int(start))
        end = min(int(end), start + max_lines - 1)
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
        body = "\n".join(lines[start - 1:end])
        project, rel = ref.split("/", 1)
        return Chunk("read", project, rel, start, min(end, len(lines)), "read", "", body)


def dedupe(chunks):
    """同一文件里行号重叠的块只留排名最高的那个，把上下文名额让给别的证据。"""
    kept = []
    for c in chunks:
        if any(k.project == c.project and k.path == c.path and k.start <= c.end and c.start <= k.end for k in kept):
            continue
        kept.append(c)
    return kept
