"""第 2/3 步：建索引 + 混合检索。

为什么要“向量 + BM25”混合：
- 向量（bge-m3）懂语义：“怎么升级固件” 能召回写着 “OTA / firmware update” 的代码。
- BM25 懂精确词：UDS 里 0x22、DID F190、函数名 parseDifop 这种标识符，向量模型经常对不准。
- 两路结果用 RRF（倒数排名融合）合并：不需要把两种分数归一化到同一尺度，鲁棒、零调参。
"""
import hashlib
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


class Reranker:
    """交叉编码器重排：把“问题 + 候选块”拼在一起逐对打分，比双塔向量精确得多，但慢，只用在前几十个候选上。

    套路：召回要“宽”（BM25+向量各取 50，RRF 融合），精排要“准”（cross-encoder 只看融合后的前 N 个）。
    """
    def __init__(self, model_path, device="cuda", max_length=512):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(str(model_path), device=device, max_length=max_length)
        if device == "cuda":
            self.model.model.half()

    def scores(self, query, texts, batch_size=16):
        return self.model.predict([(query, t) for t in texts], batch_size=batch_size, show_progress_bar=False).tolist()


def build_index(index_dir, embed_model, max_seq_len=1024, embedder=None):
    """增量向量化：按块内容 hash 复用上次的向量，只给新增/改动的块重新编码。

    套路：块 id 由“项目:路径:行号”决定，内容改了 id 可能不变、内容没改行号却可能平移，
    所以复用的键必须是内容本身的 hash（embed_text 含路径和标题），而不是 id。
    换了嵌入模型（manifest 不一致）时整体重算，因为新旧向量不在同一个空间里。
    """
    import faiss
    import numpy as np

    index_dir = Path(index_dir)
    chunks = load_chunks(index_dir / "chunks.jsonl")
    keys = [hashlib.sha1(c.embed_text().encode()).hexdigest() for c in chunks]
    old, old_vecs = {}, None
    kfile, vfile, mfile = index_dir / "vec_keys.json", index_dir / "vectors.faiss", index_dir / "manifest.json"
    if kfile.exists() and vfile.exists() and mfile.exists() and \
            json.loads(mfile.read_text(encoding="utf-8")).get("embed_model") == str(embed_model):
        prev = faiss.read_index(str(vfile))
        old_vecs = prev.reconstruct_n(0, prev.ntotal)
        old = {k: i for i, k in enumerate(json.loads(kfile.read_text(encoding="utf-8")))}
    todo = [i for i, k in enumerate(keys) if k not in old]
    new_vecs = None
    if todo:  # 一块都没变就不加载 bge-m3，连显存都不占；服务内重建时复用已加载的那份
        embedder = embedder or Embedder(embed_model)
        embedder.model.max_seq_length = max_seq_len  # 块 ≤3000 字符，1024 token 足够；越长越慢
        new_vecs = embedder.encode([chunks[i].embed_text() for i in todo]).astype(np.float32)
    dim = new_vecs.shape[1] if new_vecs is not None else old_vecs.shape[1]
    vectors = np.empty((len(chunks), dim), dtype=np.float32)
    for j, i in enumerate(todo):
        vectors[i] = new_vecs[j]
    for i, k in enumerate(keys):
        if k in old:
            vectors[i] = old_vecs[old[k]]
    index = faiss.IndexFlatIP(dim)  # 已归一化，内积 = 余弦；几万块以内暴力检索 <10ms
    index.add(vectors)
    faiss.write_index(index, str(vfile))
    kfile.write_text(json.dumps(keys), encoding="utf-8")
    stats = {"embed_model": str(embed_model), "chunks": len(chunks), "dim": int(dim), "encoded": len(todo), "reused": len(chunks) - len(todo)}
    mfile.write_text(json.dumps(stats, indent=1), encoding="utf-8")
    return stats


class Retriever:
    def __init__(self, index_dir, embedder=None, use_vectors=True, reranker=None, rerank_top=30):
        self.index_dir = Path(index_dir)
        self.use_vectors = use_vectors
        self.reranker, self.rerank_top = reranker, rerank_top
        self.route = True
        self.disabled = set()  # 后台关掉的项目（小写），检索时整体跳过
        self.index = self.embedder = None
        self._embedder = embedder
        self.reload()

    def reload(self):
        """重新读盘（ingest/index 之后调用），复用已加载的 bge-m3，不用重启服务。
        套路：先在局部变量里建好新状态，最后一次性替换属性，避免并发请求看到“半新半旧”的索引。"""
        from rank_bm25 import BM25Okapi

        chunks = load_chunks(self.index_dir / "chunks.jsonl")
        files = json.loads((self.index_dir / "files.json").read_text(encoding="utf-8"))
        projects = self.index_dir / "projects.json"
        aliases = json.loads(projects.read_text(encoding="utf-8")) if projects.exists() else {}
        bm25 = BM25Okapi([tokenize(c.embed_text()) for c in chunks])
        index = None
        if self.use_vectors:
            import faiss
            manifest = json.loads((self.index_dir / "manifest.json").read_text(encoding="utf-8"))
            index = faiss.read_index(str(self.index_dir / "vectors.faiss"))
            if index.ntotal != len(chunks):
                raise RuntimeError(f"向量数 {index.ntotal} 与块数 {len(chunks)} 不一致：ingest 之后要先跑 index")
            if self.embedder is None:
                self.embedder = self._embedder or Embedder(manifest["embed_model"])
        by_file, summary_of = {}, {}
        for c in chunks:
            if c.kind != "summary":
                by_file.setdefault((c.project, c.path), []).append(c)
            else:
                summary_of[(c.project, c.path)] = c
        self.chunks, self.by_id, self.files, self.aliases, self.bm25, self.index = \
            chunks, {c.id: c for c in chunks}, files, aliases, bm25, index
        self.by_file, self.summary_of, self.pos = by_file, summary_of, {c.id: i for i, c in enumerate(chunks)}
        return len(chunks)

    def expand(self, project):
        """选中产品 = 产品本身 + 把它当别名的资料源（如 LidarAssistant → LidarAssistant-客户资料）。"""
        p = project.lower()
        return {p} | {name.lower() for name, al in self.aliases.items() if p in (a.lower() for a in al)}

    def _bm25_rank(self, query, n, scope):
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.chunks[i].id for i in order if scores[i] > 0 and self._in(i, scope)][:n]

    def _vector_rank(self, query, n, scope):
        vec = self.embedder.encode([query])
        _, idx = self.index.search(vec, n * 8 if scope or self.disabled else n)  # 过滤前多取，过滤后仍有 n 条
        return [self.chunks[i].id for i in idx[0] if i >= 0 and self._in(i, scope)][:n]

    def _in(self, i, scope):
        p = self.chunks[i].project.lower()
        return p not in self.disabled and (not scope or p in scope)

    def search(self, query, k=8, project=None, pool=50):
        scope = self.expand(project) if project else {p.lower() for p in route_projects(query, self.aliases)} if self.route else set()
        # 套路：先在每一路内部按项目过滤，再做 RRF。若先融合后过滤，深层名次的“两路都沾边”的块会挤掉单路第一名
        rankings = [self._bm25_rank(query, pool, scope)]
        if self.index is not None:
            rankings.append(self._vector_rank(query, pool, scope))
        fused = dedupe([self.by_id[cid] for cid in rrf(rankings)])
        if self.reranker is None or len(fused) <= 1:
            return self._resolve_summaries(query, fused[:k])
        head = fused[:self.rerank_top]
        s = self.reranker.scores(query, [c.embed_text()[:1500] for c in head])
        return self._resolve_summaries(query, [c for _, c in sorted(zip(s, head), key=lambda x: -x[0])][:k])

    def _resolve_summaries(self, query, results):
        """概要块只是“检索代理”：负责把口语问题引到正确的文件，喂给模型的必须是真实代码。

        套路（small-to-big / 代理索引）：用短而“像问题”的文本做召回，命中后换回原文。
        命中概要 → 在同文件的代码块里挑与问题最相关的一块替换；该文件代码块已全在结果里则丢弃概要。
        """
        if not any(c.kind == "summary" for c in results):
            return results
        taken = {c.id for c in results}
        out = []
        for c in results:
            if c.kind != "summary":
                out.append(c)
                continue
            cand = [x for x in self.by_file.get((c.project, c.path), []) if x.id not in taken]
            if not cand:
                continue
            if len(cand) > 1:
                if self.reranker is not None:
                    s = self.reranker.scores(query, [x.embed_text()[:1500] for x in cand[:20]])
                else:
                    s = list(self.bm25.get_batch_scores(tokenize(query), [self.pos[x.id] for x in cand[:20]]))
                cand = [x for _, x in sorted(zip(s, cand[:20]), key=lambda t: -t[0])]
            taken.add(cand[0].id)
            out.append(cand[0])
        return out

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
    """同一文件里行号重叠的块只留排名最高的那个，把上下文名额让给别的证据。

    文件概要块（kind=summary）覆盖整文件，不参与重叠判断，否则它一命中就会把该文件真正的代码块全挤掉；
    同一文件的概要只留一个。
    """
    kept = []
    for c in chunks:
        if any(k.project == c.project and k.path == c.path and (k.kind == c.kind == "summary" or
               ("summary" not in (k.kind, c.kind) and k.start <= c.end and c.start <= k.end)) for k in kept):
            continue
        kept.append(c)
    return kept
