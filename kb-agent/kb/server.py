"""第 6 步：网页问答服务（局域网版）。

- 前台 /：提问、Markdown 回答、复制/点赞/收藏、问题侧栏、热门榜单。
- 后台 /admin：项目开关、引用源可见性、榜单管理、统计、待补资料、一键重新入库（热更新，不停服务）。
- GPU 生成串行化（一把锁），避免并发请求把显存打爆。

套路：对外接口只吐“过滤后”的来源；库里存完整来源，开关随时可改。
"""
import json
import shutil
import threading
import time
from pathlib import Path

HERE = Path(__file__).parent
REBUILD_FILES = ("vectors.faiss", "vec_keys.json", "manifest.json")


def serve(args, retriever, llm):
    import uvicorn
    from fastapi import Depends, FastAPI, Header, HTTPException, Request
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field

    from . import cards, projects
    from .agent import KnowledgeAgent
    from .cli import SUMMARIES, cards_dir
    from .store import Store

    store = Store(args.data_dir)
    agent = KnowledgeAgent(retriever, llm, k=args.k)
    gpu = threading.Lock()
    job = {"state": "idle", "log": [], "started": None, "finished": None}
    app = FastAPI(title="研发知识库助手")
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")

    def apply_settings():
        store.sync_projects(sorted({c.project for c in retriever.chunks}))
        retriever.disabled = {n.lower() for n, p in store.settings["projects"].items() if not p.get("enabled", True)}

    apply_settings()
    token = store.admin_token()

    # ---------- 工具 ----------
    def client_id(x_client: str = Header(default="")):
        return x_client[:64] if x_client else "anon"

    def require_admin(x_admin_token: str = Header(default="")):
        if x_admin_token != token:
            raise HTTPException(401, "管理员口令错误")

    def public_sources(sources):
        out = []
        for s in sources:
            if store.project(s["project"])["show_source"]:
                out.append(s)
            else:
                out.append({"n": s["n"], "hidden": True, "project": store.project(s["project"])["label"] or "内部资料"})
        return out

    def public_qa(d, client):
        return {"qa_id": d["id"], "question": d["question"], "answer": d["answer"], "project": d["project"],
                "sources": public_sources(d["sources"]), "seconds": d["seconds"], "ts": d["ts"],
                "likes": d["likes"], "dislikes": d["dislikes"], "favs": d["favs"], "mine": store.my_votes(d["id"], client)}

    def visible_projects():
        return [{"name": n, "label": p.get("label") or n} for n, p in sorted(store.settings["projects"].items())
                if p.get("enabled", True) and p.get("listed", True) and n in {c.project for c in retriever.chunks}]

    # ---------- 前台 ----------
    @app.get("/", response_class=HTMLResponse)
    def index():
        return (HERE / "chat.html").read_text(encoding="utf-8")

    @app.get("/api/config")
    def config():
        s = store.settings
        return {"title": s["title"], "notice": s["notice"], "examples": s["examples"],
                "allow_agent_mode": s["allow_agent_mode"], "projects": visible_projects(), "chunks": len(retriever.chunks)}

    class Ask(BaseModel):
        question: str = Field(min_length=1, max_length=2000)
        project: str | None = None
        mode: str = Field(default="rag", pattern="^(rag|agent)$")

    @app.post("/api/ask")
    def ask(req: Ask, request: Request, client: str = Depends(client_id)):
        if req.project and req.project not in {p["name"] for p in visible_projects()}:
            raise HTTPException(400, "该项目未开放")
        mode = req.mode if store.settings["allow_agent_mode"] else "rag"
        if not gpu.acquire(timeout=300):
            raise HTTPException(503, "当前提问人数较多，请稍后再试")
        try:
            res = agent.ask(req.question.strip(), project=req.project or None, mode=mode)
        finally:
            gpu.release()
        qa_id = store.add_qa(client, request.client.host if request.client else "", req.question.strip(), req.project, mode, res)
        if "没有找到" in res["answer"]:
            auto_queue("miss", store.get_qa(qa_id), "模型回答“资料中没有找到”")
        return public_qa(store.get_qa(qa_id), client)

    def auto_queue(origin, d, note):
        """系统自己发现的“不会的题”进审核队列：证据前 3 条作候选出处，管理员改两笔就能通过。
        套路：让错误主动来找人，而不是等人去日志里翻；同一问题待审中只留一条。"""
        if store.has_pending(None, d["question"]):
            return
        store.add_feedback(origin, d["question"], d["project"], "", d["evidence"][:3], note=note + "；出处为当时证据前 3 条，需核对",
                           qa_id=d["id"], client=d["client"])

    @app.get("/api/qa/{qa_id}")
    def get_qa(qa_id: int, client: str = Depends(client_id)):
        d = store.get_qa(qa_id)
        if not d:
            raise HTTPException(404, "记录不存在")
        return public_qa(d, client)

    class Vote(BaseModel):
        qa_id: int
        kind: str = Field(pattern="^(like|dislike|fav)$")
        on: bool = True

    @app.post("/api/vote")
    def vote(v: Vote, client: str = Depends(client_id)):
        if client == "anon" or not store.get_qa(v.qa_id):
            raise HTTPException(400, "无效请求")
        res = store.vote(v.qa_id, client, v.kind, v.on)
        if v.kind == "dislike" and v.on:
            auto_queue("dislike", store.get_qa(v.qa_id), "用户点踩")
        return res

    @app.get("/api/history")
    def history(client: str = Depends(client_id)):
        return {"items": store.history(client)}

    @app.get("/api/favorites")
    def favorites(client: str = Depends(client_id)):
        return {"items": store.favorites(client)}

    @app.get("/api/board")
    def board():
        hot = [{k: h[k] for k in ("question", "qa_id", "users", "asks", "likes")} for h in store.hot(limit=10)]
        return {"board": store.board(), "hot": hot}

    # ---------- 后台 ----------
    @app.get("/admin", response_class=HTMLResponse)
    def admin_page():
        return (HERE / "admin.html").read_text(encoding="utf-8")

    @app.get("/api/admin/overview", dependencies=[Depends(require_admin)])
    def overview():
        src = retriever.index_dir / "sources.json"
        sources = json.loads(src.read_text(encoding="utf-8")) if src.exists() else {}
        return {"settings": store.settings, "stats": store.stats(), "sources": sources, "job": job,
                "indexed": sorted({c.project for c in retriever.chunks})}

    @app.put("/api/admin/settings", dependencies=[Depends(require_admin)])
    def put_settings(body: dict):
        store.save_settings(body)
        apply_settings()
        return store.settings

    @app.get("/api/admin/questions", dependencies=[Depends(require_admin)])
    def questions(kind: str = "recent", q: str = ""):
        if kind == "hot":
            return {"items": store.hot(limit=50), "board": store.board()}
        if kind == "misses":
            return {"items": store.misses()}
        return {"items": store.recent(keyword=q)}

    @app.get("/api/admin/qa/{qa_id}", dependencies=[Depends(require_admin)])
    def admin_qa(qa_id: int):
        return store.get_qa(qa_id) or {}

    class Feature(BaseModel):
        qa_id: int
        action: str = Field(pattern="^(pin|hide|reset)$")

    @app.post("/api/admin/feature", dependencies=[Depends(require_admin)])
    def feature(f: Feature):
        d = store.get_qa(f.qa_id)
        if not d:
            raise HTTPException(404, "记录不存在")
        if f.action == "reset":
            store.unflag(d["norm"])
        else:
            store.feature(d["norm"], d["question"], d["id"], f.action == "pin")
        return {"board": store.board()}

    @app.post("/api/admin/rebuild", dependencies=[Depends(require_admin)])
    def rebuild(accept: bool = False):
        if job["state"] == "running":
            raise HTTPException(409, "已有任务在运行")
        job.update(state="running", log=[], started=time.time(), finished=None)
        threading.Thread(target=run_rebuild, args=(accept,), daemon=True).start()
        return job

    def run_rebuild(accept):
        """在 index.next/ 里重建并评测，过卡口后再替换正式索引并热加载；不过卡口则线上不受影响。"""
        from .cli import evaluate, gate, load_questions
        from .ingest import ingest
        from .retrieve import Retriever, build_index

        log = job["log"].append
        live, nxt = retriever.index_dir, retriever.index_dir.with_name(retriever.index_dir.name + ".next")
        try:
            nxt.mkdir(exist_ok=True)
            for f in REBUILD_FILES:  # 带上旧向量，增量编码只算改动的块
                if (live / f).exists():
                    shutil.copy2(live / f, nxt / f)
            t = time.time()
            stats = ingest(args.config, nxt / "chunks.jsonl", SUMMARIES)
            log(f"切块完成 {sum(s['chunks'] for s in stats.values())} 块，{time.time() - t:.1f}s")
            for name, s in stats.items():
                log(f"  {name}: {s['files']} 文件 / {s['chunks']} 块  {s.get('version') or ''}")
            with gpu:  # 向量化与问答共用 GPU：期间提问会排队
                t = time.time()
                embed_model = json.loads((live / "manifest.json").read_text(encoding="utf-8"))["embed_model"]
                s = build_index(nxt, embed_model, embedder=retriever.embedder)
                log(f"向量化：新编码 {s['encoded']}，复用 {s['reused']}，{time.time() - t:.1f}s")
                result = evaluate(Retriever(nxt, embedder=retriever.embedder, reranker=retriever.reranker,
                                            rerank_top=retriever.rerank_top), load_questions(config=args.config))
            log("评测：" + json.dumps(result["summary"], ensure_ascii=False))
            ok, msg = gate(result, accept=accept)
            log(msg)
            if not ok:
                job["state"] = "gate_failed"
                return
            for f in nxt.iterdir():
                f.replace(live / f.name)  # 同一文件系统内 rename 是原子的
            n = retriever.reload()
            apply_settings()
            log(f"已热加载：{n} 块")
            n_stale = queue_stale()
            if n_stale:
                log(f"发现 {n_stale} 张卡片引用的代码已变化，已放入审核队列")
            job["state"] = "done"
        except Exception as e:  # 后台任务失败要让管理员看到原因，而不是静默
            log(f"失败：{type(e).__name__}: {e}")
            job["state"] = "failed"
        finally:
            job["finished"] = time.time()

    @app.post("/api/admin/reload", dependencies=[Depends(require_admin)])
    def reload():
        with gpu:
            n = retriever.reload()
        apply_settings()
        return {"chunks": n}

    # ---------- 纠错 → 分类 → 审核 → 知识卡片 ----------
    cdir = cards_dir(args.config)

    def start_rebuild():
        if job["state"] != "running":
            job.update(state="running", log=[], started=time.time(), finished=None)
            threading.Thread(target=run_rebuild, args=(False,), daemon=True).start()

    def queue_stale():
        n = 0
        for s in cards.check_stale(cdir, retriever.files) if cdir else []:
            if store.has_pending("stale", s["question"]):
                continue
            _, meta, body = next(c for c in cards.all_cards(cdir) if str(c[0]) == s["path"])
            store.add_feedback("stale", s["question"], meta.get("product"), cards.card_answer(body), list(meta.get("refs") or {}),
                               note="引用已变化：" + "、".join(s["changed"]), card=s["path"], alts=meta.get("alts") or [])
            n += 1
        return n

    def classify(d, refs, answer, no_material):
        """自动判定错因：A 检索没找到 / B 找到但读错 / C 资料里没有。

        套路：拿“正确出处”去比对“当时喂给模型的证据”——在里面就是模型读错（B），不在就是检索漏了（A）。
        用户只写了正确答案没给文件时，用答案文本反查最像的出处当候选，管理员审核时可改。
        """
        if no_material:
            return "C", refs, ""
        note = ""
        if not refs and answer.strip():
            with gpu:
                hits = retriever.search(answer[:500], k=3, project=d["project"] or None)
            refs, note = [f"{c.project}/{c.path}:{c.start}-{c.end}" for c in hits], "出处为系统根据答案反查，需核对"
        files = {r.split(":")[0] for r in refs}
        if not files:
            return "C", refs, note
        used = {e.split(":")[0] for e in d["evidence"]}
        return ("B" if files & used else "A"), refs, note

    class Feedback(BaseModel):
        qa_id: int
        answer: str = Field(default="", max_length=8000)
        files: str = Field(default="", max_length=2000)
        no_material: bool = False
        note: str = Field(default="", max_length=500)
        alts: str = Field(default="", max_length=2000)

    @app.post("/api/feedback")
    def feedback(f: Feedback, client: str = Depends(client_id)):
        d = store.get_qa(f.qa_id)
        if not d or not (f.answer.strip() or f.files.strip() or f.note.strip()):
            raise HTTPException(400, "请至少填写正确答案、正确出处或说明之一")
        refs = [x.strip() for x in f.files.replace("，", "\n").replace(",", "\n").splitlines() if x.strip()]
        cat, refs, auto_note = classify(d, refs, f.answer, f.no_material)
        fid = store.add_feedback("correction", d["question"], d["project"], f.answer.strip(), refs, cat,
                                 "；".join(x for x in (f.note.strip(), auto_note) if x), qa_id=d["id"], client=client,
                                 alts=[a.strip() for a in f.alts.splitlines() if a.strip()])
        return {"id": fid, "category": cat}

    @app.get("/api/admin/feedback", dependencies=[Depends(require_admin)])
    def feedback_list(status: str = "pending"):
        items = store.feedback(status)
        for it in items:
            if it["qa_id"]:
                qa = store.get_qa(it["qa_id"]) or {}
                it["old_answer"], it["evidence"] = qa.get("answer", ""), qa.get("evidence", [])
        return {"items": items, "stats": store.feedback_stats(), "draft_job": draft_job,
                "products": sorted({p for p in retriever.aliases if not p.endswith(("-客户资料", "-知识卡片"))})}

    class Review(BaseModel):
        action: str = Field(pattern="^(approve|reject|delete_card)$")
        question: str = ""
        answer: str = ""
        refs: list[str] = []
        alts: list[str] = []
        product: str = ""
        rebuild: bool = True

    @app.post("/api/admin/feedback/{fid}", dependencies=[Depends(require_admin)])
    def review(fid: int, r: Review):
        fb = store.get_feedback(fid)
        if not fb:
            raise HTTPException(404, "记录不存在")
        if r.action == "reject":
            store.review(fid, "rejected")
            return {"ok": True}
        if r.action == "delete_card":
            if fb["card"] and Path(fb["card"]).exists():
                Path(fb["card"]).unlink()
            store.review(fid, "rejected")
            if r.rebuild:
                start_rebuild()
            return {"ok": True}
        if not cdir:
            raise HTTPException(400, "sources.toml 里没有 split_by_subdir 的知识卡片源")
        if not (r.question.strip() and r.answer.strip() and r.product.strip()):
            raise HTTPException(400, "问题、答案、所属产品都不能为空")
        card_id = Path(fb["card"]).stem if fb["origin"] == "stale" and fb["card"] else None
        path = cards.write_card(cdir, r.product.strip(), r.question.strip(), r.answer, r.refs, retriever.files,
                                kind=fb["category"] or fb["origin"], card_id=card_id, alts=r.alts)
        store.review(fid, "approved", str(path))
        if r.rebuild:
            start_rebuild()
        return {"ok": True, "card": str(path)}

    @app.get("/api/admin/cards", dependencies=[Depends(require_admin)])
    def card_list():
        stale = {s["path"]: s["changed"] for s in cards.check_stale(cdir, retriever.files)} if cdir else {}
        return {"dir": str(cdir), "items": [{"path": str(p), "product": p.parent.name, "question": m.get("question"),
                                             "updated": m.get("updated"), "kind": m.get("kind"), "refs": list(m.get("refs") or {}),
                                             "stale": stale.get(str(p), [])} for p, m, _ in (cards.all_cards(cdir) if cdir else [])]}

    # ---------- 批量起草：管理员贴一批“客户常问”，模型用 agent 模式逐条起草，进审核队列 ----------
    draft_job = {"state": "idle", "done": 0, "total": 0, "log": []}

    # ---------- 代码概要：用线上已加载的模型逐文件生成（每个文件单独拿 GPU 锁），完成后自动重新入库 ----------
    sum_job = {"state": "idle", "log": [], "stop": False}

    class Summ(BaseModel):
        projects: list[str] = []
        limit: int | None = None

    @app.get("/api/admin/summarize", dependencies=[Depends(require_admin)])
    def summarize_status():
        from .summarize import load_cache, todo
        pending = {}
        for ref, _ in todo(retriever.files, load_cache(SUMMARIES)):
            p = ref.split("/", 1)[0]
            pending[p] = pending.get(p, 0) + 1
        return {"job": {k: v for k, v in sum_job.items() if k != "stop"}, "pending": pending}

    @app.post("/api/admin/summarize", dependencies=[Depends(require_admin)])
    def summarize_start(s: Summ):
        if sum_job["state"] == "running":
            sum_job["stop"] = True
            return {"stopping": True}
        sum_job.update(state="running", log=[], stop=False)
        threading.Thread(target=run_summarize, args=(set(s.projects) or None, s.limit), daemon=True).start()
        return {"started": True}

    def run_summarize(projects_, limit):
        from .summarize import summarize
        try:
            n = summarize(llm, dict(retriever.files), SUMMARIES, projects_, limit, lock=gpu,
                          log=sum_job["log"].append, should_stop=lambda: sum_job["stop"])
            sum_job["log"].append(f"完成 {n} 个文件" + ("（已手动停止）" if sum_job["stop"] else "") + "，开始重新入库")
            sum_job["state"] = "done"
            if n:
                start_rebuild()
        except Exception as e:
            sum_job["log"].append(f"失败：{type(e).__name__}: {e}")
            sum_job["state"] = "failed"

    class Draft(BaseModel):
        questions: str = Field(min_length=1, max_length=20000)
        project: str | None = None

    @app.post("/api/admin/draft", dependencies=[Depends(require_admin)])
    def draft(dr: Draft):
        if draft_job["state"] == "running":
            raise HTTPException(409, "已有起草任务在运行")
        qs = list(dict.fromkeys(q.strip(" \t-*0123456789.、") for q in dr.questions.splitlines() if q.strip()))[:100]
        draft_job.update(state="running", done=0, total=len(qs), log=[])
        threading.Thread(target=run_draft, args=(qs, dr.project or None), daemon=True).start()
        return draft_job

    def run_draft(qs, project):
        # 套路：每题单独拿锁，题与题之间放行线上提问，后台批量任务不会把前台卡死
        for q in qs:
            try:
                if store.has_pending("draft", q):
                    draft_job["log"].append(f"跳过（已在队列）：{q}")
                    continue
                with gpu:
                    res = agent.ask(q, project=project, mode="agent")
                refs = [f"{s['project']}/{s['path']}:{s['start']}-{s['end']}" for s in res["sources"]]
                # 起草时一律保留检索到的前 3 条证据作出处候选：模型偶尔漏标 [n]，不能因此丢掉出处
                refs = refs or res["evidence"][:3]
                no_ans = "没有找到" in res["answer"]
                note = "资料不足，需人工补答案" if no_ans else ("" if res["sources"] else "回答未标注出处，出处为检索候选，需核对")
                store.add_feedback("draft", q, project, cards.renumber(res["answer"], res["sources"]), refs, "C" if no_ans else "", note)
                draft_job["log"].append(f"{'资料不足' if no_ans else '已起草'}（{res['seconds']}s）：{q}")
            except Exception as e:
                draft_job["log"].append(f"失败 {type(e).__name__}: {e}：{q}")
            finally:
                draft_job["done"] += 1
        draft_job["state"] = "done"

    # ---------- 新增项目：探测目录 → 追加 sources.toml → 重建 ----------
    @app.get("/api/admin/project_suggest", dependencies=[Depends(require_admin)])
    def project_suggest(root: str):
        try:
            return projects.suggest(root)
        except ValueError as e:
            raise HTTPException(400, str(e))

    class NewProject(BaseModel):
        name: str
        root: str
        include: list[str]
        aliases: list[str] = []
        link_to: list[str] = []
        ref: str = ""
        rebuild: bool = True

    @app.post("/api/admin/projects", dependencies=[Depends(require_admin)])
    def add_project(p: NewProject):
        try:
            lines = projects.add_source(args.config, p.name.strip(), p.root.strip(), [i for i in p.include if i.strip()],
                                        p.aliases, p.ref.strip(), p.link_to)
        except ValueError as e:
            raise HTTPException(400, str(e))
        if p.rebuild:
            start_rebuild()
        return {"ok": True, "toml": "\n".join(lines)}

    queue_stale()  # 启动时也查一遍：服务停机期间代码可能已经变了
    print(f"\n管理员口令：{token}（文件 {store.dir / 'admin_token'}）\n后台地址：http://{args.host}:{args.port}/admin\n", flush=True)
    uvicorn.run(app, host=args.host, port=args.port)
