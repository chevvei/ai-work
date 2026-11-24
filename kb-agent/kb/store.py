"""站点数据层：后台开关（site.json）+ 问答记录/投票/榜单（SQLite）。

套路：
- 配置和数据分开放：site.json 人能直接看懂、能手改、能备份；问答流水进 SQLite，量大也不怕。
- 存“未脱敏”的完整来源，对外接口再按开关过滤：开关随时可改，历史记录不用重算。
- 热门按“去重提问人数”而不是“提问次数”排：一个人刷 20 次不算热门。
"""
import json
import re
import secrets
import sqlite3
import threading
import time
from pathlib import Path

DEFAULT_SETTINGS = {
    "title": "研发知识库助手",
    "notice": "回答基于公司内部资料自动生成，[n] 为出处编号；关键操作请以原文为准。",
    "examples": ["LidarAssistant 在 Linux 下怎么编译？", "udslib 报 CalcKeyDll file not exist 怎么办？",
                 "rs_driver 如何解析 DIFOP 包？", "monitor 点云缺失检测怎么做？"],
    "allow_agent_mode": True,
    "board_auto_users": 3,   # 至少这么多不同的人问过
    "board_auto_likes": 2,   # 且累计点赞达到这个数，自动上榜；后台也可手动上榜/下榜
    "hot_days": 30,
    "projects": {},
}
# 新出现的项目默认值：客户资料类不单独出现在下拉框里（选产品时会自动带上）
PROJECT_DEFAULT = {"enabled": True, "listed": True, "show_source": True, "label": ""}
PUNCT = re.compile(r"[\s\?？。，,.!！:：;；、\"'“”‘’()（）\[\]【】]+")


def norm_question(q):
    return PUNCT.sub("", q.lower())[:200]


class Store:
    def __init__(self, data_dir):
        self.dir = Path(data_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.settings_path = self.dir / "site.json"
        self.lock = threading.Lock()
        self.db = sqlite3.connect(self.dir / "kb.db", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS qa(id INTEGER PRIMARY KEY, ts REAL, client TEXT, ip TEXT, question TEXT, norm TEXT,
            project TEXT, mode TEXT, answer TEXT, sources TEXT, evidence TEXT, seconds REAL, cited INTEGER,
            likes INTEGER DEFAULT 0, dislikes INTEGER DEFAULT 0, favs INTEGER DEFAULT 0);
        CREATE INDEX IF NOT EXISTS qa_norm ON qa(norm);
        CREATE TABLE IF NOT EXISTS votes(qa_id INTEGER, client TEXT, kind TEXT, ts REAL, PRIMARY KEY(qa_id, client, kind));
        CREATE TABLE IF NOT EXISTS featured(norm TEXT PRIMARY KEY, question TEXT, qa_id INTEGER, ts REAL, hidden INTEGER DEFAULT 0);
        CREATE TABLE IF NOT EXISTS feedback(id INTEGER PRIMARY KEY, ts REAL, origin TEXT, qa_id INTEGER, client TEXT,
            question TEXT, project TEXT, answer TEXT, refs TEXT, category TEXT, note TEXT,
            status TEXT DEFAULT 'pending', card TEXT, reviewed_ts REAL, alts TEXT DEFAULT '[]');
        """)
        if "alts" not in {r["name"] for r in self.db.execute("PRAGMA table_info(feedback)")}:  # 旧库补列
            self.db.execute("ALTER TABLE feedback ADD COLUMN alts TEXT DEFAULT '[]'")
            self.db.commit()
        if "followups" not in {r["name"] for r in self.db.execute("PRAGMA table_info(qa)")}:
            self.db.execute("ALTER TABLE qa ADD COLUMN followups TEXT DEFAULT '[]'")
            self.db.commit()
        self.settings = self._load_settings()

    # ---------- 后台开关 ----------
    def _load_settings(self):
        s = json.loads(json.dumps(DEFAULT_SETTINGS))
        if self.settings_path.exists():
            s.update(json.loads(self.settings_path.read_text(encoding="utf-8")))
        return s

    def save_settings(self, new):
        with self.lock:
            merged = {**self.settings, **{k: v for k, v in new.items() if k in DEFAULT_SETTINGS}}
            tmp = self.settings_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(merged, ensure_ascii=False, indent=1), encoding="utf-8")
            tmp.replace(self.settings_path)  # 先写临时文件再改名：写一半断电也不会把配置写坏
            self.settings = merged
        return merged

    def sync_projects(self, names):
        """索引里新出现的项目补默认开关；已删除的项目保留配置（重新入库时不丢设置）。"""
        projects = dict(self.settings.get("projects", {}))
        changed = False
        for n in names:
            if n not in projects:
                projects[n] = {**PROJECT_DEFAULT, "listed": not n.endswith(("-客户资料", "-知识卡片"))}
                changed = True
        if changed or not self.settings_path.exists():
            self.save_settings({"projects": projects})

    def project(self, name):
        return {**PROJECT_DEFAULT, **self.settings["projects"].get(name, {})}

    def admin_token(self):
        f = self.dir / "admin_token"
        if not f.exists():
            f.write_text(secrets.token_urlsafe(16), encoding="utf-8")
            f.chmod(0o600)
        return f.read_text(encoding="utf-8").strip()

    # ---------- 问答流水 ----------
    def add_qa(self, client, ip, question, project, mode, res):
        with self.lock:
            cur = self.db.execute(
                "INSERT INTO qa(ts,client,ip,question,norm,project,mode,answer,sources,evidence,seconds,cited) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (time.time(), client, ip, question, norm_question(question), project, mode, res["answer"],
                 json.dumps(res["sources"], ensure_ascii=False), json.dumps(res["evidence"], ensure_ascii=False),
                 res["seconds"], len(res["sources"])))
            self.db.commit()
            return cur.lastrowid

    def get_qa(self, qa_id):
        row = self.db.execute("SELECT * FROM qa WHERE id=?", (qa_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["sources"], d["evidence"] = json.loads(d["sources"]), json.loads(d["evidence"])
        d["followups"] = json.loads(d.get("followups") or "[]")
        return d

    def set_followups(self, qa_id, items):
        with self.lock:
            self.db.execute("UPDATE qa SET followups=? WHERE id=?", (json.dumps(items, ensure_ascii=False), qa_id))
            self.db.commit()

    def my_votes(self, qa_id, client):
        return [r["kind"] for r in self.db.execute("SELECT kind FROM votes WHERE qa_id=? AND client=?", (qa_id, client))]

    def vote(self, qa_id, client, kind, on):
        """赞/踩互斥；收藏独立。计数冗余存在 qa 表里，排行榜查询不用每次 JOIN。"""
        with self.lock:
            kinds = [kind] + (["dislike" if kind == "like" else "like"] if on and kind in ("like", "dislike") else [])
            for k in kinds:
                want = on if k == kind else False
                exists = self.db.execute("SELECT 1 FROM votes WHERE qa_id=? AND client=? AND kind=?", (qa_id, client, k)).fetchone()
                if want and not exists:
                    self.db.execute("INSERT INTO votes VALUES(?,?,?,?)", (qa_id, client, k, time.time()))
                elif not want and exists:
                    self.db.execute("DELETE FROM votes WHERE qa_id=? AND client=? AND kind=?", (qa_id, client, k))
                else:
                    continue
                col = {"like": "likes", "dislike": "dislikes", "fav": "favs"}[k]
                self.db.execute(f"UPDATE qa SET {col}={col}+? WHERE id=?", (1 if want else -1, qa_id))
            self.db.commit()
        row = self.db.execute("SELECT likes,dislikes,favs FROM qa WHERE id=?", (qa_id,)).fetchone()
        return {**dict(row), "mine": self.my_votes(qa_id, client)}

    def history(self, client, limit=100):
        rows = self.db.execute("SELECT id, ts, question, project FROM qa WHERE client=? ORDER BY id DESC LIMIT ?", (client, limit))
        return [dict(r) for r in rows]

    def favorites(self, client):
        rows = self.db.execute("""SELECT q.id, q.question, q.project, v.ts FROM votes v JOIN qa q ON q.id=v.qa_id
                                  WHERE v.client=? AND v.kind='fav' ORDER BY v.ts DESC LIMIT 200""", (client,))
        return [dict(r) for r in rows]

    # ---------- 热门 / 榜单 ----------
    def hot(self, days=None, limit=20):
        days = days or self.settings["hot_days"]
        rows = self.db.execute("""
            SELECT norm, COUNT(*) asks, COUNT(DISTINCT client) users, SUM(likes) likes, SUM(dislikes) dislikes,
                   SUM(favs) favs, MAX(ts) last_ts
            FROM qa WHERE ts>=? GROUP BY norm
            ORDER BY (COUNT(DISTINCT client) + 2*SUM(likes) + SUM(favs) - 2*SUM(dislikes)) DESC, asks DESC LIMIT ?""",
                               (time.time() - days * 86400, limit)).fetchall()
        return [{**dict(r), **self._best(r["norm"])} for r in rows]

    def _best(self, norm):
        """同一问题被问多次，展示“赞最多、其次最新”的那一次回答。"""
        r = self.db.execute("SELECT id, question FROM qa WHERE norm=? ORDER BY likes-dislikes DESC, id DESC LIMIT 1", (norm,)).fetchone()
        return {"qa_id": r["id"], "question": r["question"]} if r else {}

    def board(self):
        """榜单 = 后台手动上榜 ∪ 达到“人数 + 点赞”门槛的自动上榜，减去后台手动下榜的。"""
        s = self.settings
        flags = {r["norm"]: dict(r) for r in self.db.execute("SELECT * FROM featured")}
        out, seen = [], set()
        for f in sorted(flags.values(), key=lambda x: -x["ts"]):
            if not f["hidden"]:
                out.append({"norm": f["norm"], "question": f["question"], "qa_id": f["qa_id"], "pinned": True})
                seen.add(f["norm"])
        for h in self.hot(limit=100):
            if h["norm"] in seen or flags.get(h["norm"], {}).get("hidden"):
                continue
            if h["users"] >= s["board_auto_users"] and (h["likes"] or 0) >= s["board_auto_likes"]:
                out.append({"norm": h["norm"], "question": h["question"], "qa_id": h["qa_id"], "pinned": False,
                            "users": h["users"], "likes": h["likes"]})
        for o in out:  # 置顶项也用最新的最佳回答
            o.update({k: v for k, v in self._best(o["norm"]).items() if k == "qa_id"})
        return out

    def feature(self, norm, question, qa_id, on):
        with self.lock:
            self.db.execute("INSERT INTO featured(norm,question,qa_id,ts,hidden) VALUES(?,?,?,?,?) "
                            "ON CONFLICT(norm) DO UPDATE SET question=excluded.question, qa_id=excluded.qa_id, ts=excluded.ts, hidden=excluded.hidden",
                            (norm, question, qa_id, time.time(), 0 if on else 1))
            self.db.commit()

    def unflag(self, norm):
        with self.lock:
            self.db.execute("DELETE FROM featured WHERE norm=?", (norm,))
            self.db.commit()

    # ---------- 后台统计 ----------
    def stats(self):
        now = time.time()
        lt = time.localtime(now)
        today0 = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 0, 0, 0, 0, 0, -1))
        one = lambda sql, *a: self.db.execute(sql, a).fetchone()[0]
        daily = self.db.execute("""SELECT date(ts,'unixepoch','localtime') d, COUNT(*) n, COUNT(DISTINCT client) u
                                   FROM qa WHERE ts>=? GROUP BY d ORDER BY d""", (now - 14 * 86400,)).fetchall()
        return {"total": one("SELECT COUNT(*) FROM qa"), "users": one("SELECT COUNT(DISTINCT client) FROM qa"),
                "today": one("SELECT COUNT(*) FROM qa WHERE ts>=?", today0),
                "likes": one("SELECT COALESCE(SUM(likes),0) FROM qa"), "dislikes": one("SELECT COALESCE(SUM(dislikes),0) FROM qa"),
                "no_cite": one("SELECT COUNT(*) FROM qa WHERE cited=0"), "avg_seconds": one("SELECT ROUND(AVG(seconds),2) FROM qa"),
                "daily": [dict(r) for r in daily]}

    def misses(self, limit=50):
        """待补资料：没引用到任何来源，或被点踩的问题——这就是下一批 FAQ 的选题来源。"""
        rows = self.db.execute("""SELECT id, ts, question, project, cited, likes, dislikes FROM qa
                                  WHERE cited=0 OR dislikes>0 ORDER BY id DESC LIMIT ?""", (limit,))
        return [dict(r) for r in rows]

    def recent(self, limit=50, keyword=""):
        rows = self.db.execute("""SELECT id, ts, client, ip, question, project, mode, seconds, cited, likes, dislikes, favs
                                  FROM qa WHERE question LIKE ? ORDER BY id DESC LIMIT ?""", (f"%{keyword}%", limit))
        return [dict(r) for r in rows]

    # ---------- 纠错 / 起草 → 审核队列 ----------
    # origin: correction（用户纠错）| draft（后台批量起草）| stale（卡片引用的代码变了）
    #         miss（模型自己答“没找到”）| dislike（用户点踩）—— 后两种是系统自动收集的“不会的题”
    # category: A 检索没找到 | B 找到了但读错 | C 资料里没有 —— 决定该补什么（别名/切块、提示词、还是补资料）
    def add_feedback(self, origin, question, project, answer, refs, category="", note="", qa_id=None, client="", card="", alts=()):
        with self.lock:
            cur = self.db.execute("INSERT INTO feedback(ts,origin,qa_id,client,question,project,answer,refs,category,note,card,alts) "
                                  "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", (time.time(), origin, qa_id, client, question, project, answer,
                                                                      json.dumps(refs, ensure_ascii=False), category, note, card,
                                                                      json.dumps(list(alts), ensure_ascii=False)))
            self.db.commit()
            return cur.lastrowid

    @staticmethod
    def _fb(r):
        return {**dict(r), "refs": json.loads(r["refs"] or "[]"), "alts": json.loads(r["alts"] or "[]")}

    def feedback(self, status="pending", limit=200):
        rows = self.db.execute("SELECT * FROM feedback WHERE status=? OR ?='' ORDER BY id DESC LIMIT ?", (status, status, limit))
        return [self._fb(r) for r in rows]

    def get_feedback(self, fid):
        r = self.db.execute("SELECT * FROM feedback WHERE id=?", (fid,)).fetchone()
        return self._fb(r) if r else None

    def review(self, fid, status, card=None):
        with self.lock:
            self.db.execute("UPDATE feedback SET status=?, card=COALESCE(?, card), reviewed_ts=? WHERE id=?", (status, card, time.time(), fid))
            self.db.commit()

    def has_pending(self, origin, question):
        """origin=None：任意来源已有待审的同一问题就算（自动收集时避免同一题被问十次进十条）。"""
        return self.db.execute("SELECT 1 FROM feedback WHERE (origin=? OR ? IS NULL) AND question=? AND status='pending'",
                               (origin, origin, question)).fetchone() is not None

    def feedback_stats(self):
        rows = self.db.execute("SELECT status, category, COUNT(*) n FROM feedback GROUP BY status, category")
        return [dict(r) for r in rows]
