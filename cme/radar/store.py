"""Radar storage (SQLite). Original candidate data and the operator's decisions live in different tables."""
from __future__ import annotations
import json
import sqlite3
import time
from pathlib import Path

STATES = ("shortlist", "skip", "exclude")
SCHEMA = """
CREATE TABLE IF NOT EXISTS candidates(id TEXT PRIMARY KEY, origin TEXT, first_seen TEXT, last_seen TEXT, data TEXT);
CREATE TABLE IF NOT EXISTS candidate_keys(key TEXT PRIMARY KEY, candidate_id TEXT);
CREATE TABLE IF NOT EXISTS snapshots(n INTEGER PRIMARY KEY AUTOINCREMENT, candidate_id TEXT, seen_at TEXT, run_id TEXT, origin TEXT, data TEXT);
CREATE TABLE IF NOT EXISTS operator(candidate_id TEXT PRIMARY KEY, state TEXT, reason TEXT, updated_at TEXT, viewed INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS operator_log(n INTEGER PRIMARY KEY AUTOINCREMENT, candidate_id TEXT, state TEXT, reason TEXT, ts TEXT);
CREATE TABLE IF NOT EXISTS analyses(n INTEGER PRIMARY KEY AUTOINCREMENT, candidate_id TEXT, ts TEXT, status TEXT, model_requested TEXT,
  model_responded TEXT, material TEXT, result TEXT, error TEXT);
CREATE TABLE IF NOT EXISTS runs(n INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, kind TEXT, ts TEXT, params TEXT, status TEXT, detail TEXT);
"""


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.db.commit()

    # ---- ingest -------------------------------------------------------------------------
    def ingest(self, candidates, run_id: str, origin: str) -> dict:
        new = updated = 0
        for c in candidates:
            existing = None
            for k in c.get("keys") or []:
                row = self.db.execute("SELECT candidate_id FROM candidate_keys WHERE key=?", (k,)).fetchone()
                if row:
                    existing = row["candidate_id"]
                    break
            stamp = c.get("discovered_at") or now()
            cid = existing or c["id"]
            payload = json.dumps({**c, "id": cid}, ensure_ascii=False)
            if existing is None and self.db.execute("SELECT 1 FROM candidates WHERE id=?", (cid,)).fetchone():
                existing = cid
            if existing is None:
                self.db.execute("INSERT INTO candidates VALUES(?,?,?,?,?)", (cid, origin, stamp, stamp, payload))
                new += 1
            else:
                self.db.execute("UPDATE candidates SET last_seen=?, data=? WHERE id=?", (stamp, payload, cid))
                updated += 1
            for k in c.get("keys") or []:
                self.db.execute("INSERT OR IGNORE INTO candidate_keys VALUES(?,?)", (k, cid))
            self.db.execute("INSERT INTO snapshots(candidate_id, seen_at, run_id, origin, data) VALUES(?,?,?,?,?)",
                            (cid, stamp, run_id, origin, payload))
        self.db.commit()
        return {"new": new, "updated": updated}

    def log_run(self, run_id, kind, params, status, detail=""):
        self.db.execute("INSERT INTO runs(run_id, kind, ts, params, status, detail) VALUES(?,?,?,?,?,?)",
                        (run_id, kind, now(), json.dumps(params, ensure_ascii=False), status, str(detail)[:500]))
        self.db.commit()

    # ---- read ---------------------------------------------------------------------------
    def _card(self, row) -> dict:
        c = json.loads(row["data"])
        op = self.db.execute("SELECT state, reason, updated_at, viewed FROM operator WHERE candidate_id=?", (row["id"],)).fetchone()
        c["operator"] = {"state": op["state"] if op else None, "reason": op["reason"] if op else None,
                         "updated_at": op["updated_at"] if op else None, "viewed": bool(op["viewed"]) if op else False}
        c["first_seen"], c["last_seen"] = row["first_seen"], row["last_seen"]
        return c

    def get(self, cid: str):
        row = self.db.execute("SELECT * FROM candidates WHERE id=?", (cid,)).fetchone()
        return self._card(row) if row else None

    def list(self, state="feed", country="", platform="", origin="", period="", q="", limit=300) -> list:
        out = []
        for row in self.db.execute("SELECT * FROM candidates ORDER BY first_seen DESC, id"):
            c = self._card(row)
            st = c["operator"]["state"]
            if state == "feed" and st is not None:
                continue
            if state in STATES and st != state:
                continue
            if country and (c.get("country") or "").upper() != country.upper():
                continue
            if platform and (c.get("platform") or "") != platform:
                continue
            if origin and c.get("origin") != origin and row["origin"] != origin:
                continue
            if period and str(c.get("period_days") or "") != str(period):
                continue
            if q:
                hay = " ".join(str(c.get(k) or "") for k in ("brand", "brand_name", "title", "industry")).lower()
                if q.lower() not in hay:
                    continue
            out.append(c)
            if len(out) >= limit:
                break
        return out

    def counts(self) -> dict:
        one = lambda sql: self.db.execute(sql).fetchone()[0]  # noqa: E731
        return {"found": one("SELECT COUNT(*) FROM candidates"),
                "viewed": one("SELECT COUNT(*) FROM operator WHERE viewed=1"),
                "saved": one("SELECT COUNT(*) FROM operator WHERE state='shortlist'"),
                "skipped": one("SELECT COUNT(*) FROM operator WHERE state='skip'"),
                "excluded": one("SELECT COUNT(*) FROM operator WHERE state='exclude'")}

    def history(self, cid: str) -> list:
        rows = self.db.execute("SELECT seen_at, run_id, origin, data FROM snapshots WHERE candidate_id=? ORDER BY n", (cid,))
        return [{"seen_at": r["seen_at"], "run_id": r["run_id"], "origin": r["origin"],
                 "metrics": json.loads(r["data"]).get("metrics", {})} for r in rows]

    # ---- operator decisions (never touch candidate data) -----------------------------------
    def mark_viewed(self, cid: str) -> None:
        self._need(cid)
        self.db.execute("INSERT INTO operator(candidate_id, state, reason, updated_at, viewed) VALUES(?,?,?,?,1) "
                        "ON CONFLICT(candidate_id) DO UPDATE SET viewed=1", (cid, None, None, now()))
        self.db.commit()

    def decide(self, cid: str, action: str, reason: str = "") -> None:
        self._need(cid)
        reason = (reason or "").strip()
        if action == "restore":
            state = None
        elif action in STATES:
            state = action
        else:
            raise ValueError(f"unknown action {action!r}")
        if state in ("shortlist", "exclude") and not reason:
            raise ValueError(f"a reason is required for '{state}'")
        self.db.execute("INSERT INTO operator(candidate_id, state, reason, updated_at, viewed) VALUES(?,?,?,?,1) "
                        "ON CONFLICT(candidate_id) DO UPDATE SET state=excluded.state, reason=excluded.reason, "
                        "updated_at=excluded.updated_at", (cid, state, reason or None, now()))
        self.db.execute("INSERT INTO operator_log(candidate_id, state, reason, ts) VALUES(?,?,?,?)", (cid, state, reason or None, now()))
        self.db.commit()

    def decision_log(self, cid: str) -> list:
        return [dict(r) for r in self.db.execute("SELECT state, reason, ts FROM operator_log WHERE candidate_id=? ORDER BY n", (cid,))]

    # ---- analyses -----------------------------------------------------------------------
    def save_analysis(self, cid, status, model_requested="", model_responded="", material=None, result=None, error="") -> int:
        cur = self.db.execute(
            "INSERT INTO analyses(candidate_id, ts, status, model_requested, model_responded, material, result, error) VALUES(?,?,?,?,?,?,?,?)",
            (cid, now(), status, model_requested, model_responded, json.dumps(material, ensure_ascii=False) if material is not None else None,
             json.dumps(result, ensure_ascii=False) if result is not None else None, str(error)[:1000]))
        self.db.commit()
        return cur.lastrowid

    def analyses(self, cid: str) -> list:
        out = []
        for r in self.db.execute("SELECT * FROM analyses WHERE candidate_id=? ORDER BY n DESC", (cid,)):
            out.append({"n": r["n"], "ts": r["ts"], "status": r["status"], "model_requested": r["model_requested"],
                        "model_responded": r["model_responded"], "error": r["error"],
                        "material": json.loads(r["material"]) if r["material"] else None,
                        "result": json.loads(r["result"]) if r["result"] else None})
        return out

    def _need(self, cid: str) -> None:
        if not self.db.execute("SELECT 1 FROM candidates WHERE id=?", (cid,)).fetchone():
            raise KeyError(f"unknown candidate {cid!r}")

    def close(self) -> None:
        self.db.close()
