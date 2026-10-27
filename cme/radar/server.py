"""Creative Radar: local manual review UI. Standard library only; binds to 127.0.0.1.

Nothing runs by itself: FIND, analysis and every decision are explicit operator actions.
"""
from __future__ import annotations
import hashlib
import json
import os
import re
import secrets
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from ..offer import OfferError, load_offer
from ..providers import CallLog, ProviderPolicyError
from ..qa import validate_product_references
from ..registry import load_catalog, load_settings, make_analyst
from . import analyze as analyze_mod
from . import find as find_mod
from .store import Store

ROOT = Path(__file__).resolve().parent.parent.parent


class NotConfigured(Exception):
    def __init__(self, detail: str, how_to_fix: str):
        self.detail, self.how_to_fix = detail, how_to_fix
        super().__init__(detail)


class RadarApp:
    def __init__(self, data_dir, settings_path=None, catalog_path=None, ctx=None, find_client=None, getter=None,
                 runner=None, tools=None, offer_path=None, vertical_path=None, production_shots=1):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.store = Store(self.data_dir / "radar.db")
        self.settings_path = self._root_path(settings_path or ROOT / "settings.json")
        self.catalog_path = self._root_path(catalog_path or ROOT / "adapters.json")
        self.offer_path = self._root_path(offer_path or ROOT / "examples" / "offer_example.json")
        self.vertical_path = self._root_path(vertical_path or ROOT / "verticals" / "physical_goods_us.json")
        if type(production_shots) is not int or not 1 <= production_shots <= 6:
            raise ValueError("production_shots must be an integer from 1 to 6")
        self.production_shots = production_shots
        self.ctx, self.find_client, self.getter, self.runner, self.tools = ctx or {}, find_client, getter, runner, tools
        self.token = secrets.token_urlsafe(16)

    @staticmethod
    def _root_path(value):
        path = Path(value)
        return path if path.is_absolute() else ROOT / path

    def production_status(self) -> dict:
        try:
            offer = load_offer(self.offer_path)
            issues = validate_product_references(offer.photos)
            return {"state": "READY" if not issues else "BLOCKED", "offer_id": offer.offer_id,
                    "name": offer.name, "photo_issues": issues, "shots": self.production_shots}
        except (OSError, ValueError, OfferError) as e:
            return {"state": "NOT_CONFIGURED", "name": "", "photo_issues": [str(e)], "shots": self.production_shots}

    # ---- analyst (built only when the operator presses Analyze) ---------------------------
    def _analyst(self):
        if not self.settings_path.exists():
            raise NotConfigured(f"{self.settings_path.name} not found", "copy settings.example.json to settings.json")
        settings = load_settings(self.settings_path)
        if not settings.analyst_adapter:
            raise NotConfigured("analyst_adapter is not set in settings", "set analyst_adapter (see settings.example.json)")
        if not (self.ctx.get("openrouter_key") or os.getenv("OPENROUTER_API_KEY")):
            raise NotConfigured("OPENROUTER_API_KEY is not set", "export OPENROUTER_API_KEY=... and restart Creative Radar")
        try:
            call = make_analyst(settings, load_catalog(self.catalog_path), CallLog(self.data_dir / "calls.jsonl"),
                                time.strftime("radar-%Y%m%d-%H%M%S"), **self.ctx)
        except ProviderPolicyError as e:
            raise NotConfigured(str(e), "adjust allow_free / allow_unverified in settings (see README) or choose another adapter")
        except KeyError as e:
            raise NotConfigured(f"adapter {e} is not in the catalog", "fix analyst_adapter in settings")
        return call

    def analyst_status(self) -> dict:
        try:
            self._analyst()
            return {"state": "READY", "detail": "", "how_to_fix": ""}
        except NotConfigured as e:
            return {"state": "NOT_CONFIGURED", "detail": e.detail, "how_to_fix": e.how_to_fix}

    # ---- operations -----------------------------------------------------------------------
    def state(self) -> dict:
        return {"vertical": find_mod.VERTICAL, "counts": self.store.counts(), "find": find_mod.find_status(),
                "analyst": self.analyst_status(), "tools": self.tools or analyze_mod.tools_status(),
                "production": self.production_status(), "metric_fields": list(find_mod.METRIC_FIELDS)}

    def feed(self, q: dict) -> list:
        g = lambda k, d="": (q.get(k) or [d])[0]  # noqa: E731
        return self.store.list(state=g("state", "feed"), country=g("country"), platform=g("platform"),
                               origin=g("origin"), period=g("period"), q=g("q"))

    def candidate(self, cid: str) -> dict:
        c = self.store.get(cid)
        if c is None:
            raise KeyError(cid)
        c["analyses"], c["history"], c["decision_log"] = self.store.analyses(cid), self.store.history(cid), self.store.decision_log(cid)
        return c

    def do_find(self, params: dict) -> dict:
        run_id = time.strftime("find-%Y%m%d-%H%M%S")
        search_params = dict(params or {})
        # The local Control Center automatically uses the configured OfferCard as
        # FIND context unless the caller supplies an explicit offer.
        if not isinstance(search_params.get("offer"), dict) and not any(
            search_params.get(key) for key in ("target_offer", "offer_name", "product_type", "product_category")
        ):
            try:
                offer_data = json.loads(self.offer_path.read_text(encoding="utf-8"))
                if isinstance(offer_data, dict):
                    search_params["offer"] = {
                        key: offer_data.get(key)
                        for key in ("name", "product_type", "product_category")
                        if isinstance(offer_data.get(key), str) and offer_data.get(key).strip()
                    }
            except (OSError, ValueError):
                # FIND can still run from explicit keywords if the optional OfferCard is unavailable.
                pass
        try:
            res = find_mod.run_find(search_params, client=self.find_client)
        except find_mod.FindUnavailable as e:
            self.store.log_run(run_id, "find", search_params, e.state, e.detail)
            return {"status": e.state, "detail": e.detail, "how_to_fix": find_mod.find_status()["how_to_fix"]}
        except ValueError as e:
            return {"status": "BAD_REQUEST", "detail": str(e)}
        counts = self.store.ingest(res["candidates"], run_id, "found")
        strategy = (res.get("stats") or {}).get("discovery_strategy") or {}
        diagnostics = res.get("diagnostics") or {}
        self.store.log_run(run_id, "find", res["request"], "OK",
                           json.dumps({"counts": counts, "diagnostics": diagnostics,
                                       "supply_search_calls_used": strategy.get("supply_search_calls_used", 0),
                                       "performance_calls_used": strategy.get("performance_calls_used", 0)}))
        return {"status": "OK", **counts, "returned": len(res["candidates"]),
                "diagnostics": diagnostics, "discovery_strategy": strategy}

    def do_import(self, body: dict) -> dict:
        run_id = time.strftime("import-%Y%m%d-%H%M%S")
        path = str(body.get("path") or "").strip()
        if not path:
            return {"status": "BAD_REQUEST", "detail": "path to an old CME discovery JSON is required"}
        try:
            res = find_mod.import_discovery_file(path)
        except (OSError, ValueError) as e:
            self.store.log_run(run_id, "import", {"path": path}, "ERROR", str(e))
            return {"status": "BAD_REQUEST", "detail": str(e)}
        counts = self.store.ingest(res["candidates"], run_id, "imported")
        self.store.log_run(run_id, "import", {"path": path}, "OK", json.dumps(counts))
        return {"status": "OK", **counts, "note": "imported candidates are labelled 'imported', not automatically found"}

    def do_analyze(self, cid: str) -> dict:
        self.store._need(cid)
        try:
            call = self._analyst()
        except NotConfigured as e:
            return {"status": "NOT_CONFIGURED", "detail": e.detail, "how_to_fix": e.how_to_fix}
        return analyze_mod.analyze_candidate(self.store, cid, call, self.data_dir, analyst_info=lambda: call.info,
                                             getter=self.getter, runner=self.runner, tools=self.tools)

    def productions(self, cid: str) -> list:
        """Read production reports for one candidate; do not expose other local files."""
        self.store._need(cid)
        root = self.data_dir / "production"
        reports = []
        if not root.is_dir():
            return reports
        for path in root.glob("*/production.json"):
            try:
                item = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if item.get("candidate_id") == cid:
                reports.append(item)
        return sorted(reports, key=lambda x: x.get("started_at", ""), reverse=True)

    def do_produce(self, cid: str) -> dict:
        """Explicit handoff: shortlisted ad + accepted sampled-frame analysis -> existing atom/image CLI."""
        c = self.candidate(cid)
        if c["operator"].get("state") != "shortlist":
            return {"status": "NOT_SHORTLISTED", "detail": "first move this source ad to Shortlist"}
        analysis = next((x for x in c["analyses"] if x.get("status") == "OK" and isinstance(x.get("result"), dict)), None)
        if analysis is None:
            return {"status": "NO_VALID_ANALYSIS", "detail": "analyze the selected source video first; no valid analysis is saved"}
        material = analysis.get("material") or {}
        if not material.get("video_sha256") or not material.get("frames"):
            return {"status": "SOURCE_EVIDENCE_INCOMPLETE", "detail": "analysis has no saved video hash / sampled-frame evidence"}
        try:
            offer = load_offer(self.offer_path)
        except (OSError, ValueError, OfferError) as e:
            return {"status": "TARGET_OFFER_INVALID", "detail": str(e)}
        ref_issues = validate_product_references(offer.photos)
        if ref_issues:
            return {"status": "TARGET_PHOTO_BLOCKED", "detail": "; ".join(ref_issues), "provider_calls_made": 0}
        if not self.settings_path.is_file():
            return {"status": "NOT_CONFIGURED", "detail": f"{self.settings_path.name} not found; copy settings.example.json to settings.json"}
        # Only the validated analysis result crosses the boundary. Source URLs, local video/frame paths,
        # source brand identity and raw source prompts remain in Radar storage, not in the Brain payload.
        handoff = {"status": "OK", "candidate_id": cid, "analysis_id": analysis.get("n"),
                   "result": analysis["result"],
                   "source_scope": "sampled frames only; mechanism and transferability remain hypotheses"}
        run_id = "prod-" + time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(4)
        out = self.data_dir / "production" / run_id
        out.mkdir(parents=True, exist_ok=False)
        source_path = out / "source_contract.json"
        source_bytes = json.dumps(handoff, ensure_ascii=False, indent=2).encode("utf-8")
        source_path.write_bytes(source_bytes)
        brief = ("Create target-native product-ad frames for the configured OfferCard. Preserve the source's causal "
                 "commercial mechanism, but replace its product, creator, brand, props, scene, wording and composition. "
                 "Show the configured target product clearly; use only verified OfferCard facts and visible evidence.")
        cmd = [sys.executable, "-m", "cme.smoke", "all", "--offer", str(self.offer_path),
               "--settings", str(self.settings_path), "--catalog", str(self.catalog_path),
               "--vertical", str(self.vertical_path), "--out", str(out), "--source-contract", str(source_path),
               "--brief", brief, "--shots", str(self.production_shots), "--run-id", run_id]
        started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        report = {"run_id": run_id, "candidate_id": cid, "analysis_id": analysis.get("n"), "started_at": started,
                  "status": "RUNNING", "exit_code": None, "offer_id": offer.offer_id, "offer_name": offer.name,
                  "source_contract_sha256": hashlib.sha256(source_bytes).hexdigest(), "images": [], "verdicts": [],
                  "stdout_tail": "", "stderr_tail": "", "provider_calls_made": 0}
        report_path = out / "production.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            res = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=900, check=False)
            report["exit_code"] = int(res.returncode)
            report["stdout_tail"] = (res.stdout or "")[-5000:]
            report["stderr_tail"] = (res.stderr or "")[-2000:]
            report["status"] = {0: "OK", 2: "INFRA_ERROR", 3: "BLOCKED", 4: "ATOM_CONTRACT_FAILED",
                                5: "PROVIDER_ERROR", 6: "UNVERIFIED", 7: "FAILED_QA"}.get(res.returncode, "FAILED")
        except subprocess.TimeoutExpired as e:
            report["status"] = "TIMEOUT"
            report["stderr_tail"] = str(e)[:2000]
        except OSError as e:
            report["status"] = "LOCAL_EXECUTION_ERROR"
            report["stderr_tail"] = f"{type(e).__name__}: {e}"
        finally:
            report["images"] = [p.name for p in sorted((out / "shots").glob("shot_*.png"))] if (out / "shots").is_dir() else []
            if report["exit_code"] == 0 and not report["images"]:
                report["status"] = "NO_IMAGE_OUTPUT"
            report["verdicts"] = []
            receipt_root = out / "receipts"
            for receipt_path in sorted(receipt_root.glob("*.json")) if receipt_root.is_dir() else []:
                try:
                    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                if receipt.get("image_path") or receipt.get("receipt_id"):
                    report["verdicts"].append({"receipt_id": receipt.get("receipt_id", receipt_path.stem),
                                               "shot": receipt.get("shot"), "overall": receipt.get("overall"),
                                               "commercial_status": receipt.get("commercial_status")})
            calls_path = out / "calls.jsonl"
            if calls_path.is_file():
                report["provider_calls_made"] = sum(1 for line in calls_path.read_text(encoding="utf-8").splitlines() if line.strip())
            run_result_path = out / "run_result.json"
            if run_result_path.is_file():
                try:
                    run_result = json.loads(run_result_path.read_text(encoding="utf-8"))
                    report["failure_class"] = str(run_result.get("failure_class") or "")
                    report["failure_detail"] = str(run_result.get("message") or "")[:1000]
                except (OSError, ValueError):
                    report["failure_class"] = "RUN_RESULT_UNREADABLE"
            report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report


class Handler(BaseHTTPRequestHandler):
    app: RadarApp = None  # set by make_server
    port: int = 0

    def log_message(self, *a):
        pass

    # --- helpers
    def _host_ok(self) -> bool:
        return (self.headers.get("Host") or "") in {f"127.0.0.1:{self.port}", f"localhost:{self.port}"}

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        raw = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(raw)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj, ensure_ascii=False, default=str))

    def _post_ok(self) -> bool:
        origin = self.headers.get("Origin")
        return (self._host_ok() and (origin is None or origin == f"http://{self.headers.get('Host')}")
                and self.headers.get("X-Radar-Token") == self.app.token)

    # --- routes
    def do_GET(self):
        if not self._host_ok():
            return self._json(403, {"error": "bad Host header"})
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path == "/":
                return self._send(200, PAGE.replace("__TOKEN__", self.app.token), "text/html; charset=utf-8")
            if u.path == "/api/state":
                return self._json(200, self.app.state())
            if u.path == "/api/feed":
                return self._json(200, {"items": [slim(c) for c in self.app.feed(q)]})
            if u.path == "/api/candidate":
                return self._json(200, self.app.candidate((q.get("id") or [""])[0]))
            if u.path == "/api/productions":
                return self._json(200, {"items": self.app.productions((q.get("id") or [""])[0])})
            if u.path == "/api/production-file":
                run_id, name = (q.get("run") or [""])[0], (q.get("name") or [""])[0]
                if not re.fullmatch(r"prod-[A-Za-z0-9_-]{1,80}", run_id) or not re.fullmatch(r"shot_\d{2}_a\d{2}\.png", name):
                    return self._json(400, {"error": "invalid production image reference"})
                root = (self.app.data_dir / "production" / run_id / "shots").resolve()
                image_path = (root / name).resolve()
                if not image_path.is_relative_to(root) or not image_path.is_file():
                    return self._json(404, {"error": "production image not found"})
                return self._send(200, image_path.read_bytes(), "image/png")
        except KeyError as e:
            return self._json(404, {"error": f"not found: {e}"})
        except Exception as e:  # never drop the connection: report it
            return self._json(500, {"error": f"{type(e).__name__}: {e}"})
        self._json(404, {"error": "not found"})

    def do_POST(self):
        if not self._post_ok():
            return self._json(403, {"error": "forbidden: bad Host/Origin or missing token"})
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
        except (ValueError, TypeError):
            return self._json(400, {"error": "invalid JSON body"})
        p = urlparse(self.path).path
        try:
            if p == "/api/view":
                self.app.store.mark_viewed(body.get("id", ""))
                return self._json(200, {"ok": True})
            if p == "/api/decide":
                self.app.store.decide(body.get("id", ""), body.get("action", ""), body.get("reason", ""))
                return self._json(200, {"ok": True})
            if p == "/api/analyze":
                return self._json(200, self.app.do_analyze(body.get("id", "")))
            if p == "/api/produce":
                return self._json(200, self.app.do_produce(body.get("id", "")))
            if p == "/api/find":
                return self._json(200, self.app.do_find(body))
            if p == "/api/import":
                return self._json(200, self.app.do_import(body))
        except KeyError as e:
            return self._json(404, {"error": f"not found: {e}"})
        except ValueError as e:
            return self._json(400, {"error": str(e)})
        except Exception as e:  # never drop the connection: report it
            return self._json(500, {"error": f"{type(e).__name__}: {e}"})
        self._json(404, {"error": "not found"})


def slim(c: dict) -> dict:
    """The list view: no raw legacy record (it can be large)."""
    return {k: v for k, v in c.items() if k != "legacy_record"}


def make_server(app: RadarApp, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    if host not in ("127.0.0.1", "localhost"):
        raise ValueError("Creative Radar binds to localhost only")
    handler = type("BoundHandler", (Handler,), {"app": app})
    srv = ThreadingHTTPServer((host, port), handler)
    handler.port = srv.server_address[1]
    return srv


PAGE = r"""<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>Creative Radar</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{color-scheme:light dark;--bg:#fff;--fg:#1b1d21;--mut:#68707d;--bd:#d9dde3;--ac:#2b6cdf;--card:#f5f7fa;--bad:#c0392b;--ok:#1e8e4e}
@media(prefers-color-scheme:dark){:root{--bg:#14161a;--fg:#e8eaee;--mut:#9aa3b0;--bd:#2c3038;--ac:#6ea0ff;--card:#1c1f25;--bad:#ff7b6b;--ok:#52d18a}}
*{box-sizing:border-box}body{margin:0;font:14px/1.45 system-ui,sans-serif;background:var(--bg);color:var(--fg);display:flex;flex-direction:column;height:100vh}
header{padding:8px 12px;border-bottom:1px solid var(--bd);display:flex;flex-wrap:wrap;gap:8px;align-items:center}
header h1{font-size:16px;margin:0 8px 0 0}.badge{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:2px 8px;font-size:12px}
input,select,button,textarea{font:inherit;color:inherit;background:var(--card);border:1px solid var(--bd);border-radius:6px;padding:4px 8px}
button{cursor:pointer}button:hover{border-color:var(--ac)}button.primary{background:var(--ac);color:#fff;border-color:var(--ac)}button.danger{color:var(--bad)}
main{flex:1;display:flex;min-height:0}#list{width:340px;overflow:auto;border-right:1px solid var(--bd)}
.item{display:flex;gap:8px;padding:8px;border-bottom:1px solid var(--bd);cursor:pointer}.item.sel{background:var(--card);outline:2px solid var(--ac);outline-offset:-2px}
.item img{width:48px;height:64px;object-fit:cover;background:var(--card);border-radius:4px;flex:none}.item .t{min-width:0}.item .t b,.item .t span{display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.pill{font-size:11px;border:1px solid var(--bd);border-radius:8px;padding:0 6px;color:var(--mut)}
#detail{flex:1;overflow:auto;padding:12px 16px}.cols{display:flex;gap:16px;flex-wrap:wrap}.vid{width:300px}
video,.nov{width:300px;height:533px;background:#000;border-radius:8px;object-fit:contain}.nov{display:flex;align-items:center;justify-content:center;color:#ccc;text-align:center;padding:12px}
.info{flex:1;min-width:300px}h3{margin:14px 0 4px;font-size:13px;text-transform:uppercase;color:var(--mut);letter-spacing:.04em}
table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid var(--bd);padding:3px 6px;text-align:left;vertical-align:top;font-size:13px}
.unk{color:var(--mut);font-style:italic}.warn{color:var(--bad)}.box{background:var(--card);border:1px solid var(--bd);border-radius:8px;padding:8px 10px;margin:6px 0}
.acts{display:flex;gap:6px;flex-wrap:wrap;margin:10px 0}.tag{display:inline-block;margin:2px 4px 2px 0}.OBSERVED{color:var(--ok)}.INFERRED{color:var(--mut)}
#msg{padding:6px 12px;border-top:1px solid var(--bd);min-height:30px}dialog{border:1px solid var(--bd);border-radius:10px;background:var(--bg);color:var(--fg)}
</style></head><body>
<header>
<h1>Creative Radar</h1><span class="badge" title="Вертикаль зафиксирована">Physical Products</span>
<select id="tab"><option value="feed">Лента</option><option value="shortlist">Shortlist</option><option value="skip">Пропущенные</option><option value="exclude">Исключённые</option></select>
<input id="country" placeholder="Страна (US)" size="9"><select id="platform"><option value="">Платформа: все</option><option>tiktok</option></select>
<select id="origin"><option value="">Источник: все</option><option value="found">найдено поиском</option><option value="imported">импортировано</option></select>
<select id="period"><option value="">Период: все</option><option>7</option><option>30</option><option>180</option></select>
<input id="q" placeholder="Поиск: товар, бренд" size="18">
<span id="counts" class="badge"></span><span id="targetOffer" class="badge"></span>
<button id="btnFind" class="primary">Найти (FIND)</button><button id="btnImport">Импорт файла</button>
</header>
<main><div id="list"></div><div id="detail"><p class="unk">Лента пуста. Нажмите «Найти (FIND)» или импортируйте сохранённый файл discovery.</p></div></main>
<div id="msg"></div>
<dialog id="dlgFind"><form method="dialog"><h3>FIND: поиск товарных рекламных источников</h3>
<p class="unk">Каждый запуск может делать до N платных запросов к SocialCrawl. В режиме «Целевой товар» часть бюджета резервируется для Ads Top/performance; в режиме «Смежный товар» выполняется только Sponsored Search по явным ключевым словам. Source Gate и Target-Fit остаются отдельными проверками. Автоматический анализ здесь не запускается.</p>
<p>Страна <input id="fCountry" value="US" size="4"> Период <select id="fPeriod"><option>7</option><option selected>30</option><option>180</option></select></p>
<p>Запросов (макс.) <input id="fReq" type="number" min="1" max="10" value="3" style="width:60px"> Объявлений <input id="fAds" type="number" min="1" max="50" value="20" style="width:60px"></p>
<p>Режим отбора <select id="fMode"><option value="product">Целевой товар + Ads Top</option><option value="mechanism">Смежный товар / поиск механики</option></select></p>
<p id="fModeInfo" class="unk">Товарный режим: текущий поиск и ранжирование; Source Gate и Target-Fit остаются отдельными проверками.</p>
<p>Ключевые слова (через запятую)<br><input id="fKw" size="40"></p>
<p><button value="go" class="primary">Запустить поиск</button> <button value="cancel">Отмена</button></p></form></dialog>
<dialog id="dlgImp"><form method="dialog"><h3>Импорт сохранённого discovery-файла</h3>
<p class="unk">Формат старого CME (tiktok_discovery.json). Кандидаты помечаются как «импортировано».</p>
<p><input id="iPath" size="50" placeholder="/путь/к/tiktok_discovery.json"></p>
<p><button value="go" class="primary">Импортировать</button> <button value="cancel">Отмена</button></p></form></dialog>
<script>
const TOKEN="__TOKEN__";let S={items:[],i:-1,state:null,selectedId:null};
const $=id=>document.getElementById(id);
function el(t,a,...k){const e=document.createElement(t);for(const[x,v]of Object.entries(a||{})){if(x==='class')e.className=v;else if(x==='on')for(const[ev,fn]of Object.entries(v))e.addEventListener(ev,fn);else e.setAttribute(x,v)}for(const c of k.flat())if(c!=null)e.append(c.nodeType?c:document.createTextNode(String(c)));return e}
async function api(m,p,b){const r=await fetch(p,{method:m,headers:{'Content-Type':'application/json','X-Radar-Token':TOKEN},body:b?JSON.stringify(b):undefined});return r.json()}
function say(t,bad){const m=$('msg');m.textContent=t;m.className=bad?'warn':''}
const UNK=v=>v===null||v===undefined||v===''?el('span',{class:'unk'},'н/д'):String(typeof v==='number'?Math.round(v*100)/100:v);
async function refreshState(){S.state=await api('GET','/api/state');const c=S.state.counts;$('counts').textContent=`найдено ${c.found} · просмотрено ${c.viewed} · в shortlist ${c.saved}`;const p=S.state.production||{};$('targetOffer').textContent=`Оффер: ${p.name||'не настроен'} · фото: ${p.state==='READY'?'PASS':p.state==='BLOCKED'?'BLOCKED':'проверить'}`}
function qs(){const p=new URLSearchParams({state:$('tab').value,country:$('country').value.trim(),platform:$('platform').value,origin:$('origin').value,period:$('period').value,q:$('q').value.trim()});return p.toString()}
async function loadFeed(keepId){const r=await api('GET','/api/feed?'+qs());S.items=r.items;let i=keepId?S.items.findIndex(x=>x.id===keepId):-1;renderList();if(S.items.length)select(i>=0?i:0,true);else{S.i=-1;$('detail').replaceChildren(el('p',{class:'unk'},'Нет кандидатов по текущим фильтрам.'))}await refreshState()}
function renderList(){const L=$('list');L.replaceChildren(...S.items.map((c,i)=>el('div',{class:'item'+(i===S.i?' sel':''),on:{click:()=>select(i)}},
 c.cover_url?el('img',{src:c.cover_url,loading:'lazy',referrerpolicy:'no-referrer',alt:'',on:{error:e=>e.target.replaceWith(el('div',{class:'pill'},'превью недоступно'))}}):el('div',{class:'pill'},'нет превью'),
 el('div',{class:'t'},el('b',{},c.brand_name||c.brand||'бренд н/д'),el('span',{},c.title||'название н/д'),el('span',{class:'pill'},(c.origin==='imported'?'импорт · ':'')+(c.operator.viewed?'просмотрено':'новое'))))))}
async function select(i,quiet){if(i<0||i>=S.items.length)return;S.i=i;renderList();const c=S.items[i];S.selectedId=c.id;document.querySelectorAll('.item')[i]?.scrollIntoView({block:'nearest'});
 await api('POST','/api/view',{id:c.id});const full=await api('GET','/api/candidate?id='+encodeURIComponent(c.id));if(S.selectedId!==c.id)return;renderDetail(full);await refreshProductions(c.id)}
function metricsTable(c){const rows=S.state.metric_fields.map(n=>{const m=c.metrics[n];return el('tr',{},el('td',{},n),el('td',{},m?String(m.value):UNK(null)),
 el('td',{class:'unk'},m?`${m.source} · ${(m.boards||[]).join(',')} · ${m.collected_at}`:'нет данных'))});return el('table',{},el('tr',{},el('th',{},'метрика'),el('th',{},'значение'),el('th',{},'происхождение, время сбора')),rows)}
function player(c){const box=el('div',{class:'vid'});if(!c.video_url){box.append(el('div',{class:'nov'},'Видео недоступно: в записи источника нет ссылки на видео.',c.cover_url?el('img',{src:c.cover_url,referrerpolicy:'no-referrer',style:'max-width:100%;margin-top:8px'}):null));return box}
 const v=el('video',{controls:'',playsinline:'',muted:'',preload:'metadata',src:c.video_url,referrerpolicy:'no-referrer'});v.muted=true;if(c.cover_url)v.poster=c.cover_url;
 const err=el('div',{class:'warn',style:'display:none;margin-top:4px'});v.addEventListener('error',()=>{err.style.display='block';err.textContent='Видео не воспроизводится: ссылка недоступна или устарела. Откройте оригинал.'});
 box.append(v,err);setTimeout(()=>{v.play().catch(()=>{})},50);return box}
function analysisView(a){if(!a)return el('p',{class:'unk'},'Анализ не запускался.');
 const head=el('div',{class:'box'},`Статус: ${a.status} · ${a.ts} · модель запрошена: ${a.model_requested||'н/д'} · ответила: ${a.model_responded||'не сообщено'}`);
 if(a.status!=='OK')return el('div',{},head,a.error?el('p',{class:'warn'},a.error):null);
 const r=a.result,m=a.material,names={hook:'Хук',buyer_problem:'Проблема покупателя',product_intervention:'Продуктовое вмешательство',result_demo:'Демонстрация результата',cta:'CTA',mechanism_hypothesis:'Гипотеза коммерческого механизма',transferability:'Переносимость'};
 return el('div',{},head,el('div',{class:'box unk'},`Модель видела только ${m.frames_sent} кадров (не всё видео): `+m.frames.map(f=>f.t_sec+'с').join(', ')),
  Object.entries(names).map(([k,t])=>el('div',{},el('h3',{},t),el('span',{class:r[k].kind},`[${r[k].kind==='OBSERVED'?'наблюдение':'интерпретация'}] `),r[k].text)),
  el('h3',{},'Доказательства (кадры)'),r.evidence.length?r.evidence.map(e=>el('div',{},`${e.t_sec}с — ${e.observation}`)):el('p',{class:'unk'},'нет подтверждённых временных меток'),
  el('h3',{},'Ограничения и недостающие данные'),el('div',{},r.limits.text),r.limits.missing.map(x=>el('div',{class:'unk'},'• '+x)),
  el('h3',{},'Оценка механизма (гипотеза AI, не доказательство прибыльности)'),
  el('table',{},Object.entries(r.ratings).map(([k,v])=>el('tr',{},el('td',{},k),el('td',{},v===null?UNK(null):v+' / 5')))),
  r.warnings.length?el('div',{class:'warn'},'Предупреждения проверки: '+r.warnings.join('; ')):null)}
async function refreshProductions(cid){const panel=$('productionPanel');if(!panel||S.selectedId!==cid)return;let data;try{data=await api('GET','/api/productions?id='+encodeURIComponent(cid))}catch(e){return}if(!$('productionPanel')||S.selectedId!==cid)return;
 panel.replaceChildren(el('h3',{},'Производство: атомы → кадр → QA'));
 if(!data.items?.length){panel.append(el('p',{class:'unk'},'Пока нет кадров производства. Источник должен быть в Shortlist и иметь успешный анализ.'));return}
 for(const run of data.items){panel.append(el('div',{class:'box'},el('b',{},`${run.status} · ${run.offer_name||run.offer_id||'оффер'} · ${run.started_at||''}`),
  el('div',{class:'unk'},`run: ${run.run_id} · analysis: ${run.analysis_id} · exit: ${run.exit_code??'н/д'}`),
  run.images?.length?el('div',{class:'acts'},run.images.map(name=>el('img',{src:'/api/production-file?'+new URLSearchParams({run:run.run_id,name}),alt:name,loading:'lazy',style:'max-width:240px;max-height:420px;object-fit:contain;border:1px solid #aaa;margin:4px'}))):el('p',{class:'warn'},'Изображения не получены; откройте лог ниже.'),
  run.verdicts?.length?el('div',{class:'unk'},'QA: '+run.verdicts.map(v=>`shot ${v.shot??'?'} ${v.overall||'unknown'} (${v.commercial_status||'status n/a'})`).join(' · ')):null,
  run.stdout_tail?el('details',{},el('summary',{},'Лог запуска'),el('pre',{},run.stdout_tail)):null,
  run.stderr_tail?el('pre',{class:'warn'},run.stderr_tail):null))}}
function renderDetail(c){const sc=c.source_score,sk=c.source_check,op=c.operator,D=$('detail');
 const act=(label,fn,cls)=>el('button',{class:cls||'',on:{click:fn}},label);
 const decide=async(action,need)=>{let reason='';if(need){reason=prompt(need);if(reason===null)return;if(!reason.trim()){say('Причина обязательна',true);return}}
  const r=await api('POST','/api/decide',{id:c.id,action,reason});if(r.error){say(r.error,true);return}say('Сохранено');const next=S.items[S.i+1]?.id;await loadFeed(action==='skip'||action==='exclude'||action==='shortlist'||action==='restore'?next:c.id)};
 D.replaceChildren(el('div',{class:'cols'},player(c),el('div',{class:'info'},
  el('h2',{style:'margin:0'},c.brand_name||c.brand||'бренд н/д'),el('div',{},c.title||UNK(null)),
  el('div',{class:'unk'},`${c.platform} · ${c.country||'гео н/д'} · период ${c.period_days||'н/д'} дн. · ${c.origin==='imported'?'ИМПОРТ (не найдено поиском)':'найдено поиском'} · товар/индустрия: ${c.industry||'н/д'}`),
  el('div',{class:'unk'},`длительность: ${c.duration_seconds??'н/д'} · первое появление: ${c.first_seen} · последнее: ${c.last_seen}`),
  el('div',{class:'acts'},act('Анализировать',async()=>{if(!confirm('Анализ делает один запрос к модели по кадрам этого ролика. Продолжить?'))return;say('Анализ выполняется…');const r=await api('POST','/api/analyze',{id:c.id});
   say(r.status==='OK'?'Анализ сохранён':`${r.status}: ${r.detail||''} ${r.how_to_fix||''}`,r.status!=='OK');const full=await api('GET','/api/candidate?id='+encodeURIComponent(c.id));renderDetail(full)},'primary'),
   act('В shortlist',()=>decide('shortlist','Почему отбираете это объявление?')),act('Пропустить',()=>decide('skip')),act('Исключить',()=>decide('exclude','Причина исключения:'),'danger'),
   c.source_url?act('Открыть оригинал',()=>window.open(c.source_url,'_blank','noopener')):el('span',{class:'unk'},'нет ссылки на оригинал'),
   op.state?act('Вернуть в ленту',()=>decide('restore')):null,
   op.state==='shortlist'&&c.analyses?.some(a=>a.status==='OK'&&a.result)?act('Создать кадр',async()=>{if(!confirm('Будет вызван текущий Brain и генератор изображения для выбранного целевого оффера. Фото товара проверяется до вызовов. Продолжить?'))return;say('Передача анализа в атомы и производство кадра…');const r=await api('POST','/api/produce',{id:c.id});say(r.status==='OK'?`Производство завершено: ${r.images?.length||0} изображений. Вердикт см. ниже.`:`${r.status}: ${r.detail||r.stderr_tail||'подробности в панели результата'}`,r.status!=='OK');await refreshProductions(c.id)},'primary'):null),
  op.state?el('div',{class:'box'},`Решение оператора: ${op.state} · ${op.reason||'без причины'} · ${op.updated_at}`):null,
  el('h3',{},'Оценка источника (старый ранжировщик, только по данным источника)'),
  sc?el('table',{},['final_score','performance_score','commercial_score','funnel_fit_score','evidence_score','longevity_score','duration_fit_score','action'].map(k=>el('tr',{},el('td',{},k),el('td',{},UNK(sc[k]))))):el('p',{class:'unk'},'оценки нет (импорт без решения старого ранжировщика)'),
  el('h3',{},'Проверка источника'),el('div',{class:'box'},`статус: ${sk.status} · класс: ${sk.classification??'н/д'} · scope: ${sk.scope??'н/д'}`,el('div',{class:'unk'},sk.reason||'')),
  el('h3',{},'Source Selection (до Target-Fit)'),c.source_selection?el('div',{class:'box'},
   el('b',{},`mode=${c.source_selection.mode||'н/д'} · role=${c.source_selection.role||'н/д'} · stage=${c.stage||'н/д'}`),
   el('div',{},`source_gate_eligible=${c.source_gate_eligible===true?'true':'false'} · target_fit_status=${c.target_fit_status||'NOT_EVALUATED'}`),
   el('div',{},`attention proxy=${c.source_selection.attention?.score??'н/д'} / 100 · coverage=${c.source_selection.attention?.coverage??'н/д'} · commercial proof=no`),
   el('div',{class:'unk'},c.source_selection.role_reason||c.source_selection.reason||'')):
   el('p',{class:'unk'},'Source Selection ещё не классифицирована.'),
  el('h3',{},'Коммерческие доказательства (не прогноз прибыли)'),c.commercial_evidence?el('div',{class:'box'},
    el('b',{},c.commercial_evidence.level||'NO_EVIDENCE'),el('div',{},c.commercial_evidence.explanation||''),
    el('div',{class:'unk'},'Сигналы: '+((c.commercial_evidence.signals||[]).join(', ')||'нет')),
    (c.commercial_evidence.missing||[]).length?el('div',{class:'unk'},'Не хватает: '+c.commercial_evidence.missing.join('; ')):null):el('p',{class:'unk'},'Нет классификации доказательств'),
  el('h3',{},'Почему попало в выдачу'),el('div',{class:'box'},c.why.explanation?(typeof c.why.explanation==='string'?c.why.explanation:JSON.stringify(c.why.explanation)):UNK(null),
   el('div',{},(c.why.tags||[]).map(t=>el('span',{class:'pill tag'},t)))),
  el('h3',{},'Метрики источника'),metricsTable(c),
  c.history.length>1?el('div',{class:'unk'},`снимков метрик в истории: ${c.history.length}`):null)),
  el('h3',{},'Анализ ролика (AI, по требованию)'),analysisView(c.analyses[0]),
  c.analyses.length>1?el('div',{class:'unk'},`предыдущих попыток анализа: ${c.analyses.length-1}`):null,
  el('section',{id:'productionPanel',class:'box'},el('h3',{},'Производство: атомы → кадр → QA'),el('p',{class:'unk'},'Загрузка истории производства…')));refreshProductions(c.id)}
$('btnFind').onclick=()=>{const s=S.state?.find;if(s&&s.state!=='READY'){say(`FIND ${s.state}: ${s.detail}. ${s.how_to_fix}`,true)}$('dlgFind').showModal()};
function syncFindMode(){
 const mechanism=$('fMode').value==='mechanism';
 $('fKw').required=mechanism;
 $('fKw').placeholder=mechanism?'Введите явные ключевые слова для смежного товара':'Ключевые слова товара (необязательно)';
 $('fModeInfo').textContent=mechanism
  ?'Режим механик использует только Sponsored Search. Роль товара определяется по тексту самого объявления; attention proxy не доказывает продажи. Target-Fit на этапе FIND не выполняется.'
  :'Товарный режим сохраняет прежний поиск. Данные внимания и коммерческие сигналы не заменяют Source Gate и независимый Target-Fit.';
}
$('fMode').addEventListener('change',syncFindMode);syncFindMode();
$('dlgFind').addEventListener('close',async()=>{if($('dlgFind').returnValue!=='go')return;say('Поиск выполняется…');
 const r=await api('POST','/api/find',{country:$('fCountry').value,period:+$('fPeriod').value,max_requests:+$('fReq').value,max_ads:+$('fAds').value,selection_mode:$('fMode').value,keywords:$('fKw').value.split(',').map(x=>x.trim()).filter(Boolean)});
 const d=r.diagnostics||{};
 const note=r.status==='OK'?(r.returned>0?`Найдено: новых ${r.new}, обновлено ${r.updated}. FIND ${d.outcome||'OK'}; DIRECT ${d.direct_relevant_ads??'н/д'}, EXPANDED ${d.expanded_relevant_ads??'н/д'}.`:`FIND ${d.outcome||'без кандидатов'}: API-строк ${d.raw_ads_returned??0}; DIRECT ${d.direct_relevant_ads??0}; EXPANDED ${d.expanded_relevant_ads??0}; непроверенных ${d.unverified_ads??0}. ${d.explanation||''}${(d.search_queries_executed||[]).length?' Запросы: '+d.search_queries_executed.join(', '):''}`):`${r.status}: ${r.detail||''} ${r.how_to_fix||''}`;
 say(note,r.status!=='OK'||!r.returned);await loadFeed()});
$('btnImport').onclick=()=>$('dlgImp').showModal();
$('dlgImp').addEventListener('close',async()=>{if($('dlgImp').returnValue!=='go')return;const r=await api('POST','/api/import',{path:$('iPath').value});
 say(r.status==='OK'?`Импортировано: новых ${r.new}, обновлено ${r.updated} (помечены как «импорт»)`:`${r.status}: ${r.detail}`,r.status!=='OK');await loadFeed()});
for(const id of['tab','country','platform','origin','period'])$(id).addEventListener('change',()=>loadFeed());
$('q').addEventListener('input',()=>{clearTimeout(window._t);window._t=setTimeout(()=>loadFeed(),250)});
document.addEventListener('keydown',e=>{if(['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName)||document.querySelector('dialog[open]'))return;
 if(e.key==='ArrowDown'||e.key==='ArrowRight'){e.preventDefault();select(S.i+1)}else if(e.key==='ArrowUp'||e.key==='ArrowLeft'){e.preventDefault();select(S.i-1)}
 else if(e.key===' '){const v=document.querySelector('video');if(v){e.preventDefault();v.paused?v.play().catch(()=>{}):v.pause()}}});
refreshState().then(()=>loadFeed());
</script></body></html>"""
