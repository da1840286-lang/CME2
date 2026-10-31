"""Real-browser test of Creative Radar (skipped when Playwright/Chromium is unavailable). SYNTHETIC data only."""
import functools, http.server, json, os, shutil, subprocess, tempfile, threading, unittest
from pathlib import Path
from unittest import mock

import cme.radar.find as F
from cme.radar.legacy.discovery import tiktok_socialcrawl as T
from cme.radar.server import RadarApp, make_server
from cme.radar.store import Store
from tests.radar_fixtures import FakeResp, synthetic_post, synthetic_response

try:
    from playwright.sync_api import sync_playwright
    with sync_playwright() as _p:
        _p.chromium.launch().close()
    HAS_BROWSER = True
except Exception:
    HAS_BROWSER = False

HAS_FFMPEG = bool(shutil.which("ffmpeg"))
ROOT = Path(__file__).resolve().parent.parent
XSS_TITLE = "<img src=x onerror=window.__xss=1> [SYNTHETIC TEST] kitchen gadget"


@unittest.skipUnless(HAS_BROWSER, "Playwright/Chromium not available")
class RadarInBrowser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        media = Path(cls.tmp.name) / "media"
        media.mkdir()
        if HAS_FFMPEG:
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=duration=3:size=270x480:rate=10",
                            "-c:v", "libvpx", "-pix_fmt", "yuv420p", str(media / "ok.webm")], check=True)  # VP8: plays in open-source Chromium
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(media))
        handler.log_message = lambda *a, **k: None
        cls.media_srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=cls.media_srv.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.media_srv.server_address[1]}"
        cls.pw = sync_playwright().start()
        cls.browser = cls.pw.chromium.launch(args=["--autoplay-policy=no-user-gesture-required"])

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.pw.stop()
        cls.media_srv.shutdown()
        cls.media_srv.server_close()
        cls.tmp.cleanup()

    def setUp(self):
        """Fresh database and server per test: tests never see each other's decisions."""
        self.work = tempfile.TemporaryDirectory()
        tmp = Path(self.work.name)
        for k in ("SOCIALCRAWL_API_KEY", "OPENROUTER_API_KEY"):
            os.environ.pop(k, None)
        items = [synthetic_post("7001", brand="alphabrand", title=XSS_TITLE, video_url=self.base + "/ok.webm"),
                 synthetic_post("7002", brand="betabrand", video_url="http://127.0.0.1:9/dead.mp4"),
                 synthetic_post("7003", brand="gammabrand", views=None, ctr=None)]
        with mock.patch.object(T.requests, "get", lambda *a, **k: FakeResp(synthetic_response(items))), \
                mock.patch.dict(os.environ, {"HOME": str(tmp)}):
            cands = F.run_find({"country": "US", "max_requests": 1}, client=T.SocialCrawlTikTokTopAds(api_key="K"))["candidates"]
        # the old engine drops ads without a video, so a record without one can only arrive by import
        gamma = next(c for c in cands if c["ad_id"] == "7003")
        gamma["video_url"], gamma["origin"] = None, "imported"
        self.app = RadarApp(tmp / "radar_data", settings_path=tmp / "none.json", catalog_path=ROOT / "adapters.json",
                            tools={"state": "READY", "detail": "", "how_to_fix": ""})
        self.app.store.ingest(cands, "ui-test", "found")
        self.srv = make_server(self.app, port=0)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}/"

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()
        self.app.store.close()
        self.work.cleanup()

    def page(self):
        pg = self.browser.new_page(viewport={"width": 1280, "height": 800})
        self.errors = []
        pg.on("pageerror", lambda e: self.errors.append(str(e)))
        pg.goto(self.url)
        pg.wait_for_selector(".item")
        return pg

    def test_feed_navigation_video_and_honest_placeholders(self):
        pg = self.page()
        self.assertEqual(pg.locator(".item").count(), 3)
        self.assertIn("найдено 3", pg.inner_text("#counts"))
        self.assertIn("Physical Products", pg.inner_text("header"))
        # HTML from an ad title is shown as text, never executed
        self.assertIsNone(pg.evaluate("window.__xss"))
        self.assertIn("<img src=x", pg.inner_text("#list"))
        first_brand = pg.inner_text("#detail h2")
        pg.keyboard.press("ArrowDown")
        pg.wait_for_function("prev => document.querySelector('#detail h2').innerText !== prev", arg=first_brand)
        pg.keyboard.press("ArrowDown")
        pg.wait_for_function("document.querySelector('#detail h2').innerText.toLowerCase().includes('gammabrand')")
        self.assertIn("Видео недоступно", pg.inner_text("#detail"))     # no video URL: honest, no fake player
        self.assertEqual(pg.locator("#detail video").count(), 0)
        self.assertIn("н/д", pg.inner_text("#detail"))                  # unknown metrics shown as unknown
        pg.keyboard.press("ArrowUp")
        pg.wait_for_function("document.querySelector('#detail h2').innerText.toLowerCase().includes('betabrand')")
        pg.wait_for_selector("#detail video")
        pg.wait_for_selector("text=Видео не воспроизводится", timeout=8000)  # dead link: honest error
        pg.close()
        self.assertEqual(self.errors, [])

    @unittest.skipUnless(HAS_FFMPEG, "ffmpeg not installed")
    def test_current_video_loads_and_only_the_selected_one_is_a_player(self):
        pg = self.page()
        pg.wait_for_selector("#detail video")
        pg.wait_for_function("document.querySelector('#detail video').duration > 0", timeout=8000)
        self.assertEqual(pg.locator("video").count(), 1)
        pg.close()

    def test_analyze_without_key_is_not_configured_and_no_fake_result(self):
        pg = self.page()
        pg.on("dialog", lambda d: d.accept())
        pg.get_by_role("button", name="Анализировать", exact=True).click()
        pg.wait_for_function("document.querySelector('#msg').innerText.includes('NOT_CONFIGURED')")
        self.assertIn("settings", pg.inner_text("#msg"))
        self.assertIn("Анализ не запускался", pg.inner_text("#detail"))
        pg.close()

    def test_find_without_key_reports_state_and_keeps_feed(self):
        pg = self.page()
        pg.click("#btnFind")
        pg.get_by_role("button", name="Запустить поиск").click()
        pg.wait_for_function("(() => { const t = document.querySelector('#msg').innerText; return t.includes('NOT_CONFIGURED') && !t.includes('выполняется'); })()")
        self.assertIn("SOCIALCRAWL_API_KEY", pg.inner_text("#msg"))
        pg.close()

    def test_shortlist_skip_exclude_and_persistence_after_reload(self):
        pg = self.page()
        reasons = iter(["сильная демонстрация", "не физический товар"])
        pg.on("dialog", lambda d: d.accept(next(reasons)) if d.type == "prompt" else d.accept())
        pg.get_by_role("button", name="В shortlist", exact=True).click()
        pg.wait_for_function("document.querySelector('#counts').innerText.includes('в shortlist 1')")
        pg.get_by_role("button", name="Исключить", exact=True).click()
        pg.wait_for_function("document.querySelectorAll('.item').length === 1")
        self.assertEqual(pg.locator(".item").count(), 1)
        pg.close()
        # a fresh page load, and then a fresh Store on the same file, still have both decisions
        pg2 = self.browser.new_page()
        pg2.goto(self.url)
        pg2.wait_for_function("document.querySelector('#counts').innerText.includes('в shortlist 1')")
        pg2.select_option("#tab", "shortlist")
        pg2.wait_for_function("t => document.querySelector('#detail').innerText.includes(t)", arg="сильная демонстрация")
        pg2.select_option("#tab", "exclude")
        pg2.wait_for_function("t => document.querySelector('#detail').innerText.includes(t)", arg="не физический товар")
        pg2.close()
        reopened = Store(self.app.store.path)
        self.assertEqual(len(reopened.list(state="shortlist")), 1)
        self.assertEqual(len(reopened.list(state="exclude")), 1)
        reopened.close()

    def test_screenshot_for_review(self):
        pg = self.page()
        out = os.environ.get("RADAR_SCREENSHOT")
        if out:
            pg.screenshot(path=out)
        pg.close()


if __name__ == "__main__":
    unittest.main()
