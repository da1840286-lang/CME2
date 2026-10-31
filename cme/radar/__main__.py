"""python -m cme.radar --data radar_data [--port 8765] [--import PATH]"""
from __future__ import annotations
import argparse
import sys
import webbrowser
from pathlib import Path
from .find import find_status
from .server import ROOT, RadarApp, make_server


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="cme.radar", description="Creative Radar: manual FIND review for Physical Products")
    ap.add_argument("--data", default="radar_data", help="folder for the database, frames and call log")
    ap.add_argument("--settings", default=None)
    ap.add_argument("--catalog", default=None)
    ap.add_argument("--offer", default=str(ROOT / "examples" / "offer_example.json"),
                    help="target OfferCard JSON; use a real product photo, not the example placeholder")
    ap.add_argument("--vertical", default=str(ROOT / "verticals" / "physical_goods_us.json"))
    ap.add_argument("--production-shots", type=int, default=1, help="number of sequential frames to request (1..6)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--import", dest="import_path", default=None, help="import an old CME discovery JSON before starting")
    ap.add_argument("--open", action="store_true", help="open the browser")
    a = ap.parse_args(argv)
    if not 1 <= a.production_shots <= 6:
        ap.error("--production-shots must be between 1 and 6")
    app = RadarApp(a.data, a.settings, a.catalog, offer_path=a.offer, vertical_path=a.vertical,
                   production_shots=a.production_shots)
    if a.import_path:
        print("import:", app.do_import({"path": a.import_path}))
    srv = make_server(app, port=a.port)
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    print(f"Creative Radar: {url}\nFIND: {find_status()['state']} | analyst: {app.analyst_status()['state']} | data: {a.data}")
    if a.open:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
