"""SYNTHETIC TEST FIXTURES. Never real ads, never mixed with a real shortlist or database."""
import json


def synthetic_post(ad_id, brand="synthbrand", title="[SYNTHETIC TEST] Shop now free shipping on our kitchen gadget",
                   ctr=2.4, cvr=1.1, views=120000, video_url="https://example.invalid/v/{id}.mp4", duration=21, text_extra=""):
    return {"post": {
        "id": ad_id, "url": f"https://example.invalid/ad/{ad_id}?sig=abc",
        "author": {"username": brand, "display_name": brand.title()},
        "content": {"text": title + text_extra, "media_urls": [video_url.format(id=ad_id)], "thumbnail_url": f"https://example.invalid/c/{ad_id}.jpg",
                    "duration_seconds": duration},
        "engagement": {"views": views, "likes": 900, "comments": 12, "shares": 5},
        "ext": {"ad": {"ctr": ctr, "cvr": cvr, "rank": 3, "industry": "Home & Garden", "objective": "Conversions",
                       "landing_page_url": "https://example.invalid/shop/products/gadget"}},
        "video": {"id": f"v{ad_id}"}},
        "computed": {"language": "en"}}


def synthetic_response(items):
    return {"success": True, "data": {"items": items}, "credits_used": 1, "credits_remaining": 99, "request_id": "synthetic"}


class FakeResp:
    def __init__(self, payload, status=200):
        self.payload, self.status_code, self.ok = payload, status, status < 400

    def json(self):
        return self.payload
