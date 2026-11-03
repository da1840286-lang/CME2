import os

# The pre-existing reference-cleaning tests exercise the legacy Tesseract engine. Production defaults to the
# PP-OCR + LaMa engine (cme.text_clean); tests for that engine select it explicitly (see test_text_clean.py).
os.environ.setdefault("CME_UNBRANDED_ENGINE", "legacy")
