"""
================================================================================
AI-ASSISTED KYC DOCUMENT REVIEW AGENT
================================================================================

WHAT THIS IS
------------
A portfolio project demonstrating an AI agent pipeline for identity document
review. Upload an ID document (license, passport, etc.) and the system:

  1. EXTRACTS structured fields (name, DOB, ID number, expiry) using a free,
     open vision-language model (with automatic fallback across providers)
     via HuggingFace Inference Providers
  2. VISUALLY INSPECTS the same image for signs of tampering (font
     mismatches, misalignment, blur inconsistencies, background seams) -
     the SAME model call does both jobs at once, returning structured JSON
  3. RUNS ERROR LEVEL ANALYSIS (ELA) - a classical, non-AI image forensics
     technique - as an independent second opinion. ELA re-compresses the
     image and diffs it against the original; regions that were edited/
     pasted in later often show a different "error level" than untouched
     regions, which this app summarizes into a single 0-100 fraud_score.
  4. COMBINES all three signals (extraction confidence + AI visual flags +
     ELA score) into one pass / flagged / fail recommendation, with plain-
     English reasons
  5. LOGS every review to SQLite, including what the human reviewer finally
     decided - so real accuracy numbers can be reported later instead of
     just demoing the happy path

WHY IT'S BUILT THIS WAY
------------------------
- The AI NEVER auto-approves or auto-rejects on its own. A human always
  makes the final call - the system's job is to make that decision fast
  and well-informed, not to replace the person making it.
- TWO INDEPENDENT fraud signals (AI visual inspection + classical ELA)
  are used instead of one, because each has different blind spots. The AI
  can reason about context (an odd font, a weird phrasing) that ELA can't
  see; ELA can catch pixel-level edits that "look fine" to a glance but
  were re-compressed differently than the rest of the image.
- Uses FREE, open models (with automatic provider fallback) through
  HuggingFace Inference Providers rather than a paid API - built to run
  entirely within a free or HF PRO account, so this project costs nothing
  to build or demo. Falls over to the next model/provider combo if one
  errors out, since individual providers occasionally reject or fail on
  specific images/models for reasons unrelated to the app itself.
- Every review is logged with the human's final decision, which is what
  eventually lets you measure real precision/recall instead of only
  showing the happy path in a demo.

HONEST LIMITATIONS (worth saying out loud in interviews)
----------------------------------------------------------
- ELA has known false positives on legitimately re-compressed/resized
  images - it's a SIGNAL, not an auto-reject rule.
- This is a proof of concept, not a production KYC system. Real systems
  also need liveness detection, sanctions-list/database checks, and
  regulatory compliance review.
- Extraction accuracy depends heavily on image quality, lighting, and
  which open model is used - open models are generally a bit less
  reliable at strict JSON formatting than closed frontier models, which
  is why this code defensively handles malformed JSON output.
- Inference Providers' model catalogs change over time - a model/provider
  combo that works today can stop being hosted later. CANDIDATE_MODELS
  below is a list specifically so the app can fail over instead of
  breaking outright when that happens; if ALL candidates ever fail, check
  huggingface.co/docs/inference-providers/tasks/image-text-to-text for
  current live options and update the list.

SETUP
-----
1. Set your HF_TOKEN as a Space secret (Settings -> Repository secrets),
   or export it locally: export HF_TOKEN=your_token_here
   Get a free token at https://huggingface.co/settings/tokens
   The token needs the "Make calls to Inference Providers" permission -
   fine-grained tokens don't have this on by default.
2. Make sure requirements.txt includes: streamlit, huggingface_hub,
   pillow, python-dateutil, numpy, pymupdf
3. On HF Spaces, this file should be named app.py with SDK set to
   "Streamlit" in your Space's README/config.

================================================================================
"""

import streamlit as st
import os
import tempfile
import json
import sqlite3
import base64
import io
from datetime import date, datetime, timezone

import numpy as np
from PIL import Image, ImageChops
from dateutil import parser as date_parser
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError
import fitz  # PyMuPDF - used to convert PDF pages to images


# ==============================================================================
# SECTION 1: DATABASE (was db.py)
# Handles storage of every document review: what was extracted, what
# validation found, and what the human reviewer decided. This log is what
# lets you report real accuracy numbers later.
# ==============================================================================

DB_PATH = "kyc_reviews.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            extracted_json TEXT,
            validation_status TEXT,
            validation_reasons TEXT,
            human_decision TEXT,
            created_at TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def save_review(filename, extracted, validation_status, validation_reasons):
    conn = get_connection()
    cur = conn.execute(
        """
        INSERT INTO reviews (filename, extracted_json, validation_status,
                              validation_reasons, human_decision, created_at)
        VALUES (?, ?, ?, ?, NULL, ?)
        """,
        (
            filename,
            json.dumps(extracted),
            validation_status,
            json.dumps(validation_reasons),
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


def update_decision(row_id, decision):
    """decision should be 'approved' or 'rejected'."""
    conn = get_connection()
    conn.execute(
        "UPDATE reviews SET human_decision = ? WHERE id = ?", (decision, row_id)
    )
    conn.commit()
    conn.close()


def get_all_reviews():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM reviews ORDER BY id DESC").fetchall()
    conn.close()
    return rows


# ==============================================================================
# SECTION 2: FRAUD DETECTION - ERROR LEVEL ANALYSIS (was fraud_detection.py)
# A classical, non-AI image-forensics technique that runs ALONGSIDE the
# model's visual inspection, as an independent check.
# ==============================================================================

# Shared cap for any image we downscale before sending to the vision API or
# saving to disk for display - phone-camera photos can be huge (3000px+ per
# side), and an oversized image loading in late visibly jolts the Streamlit
# layout as it resizes.
MAX_IMAGE_DIMENSION = 2048

# The uploaded document preview is just being eyeballed by a human, so it
# doesn't need to be sharp/large - a full-resolution phone-camera photo
# loading in without any reserved layout space visibly jolted the page.
DISPLAY_IMAGE_DIMENSION = 900


def _make_display_copy(image_path, max_dim=DISPLAY_IMAGE_DIMENSION, quality=85):
    """Small JPEG copy of an image for display only - the original at
    image_path is left untouched for actual processing (ELA, the vision
    API call), which need the closest thing to the original data."""
    image = Image.open(image_path).convert("RGB")
    if max(image.size) > max_dim:
        image.thumbnail((max_dim, max_dim), Image.LANCZOS)
    display_path = image_path.rsplit(".", 1)[0] + "_preview.jpg"
    image.save(display_path, "JPEG", quality=quality)
    return display_path


def crop_region(image_path, bbox, label, padding=0.15, max_dim=DISPLAY_IMAGE_DIMENSION):
    """
    Crops a normalized [x1, y1, x2, y2] (each 0.0-1.0) bounding box out of
    image_path, with padding added around it for context, and saves it as a
    small labeled JPEG. Returns the crop's path, or None if bbox is
    missing/malformed/degenerate.

    The bounding box comes from the vision model's own best-effort spatial
    grounding - general-purpose VLMs (as opposed to models specifically
    trained for object detection) are often imprecise at this, which is why
    this pads generously rather than cropping tight: a slightly-too-wide
    crop that still contains the field is far more useful to a reviewer
    than a pixel-tight crop that's cut off the actual content because the
    model's coordinates were a bit off.
    """
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return None

    x1, x2 = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
    y1, y2 = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
    if x2 - x1 < 0.01 or y2 - y1 < 0.01:
        return None  # degenerate box - not worth cropping

    pad_x, pad_y = (x2 - x1) * padding, (y2 - y1) * padding
    x1, x2 = max(0.0, x1 - pad_x), min(1.0, x2 + pad_x)
    y1, y2 = max(0.0, y1 - pad_y), min(1.0, y2 + pad_y)

    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    box_px = (int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h))
    if box_px[2] <= box_px[0] or box_px[3] <= box_px[1]:
        return None

    crop = image.crop(box_px)
    if max(crop.size) > max_dim:
        crop.thumbnail((max_dim, max_dim), Image.LANCZOS)

    crop_path = image_path.rsplit(".", 1)[0] + f"_crop_{label}.jpg"
    crop.save(crop_path, "JPEG", quality=88)
    return crop_path


def compute_ela(image_path: str, quality: int = 90) -> float:
    """
    Runs Error Level Analysis on an image.

    How it works:
    1. Re-save the image as JPEG at a known quality.
    2. Diff it against the original, pixel by pixel.
    3. Untouched regions of a genuine photo lose detail fairly uniformly
       when re-compressed. Regions edited/pasted in later were often saved
       at a DIFFERENT compression generation, so they "light up" brighter/
       differently in the difference map.
    4. Summarize that map's regional variance into a single 0-100
       fraud_score.

    Returns: fraud_score
    """
    original = Image.open(image_path).convert("RGB")

    buffer_path = image_path + "_resaved_tmp.jpg"
    original.save(buffer_path, "JPEG", quality=quality)
    resaved = Image.open(buffer_path)

    diff = ImageChops.difference(original, resaved)
    diff_array = np.array(diff).astype(float)

    max_diff = diff_array.max() if diff_array.max() > 0 else 1
    scale = 255.0 / max_diff
    amplified = np.clip(diff_array * scale, 0, 255).astype(np.uint8)

    grayscale = np.array(Image.fromarray(amplified).convert("L")).astype(float)
    region_std = _regional_std(grayscale)
    # Divisor calibrated empirically (not guessed): region_std for a clean,
    # low-detail photo lands near 0; a clean but text/edge-heavy document
    # (which legitimately produces some ELA signal even with zero tampering)
    # lands around 2-3; a visibly pasted-in patch from a different JPEG
    # generation lands around 5-6. A divisor of 40 made even obvious
    # tampering score under 15/100, so the flag threshold below was
    # effectively unreachable - it never fired regardless of input. 8.0
    # spreads that same range across roughly 0-70+, so the score actually
    # differentiates instead of clustering near zero for everything.
    fraud_score = float(np.clip(region_std / 8.0 * 100, 0, 100))

    os.remove(buffer_path)
    return round(fraud_score, 1)


def _regional_std(grayscale_array, block_size=16):
    """
    Splits the image into blocks and measures how much block-average
    brightness varies across the image. High variance = some regions have
    a very different error level than others = possible tamper site.

    Vectorized via reshape instead of a Python loop over every block (was
    ~15x slower on a large photo) - verified to produce bit-identical
    results to the loop version it replaced, including on image sizes that
    aren't an exact multiple of block_size.
    """
    h, w = grayscale_array.shape
    n_y = len(range(0, h - block_size, block_size))
    n_x = len(range(0, w - block_size, block_size))
    if n_y == 0 or n_x == 0:
        return 0.0
    cropped = grayscale_array[:n_y * block_size, :n_x * block_size]
    block_means = cropped.reshape(n_y, block_size, n_x, block_size).mean(axis=(1, 3))
    return float(np.std(block_means))


# ==============================================================================
# SECTION 3: EXTRACTION + AI VISUAL INSPECTION (was extraction.py)
# Uses a free, open vision-language model via HuggingFace Inference
# Providers to extract fields AND inspect for tampering signs in a
# single call, forced into structured JSON output.
# ==============================================================================

# We try a LIST of model+provider combos in order, falling over to the next
# one if a provider errors out. Each entry here was verified LIVE against
# huggingface.co/api/models/<model>?expand[]=inferenceProviderMapping before
# being added - Inference Providers' catalogs change over time, so a combo
# that isn't independently verified tends to go stale fast (this list
# previously pointed at meta-llama/Llama-3.2-11B-Vision-Instruct via novita
# and nscale, and Qwen/Qwen2.5-VL-3B-Instruct via auto-routing - all three
# had stopped being hosted by the time this ran). If ALL candidates below
# ever fail, check huggingface.co/docs/inference-providers/tasks/image-text-to-text
# for what's currently live and update this list.
CANDIDATE_MODELS = [
    ("google/gemma-4-31B-it", "together"),
    ("google/gemma-4-31B-it", "deepinfra"),
    ("google/gemma-4-31B-it", "novita"),
    ("google/gemma-4-31B-it", "featherless-ai"),
]

_hf_token = os.environ.get("HF_TOKEN")

EXTRACTION_PROMPT = """You are a KYC document inspection assistant. You will be shown \
an identity document (license, passport, or ID card).

Do two things and respond with ONLY valid JSON, no other text, no markdown code fences, \
and no pretty-printing - output it as a single compact line with no extra whitespace \
or line breaks, to keep the response short. Do not think out loud, do not explain your \
reasoning, and do not write anything before or after the JSON - your entire response \
must be the JSON object and nothing else:

1. EXTRACT these fields as best you can read them, and for each one give its
   approximate location in the image as "bbox": [x1, y1, x2, y2], where each
   number is a FRACTION from 0.0 to 1.0 of the image's width (for x1/x2) or
   height (for y1/y2), with (0,0) at the top-left corner and (1,1) at the
   bottom-right corner - NOT pixel coordinates:
   - document_type (e.g. "drivers_license", "passport", "national_id") - no bbox needed
   - full_name
   - date_of_birth (YYYY-MM-DD if possible)
   - id_number
   - expiry_date (YYYY-MM-DD if possible)
   - each field's extraction_confidence: "high", "medium", or "low"

2. VISUALLY INSPECT for signs of digital tampering. Look specifically for:
   - Font inconsistencies between different text fields (mismatched typeface,
     weight, or size that doesn't match the document's official font)
   - Misaligned text (a field that sits slightly off-baseline vs the rest)
   - Inconsistent sharpness/blur (one region noticeably sharper or blurrier
     than the surrounding document, suggesting a pasted-in edit)
   - Background/texture seams around a field (visible rectangle edges,
     color mismatches, or texture discontinuities)
   Report what you observe honestly, including if nothing looks suspicious.
   For each issue you flag as true, add an entry to "flagged_regions" with a
   short label and its approximate bounding box in the same fractional
   [x1,y1,x2,y2] format, pointing at the specific area that looks suspicious.

Respond in exactly this JSON shape:
{
  "document_type": "...",
  "fields": {
    "full_name": {"value": "...", "confidence": "high|medium|low", "bbox": [0.0,0.0,1.0,1.0]},
    "date_of_birth": {"value": "...", "confidence": "high|medium|low", "bbox": [0.0,0.0,1.0,1.0]},
    "id_number": {"value": "...", "confidence": "high|medium|low", "bbox": [0.0,0.0,1.0,1.0]},
    "expiry_date": {"value": "...", "confidence": "high|medium|low", "bbox": [0.0,0.0,1.0,1.0]}
  },
  "visual_inspection": {
    "font_inconsistency": true/false,
    "misalignment": true/false,
    "blur_inconsistency": true/false,
    "background_seam": true/false,
    "notes": "short plain-English explanation of what you saw",
    "flagged_regions": [{"label": "e.g. font_inconsistency", "bbox": [0.0,0.0,1.0,1.0]}]
  }
}
"""


def _encode_image_data_uri(image_path):
    """
    Downscales/re-encodes as JPEG before sending to the vision model.
    Phone-camera ID photos are often several MB at 3000px+ per side; sending
    that raw as base64 can get rejected by some providers (e.g. a generic
    422 "no valid response generated" with no other explanation).
    """
    image = Image.open(image_path).convert("RGB")
    if max(image.size) > MAX_IMAGE_DIMENSION:
        image.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=90)
    b64 = base64.standard_b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def extract_and_inspect(image_path: str, retries_per_model: int = 0) -> dict:
    """
    Sends the ID image to a vision-language model (via HF Inference
    Providers) for extraction + visual fraud inspection in one call.
    Returns a parsed dict matching the JSON shape in EXTRACTION_PROMPT.

    Tries each (model, provider) combo in CANDIDATE_MODELS in order,
    falling over to the next combo immediately on any error (e.g. a
    provider-specific content-filter quirk, an auth issue, a truncated/
    malformed response, or a timeout). This matters in practice:
    individual providers occasionally reject or fail on specific images
    for reasons unrelated to image quality, so a single hardcoded
    model+provider is a fragile choice for a production-style pipeline.
    retries_per_model defaults to 0 (no same-provider retry on timeout) -
    a provider that's timing out is usually still slow/down on a second
    try, so retrying it just doubles the wait (up to the full timeout,
    twice) before ever reaching a candidate that might actually work;
    failing over immediately gets to a working provider faster.
    """
    import httpx

    image_uri = _encode_image_data_uri(image_path)
    errors_seen = []

    for model_id, provider in CANDIDATE_MODELS:
        client = InferenceClient(provider=provider, token=_hf_token, timeout=60)

        for attempt in range(1, retries_per_model + 2):
            try:
                response = client.chat_completion(
                    model=model_id,
                    max_tokens=4096,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": image_uri}},
                                {"type": "text", "text": EXTRACTION_PROMPT},
                            ],
                        }
                    ],
                )
                raw_text = response.choices[0].message.content.strip()
                finish_reason = response.choices[0].finish_reason

                if raw_text.startswith("```"):
                    raw_text = raw_text.strip("`")
                    if raw_text.lower().startswith("json"):
                        raw_text = raw_text[4:].strip()

                try:
                    return json.loads(raw_text)  # success - return immediately
                except json.JSONDecodeError as e:
                    truncated_hint = (
                        " - response was cut off (finish_reason='length'), hit "
                        "the max_tokens budget before finishing the JSON"
                        if finish_reason == "length" else ""
                    )
                    errors_seen.append(
                        f"{model_id} ({provider}): invalid JSON{truncated_hint}: {e}"
                    )
                    break  # move on to the next candidate, no point retrying

            except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.ConnectError) as e:
                if attempt <= retries_per_model:
                    continue  # retry the SAME model/provider once
                errors_seen.append(f"{model_id} ({provider}): timeout after retry - {e}")
                break  # move on to the next candidate

            except HfHubHTTPError as e:
                status = e.response.status_code if e.response is not None else None
                if status == 401:
                    errors_seen.append(
                        f"{model_id} ({provider}): 401 Unauthorized - HF_TOKEN likely "
                        f"lacks the \"Make calls to Inference Providers\" permission "
                        f"(huggingface.co/settings/tokens)"
                    )
                elif status == 422:
                    errors_seen.append(
                        f"{model_id} ({provider}): 422 - provider couldn't produce a "
                        f"valid response for this image"
                    )
                else:
                    errors_seen.append(f"{model_id} ({provider}): {e}")
                break  # move on to the next candidate, no point retrying this error type

    raise RuntimeError(
        "All AI model providers failed to produce a usable response. "
        "Details:\n" + "\n".join(errors_seen)
    )


# ==============================================================================
# SECTION 4: VALIDATION / GUARDRAILS (was validation.py)
# Combines extraction confidence + AI visual flags + ELA score into one
# final decision. Any ONE strong signal is enough to flag for review - the
# AI never auto-approves or auto-rejects on its own.
# ==============================================================================

ELA_FLAG_THRESHOLD = 35.0  # fraud_score above this contributes to a flag
# Sits between a clean but text/edge-heavy document (~25-30 under the
# recalibrated scoring in compute_ela) and a visibly pasted-in patch
# (~65-80) - see the comment there for how these ranges were derived.


def validate(extracted: dict, ela_score: float) -> dict:
    """
    Returns: {"status": "pass" | "flagged" | "fail", "reasons": [str, ...]}
    """
    reasons = []
    status = "pass"

    fields = extracted.get("fields", {})
    inspection = extracted.get("visual_inspection", {})

    # 1. Expiry check
    expiry_raw = fields.get("expiry_date", {}).get("value")
    if expiry_raw:
        try:
            expiry_date = date_parser.parse(expiry_raw).date()
            if expiry_date < date.today():
                status = "fail"
                reasons.append(f"Document expired on {expiry_date.isoformat()}")
        except (ValueError, TypeError):
            status = "flagged"
            reasons.append("Could not parse expiry date - needs human check")
    else:
        status = "flagged"
        reasons.append("No expiry date extracted")

    # 2. Extraction confidence check
    low_confidence_fields = [
        name for name, data in fields.items()
        if data.get("confidence") == "low"
    ]
    if low_confidence_fields:
        status = _escalate(status, "flagged")
        reasons.append(f"Low extraction confidence on: {', '.join(low_confidence_fields)}")

    # 3. AI visual tampering signals
    tamper_flags = [
        key for key in ("font_inconsistency", "misalignment",
                         "blur_inconsistency", "background_seam")
        if inspection.get(key) is True
    ]
    if tamper_flags:
        status = _escalate(status, "flagged")
        reasons.append(f"AI visual inspection flagged: {', '.join(tamper_flags)}")
        if inspection.get("notes"):
            reasons.append(f"AI notes: {inspection['notes']}")

    # 4. ELA forensic score
    if ela_score >= ELA_FLAG_THRESHOLD:
        status = _escalate(status, "flagged")
        reasons.append(
            f"Error Level Analysis score {ela_score} is above threshold "
            f"({ELA_FLAG_THRESHOLD}) - possible edited region"
        )

    if not reasons:
        reasons.append("All checks passed - no issues detected")

    return {"status": status, "reasons": reasons}


def _escalate(current_status: str, new_status: str) -> str:
    """Status severity: pass < flagged < fail. Never downgrade severity."""
    order = {"pass": 0, "flagged": 1, "fail": 2}
    return new_status if order[new_status] > order[current_status] else current_status


def convert_pdf_to_image(pdf_path: str, dpi: int = 200) -> str:
    """
    Converts the FIRST page of a PDF to a PNG image and returns the new
    file's path. ID documents are almost always scanned/exported as a
    single-page PDF, so we only need page 1.
    """
    doc = fitz.open(pdf_path)
    page = doc[0]
    zoom = dpi / 72  # PDF default is 72 DPI
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix)
    output_path = pdf_path.rsplit(".", 1)[0] + "_page1.png"
    pix.save(output_path)
    doc.close()
    return output_path


# ==============================================================================
# SECTION 5: STREAMLIT UI (was app.py)
# Two tabs: review a new document, and see history/stats across all
# past reviews (this is what you'd actually quote in interviews).
# ==============================================================================

st.set_page_config(page_title="AI KYC Review Agent", layout="wide")
init_db()

st.title("AI-Assisted KYC Document Review")
st.caption(
    "Upload an ID document. The system extracts its fields, checks for "
    "signs of tampering (AI visual inspection + Error Level Analysis), "
    "and gives you a fast, explained recommendation - final call is always yours."
)

st.info(
    "**What this tool does (and doesn't do):** it does NOT decide whether a "
    "document is fake or real. It reads the document, then flags anything "
    "that looks worth a closer look - like a smoke detector, not a "
    "firefighter. A human reviewer always makes the final approve/reject "
    "call, since AI and pixel-analysis signals can occasionally be wrong in "
    "both directions (missing real issues, or flagging harmless ones)."
)

tab_review, tab_history = st.tabs(["Review New Document", "History & Stats"])

# --- TAB 1: Review a new document ---
with tab_review:
    if not os.environ.get("HF_TOKEN"):
        st.warning(
            "No HF_TOKEN found. On HF Spaces, add it under Settings -> "
            "Repository secrets. Locally: `export HF_TOKEN=your_token_here`. "
            "Get a free token at huggingface.co/settings/tokens"
        )

    uploaded_file = st.file_uploader(
        "Upload an ID document (jpg/png/pdf)", type=["jpg", "jpeg", "png", "pdf"]
    )

    if uploaded_file is not None:
        # Cache the temp file path (and PDF conversion, and a small preview
        # copy) per uploaded file - without this, EVERY rerun (Approve/
        # Reject clicks, not just a fresh upload) rewrote a brand new temp
        # file, re-ran the PDF conversion, and re-served the full-size
        # original under a new path each time - one more source of layout
        # jolt beyond just the image being large.
        file_key = f"file_{uploaded_file.name}_{uploaded_file.size}"
        if file_key not in st.session_state:
            suffix = "." + uploaded_file.name.rsplit(".", 1)[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name

            # If a PDF was uploaded, convert its first page to an image -
            # everything downstream (extraction, ELA) works on images only.
            if suffix.lower() == ".pdf":
                with st.spinner("Converting PDF page 1 to an image..."):
                    try:
                        tmp_path = convert_pdf_to_image(tmp_path)
                    except Exception as e:
                        st.error(f"Could not read PDF: {e}")
                        st.stop()

            st.session_state[file_key] = {
                "tmp_path": tmp_path,
                "preview_path": _make_display_copy(tmp_path),
            }

        tmp_path = st.session_state[file_key]["tmp_path"]

        col_img, col_result = st.columns([1, 1.4])

        with col_img:
            # A resized preview, not the full-resolution original - a
            # phone-camera photo can be several MB, and a large image
            # loading in without any reserved layout space visibly jolted
            # the page.
            st.image(st.session_state[file_key]["preview_path"],
                       caption="Uploaded document", use_container_width=True)

        # Cache the pipeline result per uploaded file so it only runs once -
        # without this, Streamlit reruns the whole script (and would re-call
        # the API, re-run ELA, and insert a duplicate DB row) on every
        # widget interaction, including clicking Approve/Reject below.
        result_key = f"result_{uploaded_file.name}_{uploaded_file.size}"

        run_clicked = st.button("Run review", type="primary")
        if run_clicked:
            with st.spinner("Running extraction, visual inspection, and ELA... "
                              "(can take up to a minute if a model is cold-starting, "
                              "or longer if it has to fail over to a backup provider)"):
                try:
                    extracted = extract_and_inspect(tmp_path)
                    ela_score = compute_ela(tmp_path)
                    result = validate(extracted, ela_score)
                    row_id = save_review(
                        uploaded_file.name, extracted, result["status"], result["reasons"]
                    )

                    # Zoomed-in crops of each field, plus any specific area
                    # the AI flagged as suspicious - generated once here and
                    # cached, rather than a full-image heatmap the reviewer
                    # has to interpret. Crops are best-effort: the model's
                    # bounding boxes can be missing or imprecise (see
                    # crop_region's docstring), so any field without a
                    # usable one is simply skipped rather than shown broken.
                    field_crops = [
                        (field, data.get("value"), crop_path)
                        for field, data in extracted.get("fields", {}).items()
                        for crop_path in [crop_region(tmp_path, data.get("bbox"), field)]
                        if crop_path
                    ]
                    flagged_crops = [
                        (region.get("label", "flagged area"), crop_path)
                        for region in extracted.get("visual_inspection", {}).get("flagged_regions", [])
                        for crop_path in [crop_region(tmp_path, region.get("bbox"),
                                                          f"flag_{region.get('label', 'area')}")]
                        if crop_path
                    ]

                    st.session_state[result_key] = {
                        "extracted": extracted,
                        "ela_score": ela_score,
                        "result": result,
                        "row_id": row_id,
                        "field_crops": field_crops,
                        "flagged_crops": flagged_crops,
                    }
                except Exception as e:
                    import traceback
                    print("=== PIPELINE ERROR ===")
                    traceback.print_exc()
                    st.error(f"Pipeline error: {type(e).__name__}: {e}")
                    with st.expander("Full error details"):
                        st.code(traceback.format_exc())
                    st.stop()

        if result_key in st.session_state:
            state = st.session_state[result_key]
            extracted = state["extracted"]
            ela_score = state["ela_score"]
            result = state["result"]
            row_id = state["row_id"]
            field_crops = state.get("field_crops", [])
            flagged_crops = state.get("flagged_crops", [])

            with col_result:
                status_color = {"pass": "green", "flagged": "orange", "fail": "red"}
                st.markdown(f"### Status: :{status_color[result['status']]}[{result['status'].upper()}]")

                st.markdown("**Extracted fields:**")
                # One markdown call for the whole list, not one per field -
                # each st.write() is sent as its own incremental update over
                # Streamlit's websocket, and a burst of separate line-by-line
                # insertions arriving in quick succession was visibly
                # jittering the page as they populated.
                field_lines = "\n".join(
                    f"- {field}: `{data.get('value')}` (confidence: {data.get('confidence')})"
                    for field, data in extracted.get("fields", {}).items()
                )
                st.markdown(field_lines)

                if field_crops:
                    st.markdown("**Field close-ups** (zoomed in from the original document):")
                    crop_cols = st.columns(len(field_crops))
                    for col, (field, value, crop_path) in zip(crop_cols, field_crops):
                        with col:
                            st.image(crop_path, caption=f"{field}: {value}",
                                       use_container_width=True)

                st.markdown(f"**ELA fraud score:** {ela_score} / 100")

                if flagged_crops:
                    st.markdown("**Flagged regions** (AI visual inspection - side by side "
                                 "with the reasons below):")
                    flag_cols = st.columns(len(flagged_crops))
                    for col, (label, crop_path) in zip(flag_cols, flagged_crops):
                        with col:
                            st.image(crop_path, caption=label, use_container_width=True)

                st.markdown("**Reasons:**")
                st.markdown("\n".join(f"- {reason}" for reason in result["reasons"]))

            st.divider()
            st.markdown("### Your decision")
            decision_col1, decision_col2 = st.columns(2)
            if decision_col1.button("Approve", type="primary"):
                update_decision(row_id, "approved")
                st.success("Marked as approved and logged.")
            if decision_col2.button("Reject"):
                update_decision(row_id, "rejected")
                st.error("Marked as rejected and logged.")

# --- TAB 2: History and stats ---
with tab_history:
    reviews = get_all_reviews()

    if not reviews:
        st.info("No reviews logged yet - go run one in the first tab.")
    else:
        total = len(reviews)
        flagged = sum(1 for r in reviews if r["validation_status"] == "flagged")
        failed = sum(1 for r in reviews if r["validation_status"] == "fail")
        passed = sum(1 for r in reviews if r["validation_status"] == "pass")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total reviewed", total)
        c2.metric("Passed", passed)
        c3.metric("Flagged", flagged)
        c4.metric("Failed", failed)

        st.divider()
        for r in reviews:
            with st.expander(f"#{r['id']} - {r['filename']} - {r['validation_status']}"):
                st.write("Human decision:", r["human_decision"] or "not yet reviewed")
                st.write("Reasons:", r["validation_reasons"])
                st.write("Extracted:", r["extracted_json"])
                st.write("Logged at:", r["created_at"])