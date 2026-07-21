# AI-Assisted KYC Document Review Agent

A portfolio project demonstrating an AI agent pipeline for identity document review — combining LLM-based field extraction, LLM-based visual tamper inspection, and a classical image-forensics technique (Error Level Analysis), with a human always making the final approve/reject call.

**[Live demo on Hugging Face Spaces](https://huggingface.co/spaces/Junio7191/kyc-review-agent)**

## Why this project

Financial and fintech companies spend significant manual effort verifying identity documents (KYC). This project shows an end-to-end approach to assisting that process with AI — without removing the human from the loop, since fully-automated identity decisions carry real risk (false rejections, missed fraud, compliance exposure).

## What it does

- Upload an ID document (JPG, PNG, or PDF)
- **Extracts** structured fields — name, date of birth, ID number, expiry date — with a per-field confidence score, using a free, open vision-language model via [Hugging Face Inference Providers](https://huggingface.co/docs/inference-providers/index)
- **Visually inspects** the same image for tampering signs (font mismatches, misalignment, blur inconsistencies, background seams) in the *same* model call that does the extraction
- Shows **zoomed-in crops** of each extracted field and of any flagged suspicious region, cropped directly from the original image — so a reviewer can see exactly what the AI is talking about instead of interpreting an abstract heatmap
- Runs **Error Level Analysis (ELA)**, a classical forensic technique, as an independent second opinion that doesn't depend on the LLM at all
- Combines all signals into a **pass / flagged / fail** status with plain-English reasons
- Logs every review — including the human's final decision — to SQLite, so real precision/recall can be reported later instead of just demoing the happy path

## Design decisions worth mentioning in interviews

- **No auto-approval or auto-rejection on AI judgement alone.** A human always makes the final call — the system's job is to make that decision fast and well-informed, not to replace it.
- **Two independent fraud signals** (LLM visual inspection + ELA) rather than one, because each has different blind spots. The LLM can reason about context (an odd font, a weird phrasing) that ELA can't see; ELA can catch pixel-level edits that "look fine" to a glance but were re-compressed differently than the rest of the image.
- **Automatic provider fallback.** Hugging Face Inference Providers' model catalog changes over time, and individual providers occasionally reject or fail on specific images for reasons unrelated to the app. Rather than hardcoding one model/provider, the app tries a short list in order and fails over automatically.
- **Field crops instead of a heatmap.** An earlier version showed a full-image ELA heatmap, but a blotchy diff visualization is hard to act on. Cropping the exact region a field (or a flagged issue) lives in gives a reviewer something concrete to look at.
- **Escalation-only status logic** — a check can only make the outcome *more* severe, never less, avoiding a bug where one passing check accidentally overrides a real red flag.
- **Every review is logged**, including the human's final decision — this is what eventually lets you measure real accuracy instead of only demoing the happy path.

## Honest limitations

- ELA has known false positives on legitimately re-compressed/resized images — it's a *signal*, not an auto-reject rule.
- This is a proof of concept, not a production KYC system. Real systems also need liveness detection, sanctions-list/database checks, and regulatory compliance review.
- Extraction accuracy depends heavily on image quality, lighting, and which open model is currently serving the request.
- The field/region bounding boxes come from the vision model's own best-effort spatial grounding. General-purpose models (as opposed to ones specifically trained for object detection) aren't always precise at this, so crops are padded generously rather than pixel-tight, and a crop is simply skipped if the model's coordinates come back missing or unusable.

## Tech stack

- [Streamlit](https://streamlit.io/) — UI
- [Hugging Face Inference Providers](https://huggingface.co/docs/inference-providers/index) via `huggingface_hub` — free, open vision-language model access
- [Pillow](https://python-pillow.org/) / [NumPy](https://numpy.org/) — image processing and the ELA implementation
- [PyMuPDF](https://pymupdf.readthedocs.io/) — PDF → image conversion
- SQLite — review history storage

## Getting started

```bash
git clone <your-repo-url>
cd <your-repo-folder>
pip install -r requirements.txt
export HF_TOKEN=your_huggingface_token_here
streamlit run app.py
```

Get a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) — a free account is enough. Make sure the token has the **"Make calls to Inference Providers"** permission enabled (off by default on fine-grained tokens).

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

### Deploying to Hugging Face Spaces

1. Create a new Space with the **Streamlit** SDK.
2. Upload `app.py` and `requirements.txt`.
3. Add `HF_TOKEN` under **Settings → Repository secrets**.

## Testing it

You'll need sample ID images to test with. **Never use real people's actual identity documents.** Options:

- Search for publicly available "specimen ID" or "sample passport" template images used for testing purposes (many government and stock-photo sites publish blank specimen templates)
- Deliberately create an "edited" version (paste a different photo/date into a copy using any image editor) to see whether the AI inspection and ELA actually catch it

## Project structure

Everything lives in a single `app.py`, organized into clearly labeled sections:

| Section | Purpose |
|---|---|
| Database | SQLite logging of every review and the human's final decision |
| Fraud detection (ELA) | Classical, non-AI image-forensics check |
| Extraction + AI visual inspection | Calls the vision-language model (with provider fallback) to extract fields, locate them, and inspect for tampering |
| Validation / guardrails | Combines all signals into a final pass/flagged/fail decision |
| Streamlit UI | Upload → review → approve/reject, plus a history/stats dashboard |

## License

[MIT](LICENSE)
