---
title: Kyc Review Agent
emoji: 🕵️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
pinned: false
---

# AI-Assisted KYC Document Review Agent

A portfolio project demonstrating an AI agent pipeline for identity
document review — combining LLM-based extraction, LLM-based visual
fraud inspection, and a classical image-forensics technique (Error
Level Analysis), with a human always making the final approve/reject
call.

## Why this project

Financial and fintech companies spend significant manual effort
verifying identity documents (KYC). This project shows an end-to-end
approach to *assisting* that process with AI — without removing the
human from the loop, since fully-automated identity decisions carry
real risk (false rejections, missed fraud, compliance exposure).

## What it does

1. **Upload** an ID document image (driver's license, passport, etc.)
2. **Extract** structured fields (name, DOB, ID number, expiry) using
   a free, open vision-language model (Qwen2-VL via HuggingFace), with
   per-field confidence scores
3. **Visually inspect** the document for tampering signs — font
   mismatches, misalignment, blur inconsistencies, background seams —
   using the same model call
4. **Run Error Level Analysis (ELA)**, a classical forensic technique,
   as an independent second opinion that doesn't depend on the AI model at all
5. **Combine all signals** into a pass / flagged / fail status with
   plain-English reasons
6. **Log every decision** (AI recommendation + human's actual choice)
   to SQLite, so you can report real accuracy numbers later

## Setup

This project runs on a free, open vision-language model (Qwen2-VL) through
HuggingFace's Inference Providers — no paid API required. If you have HF PRO,
your monthly included Inference Provider credits cover this comfortably; a
handful of test images costs a tiny fraction of a cent. Free HF accounts also
have a small included quota that works for light testing.

**Running on HuggingFace Spaces:** just add `HF_TOKEN` as a repository
secret under Settings — the Space builds and runs automatically.

**Running locally:**
```bash
pip install -r requirements.txt
export HF_TOKEN=your_huggingface_token_here
streamlit run app.py
```

Get a token at https://huggingface.co/settings/tokens (free — just needs an
HF account; "Read" access is enough).

Then open the local URL Streamlit prints (usually http://localhost:8501).

## Project structure

| File | Purpose |
|---|---|
| `app.py` | Single-file app: SQLite logging, Error Level Analysis, model extraction + visual inspection, validation rules, and the Streamlit UI, all in one place with a full explanatory header comment |
| `requirements.txt` | Python dependencies |

## Testing it

You'll need sample ID images to test with. **Never use real people's
actual identity documents.** Options:
- Generate mock IDs (plain text on a template background, e.g. with PIL)
- Search for publicly available "sample ID template" images used for
  testing purposes (many government/company sites publish blank
  specimen templates)
- Deliberately create a few "edited" versions (e.g. paste a different
  photo/date into a copy using any image editor) to test whether the
  ELA and AI visual inspection actually catch it

## Design decisions worth mentioning in interviews

- **No auto-approval or auto-rejection on AI judgement alone.** A
  human always makes the final call — the system's job is to make
  that decision fast and well-informed, not to replace it.
- **Two independent fraud signals** (AI visual inspection + ELA)
  rather than one, because each has different blind spots. The AI
  can reason about context (odd font, weird phrasing) that ELA can't
  see; ELA can catch pixel-level edits that "look fine" to a glance.
- **Escalation-only status logic** — a flag can only make the outcome
  more severe, never less, avoiding a bug where one passing check
  accidentally overrides a real red flag.
- **Every review is logged**, including the human's final decision —
  this is what lets you eventually measure real precision/recall
  instead of only demoing the happy path.

## Honest limitations (also worth saying out loud in interviews)

- ELA has known false positives on legitimately re-compressed/resized
  images, which is exactly why it's a *signal*, not an auto-reject rule
- This is a proof of concept, not a production KYC system — real
  systems also need liveness detection, database/sanctions-list
  checks, and regulatory compliance review
- Extraction accuracy depends heavily on image quality, lighting, and
  which open model is used
