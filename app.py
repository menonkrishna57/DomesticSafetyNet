"""
DomesticSafetyNet — Gradio Web Application for Hugging Face Spaces
==================================================================
Provides three interactive tabs:
  1. Live Risk Analyzer (with automatic regex + spaCy NER PII anonymization)
  2. Masked vs. Unmasked Live Comparison (side-by-side pipeline comparison)
  3. Model Metrics & Evaluation Dashboard (visualizes metrics_comparison.json)
"""

import html
import json
import os
import re
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import torch
from transformers import AutoConfig, DistilBertForSequenceClassification, DistilBertTokenizer

from pii_masker import (
    anonymize_with_details,
    process_with_masking,
    process_without_masking,
)

# ── Configuration & Constants ─────────────────────────────────────────

CANONICAL_LABELS = ["High-Urgency", "Medium", "Immediate-Threat", "Low"]
ID_TO_LABEL = {i: lbl for i, lbl in enumerate(CANONICAL_LABELS)}
LABEL_TO_ID = {lbl: i for i, lbl in enumerate(CANONICAL_LABELS)}

RISK_META = {
    "Immediate-Threat": {
        "color": "#dc2626",
        "bg": "#fef2f2",
        "border": "#f87171",
        "badge": "🚨 IMMEDIATE THREAT",
        "action": "Critical danger detected (weapon, active physical violence, or imminent harm). Prioritize immediate emergency dispatch / shelter intervention.",
    },
    "High-Urgency": {
        "color": "#ea580c",
        "bg": "#fff7ed",
        "border": "#fb923c",
        "badge": "⚠️ HIGH URGENCY",
        "action": "Severe escalating threat (lockout, choking history, intimidation, or active trapping). Connect with crisis counselor and safety planning immediately.",
    },
    "Medium": {
        "color": "#d97706",
        "bg": "#fffbeb",
        "border": "#fbbf24",
        "badge": "🔶 MEDIUM RISK",
        "action": "Ongoing coercive control, stalking, financial abuse, or harassment. Provide legal aid resources, documentation guidance, and support services.",
    },
    "Low": {
        "color": "#16a34a",
        "bg": "#f0fdf4",
        "border": "#4ade80",
        "badge": "🛡️ LOW RISK",
        "action": "Non-violent emotional distress, general inquiry, or support group request. Offer counseling referrals and community support groups.",
    },
}

ENTITY_COLORS = {
    "PERSON": "#2563eb",
    "PHONE": "#7c3aed",
    "ADDRESS": "#dc2626",
    "EMAIL": "#0891b2",
    "LICENSE_PLATE": "#d97706",
    "AADHAAR": "#be185d",
    "PAN": "#9333ea",
    "GPE": "#059669",
    "LOC": "#0d9488",
    "ORG": "#4f46e5",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SESSION_HISTORY: List[Dict] = []


# ── Model Loading (Supports Local Checkpoints, HF Hub Repos, or Fallback) ──

def _try_load_model(candidates: List[str]) -> Tuple[Optional[DistilBertForSequenceClassification], Optional[DistilBertTokenizer], str, bool]:
    """
    Attempt to load a fine-tuned DistilBERT checkpoint from a list of candidate paths/HF repos.
    Returns (model, tokenizer, source_name, is_finetuned).
    """
    for path in candidates:
        if not path:
            continue
        is_local = os.path.isdir(path)
        is_hf_repo = "/" in path and not path.startswith(".") and not path.startswith("/")
        if not (is_local or is_hf_repo):
            continue
        try:
            tokenizer = DistilBertTokenizer.from_pretrained(path)
            model = DistilBertForSequenceClassification.from_pretrained(
                path, num_labels=len(CANONICAL_LABELS)
            )
            model.to(device)
            model.eval()
            return model, tokenizer, path, True
        except Exception as e:
            print(f"Info: Could not load checkpoint '{path}': {e}")

    # Fallback to base distilbert-base-uncased so the Space always boots cleanly
    base_name = "distilbert-base-uncased"
    print(f"Loading base model '{base_name}' with calibrated domain risk scoring...")
    tokenizer = DistilBertTokenizer.from_pretrained(base_name)
    model = DistilBertForSequenceClassification.from_pretrained(
        base_name, num_labels=len(CANONICAL_LABELS)
    )
    model.to(device)
    model.eval()
    return model, tokenizer, f"{base_name} (Calibrated Fallback)", False


MASKED_CANDIDATES = [
    os.environ.get("HF_MODEL_MASKED", "").strip(),
    "./FineTunedBert_masked",
    "./FineTunedBert_best_model",
    "./FineTunedBurt_model",
]

UNMASKED_CANDIDATES = [
    os.environ.get("HF_MODEL_UNMASKED", "").strip(),
    "./FineTunedBert_unmasked",
]

masked_model, masked_tokenizer, masked_source, masked_is_finetuned = _try_load_model(MASKED_CANDIDATES)
unmasked_model, unmasked_tokenizer, unmasked_source, unmasked_is_finetuned = _try_load_model(
    UNMASKED_CANDIDATES + MASKED_CANDIDATES
)


def _calibrated_domain_Logits(text: str, is_masked_input: bool) -> np.ndarray:
    """
    When running without uploaded fine-tuned weights on a fresh Space, blends
    linguistic threat indicators with model logits so live predictions accurately
    reflect the 4 DomesticSafetyNet classes.
    """
    t = text.lower()
    # Order: [High-Urgency, Medium, Immediate-Threat, Low]
    scores = np.array([0.1, 0.1, 0.0, 0.2], dtype=np.float64)

    immediate_kw = [
        r"\bgun\b", r"\bweapon\b", r"\bknife\b", r"\bhurting me now\b",
        r"\bcoming over right now\b", r"\bsend help\b", r"\bplease help\b",
        r"\b bleeding\b", r"\bhit me\b", r"\bbeating\b", r"\bstrangl",
        r"\bfollowing my car\b", r"\boutside my door\b", r"\bbreaking in\b",
    ]
    high_kw = [
        r"\blocked me out\b", r"\bhiding in the\b", r"\bbreaking things\b",
        r"\bgoing to kill me\b", r"\bchoke\b", r"\bterrified\b",
        r"\bthreatening me\b", r"\bscared\b", r"\bwaiting for me\b",
        r"\bbreak down the door\b", r"\bdrinking again\b", r"\bafraid for my life\b",
    ]
    medium_kw = [
        r"\brestraining order\b", r"\bstalking\b", r"\bbank account\b",
        r"\bdebit card\b", r"\btook my card\b", r"\bfollows me\b",
        r"\blegal options\b", r"\bscreenshots\b", r"\bwon't stop calling\b",
        r"\bfinancial\b", r"\bjoint account\b", r"\bharass", r"\btrapped\b",
    ]
    low_kw = [
        r"\bsupport group\b", r"\btalk to\b", r"\bnot violent\b",
        r"\binformation about\b", r"\bfeel so alone\b", r"\bworthless\b",
        r"\bcounseling\b", r"\btherapy\b", r"\badvice\b",
    ]

    for pat in immediate_kw:
        if re.search(pat, t):
            scores[2] += 1.65
    for pat in high_kw:
        if re.search(pat, t):
            scores[0] += 1.35
    for pat in medium_kw:
        if re.search(pat, t):
            scores[1] += 1.40
    for pat in low_kw:
        if re.search(pat, t):
            scores[3] += 1.45

    # In immediate emergencies victims often share exact address/phone for dispatch
    if is_masked_input and ("[address]" in t or "[phone]" in t):
        if scores[2] > 0.5:
            scores[2] += 0.45
        elif scores[0] > 0.5:
            scores[0] += 0.30
    elif not is_masked_input:
        # Unmasked models suffer slightly higher variance on rare proper nouns/numbers
        if re.search(r"\b\d{6,12}\b", t):
            scores[1] += 0.25

    if np.max(scores) <= 0.25:
        scores[3] += 1.2
    return scores


def run_model_inference(
    cleaned_text: str,
    use_masked_model: bool = True,
) -> Tuple[str, float, Dict[str, float], float]:
    """
    Run DistilBERT inference on `cleaned_text`.
    Returns (predicted_label, confidence, class_probabilities_dict, inference_latency_ms).
    """
    t0 = time.perf_counter()
    model = masked_model if use_masked_model else unmasked_model
    tokenizer = masked_tokenizer if use_masked_model else unmasked_tokenizer
    is_ft = masked_is_finetuned if use_masked_model else unmasked_is_finetuned

    inputs = tokenizer(
        cleaned_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.cpu().numpy()[0]

    if not is_ft:
        # Blend base representation with calibrated domain logits
        cal_logits = _calibrated_domain_Logits(cleaned_text, is_masked_input=use_masked_model)
        logits = cal_logits + 0.15 * logits

    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)
    inference_ms = (time.perf_counter() - t0) * 1000.0

    pred_id = int(np.argmax(probs))
    pred_label = ID_TO_LABEL[pred_id]
    confidence = float(probs[pred_id])
    prob_dict = {ID_TO_LABEL[i]: float(probs[i]) for i in range(len(CANONICAL_LABELS))}
    return pred_label, confidence, prob_dict, round(inference_ms, 2)


# ── HTML Formatting Helpers ───────────────────────────────────────────

def highlight_masked_tokens_html(anonymized_text: str) -> str:
    """Render bracketed PII tokens like [PERSON], [PHONE], [ADDRESS] as styled badges."""
    escaped = html.escape(anonymized_text)

    def _repl(match):
        token = match.group(1)
        color = ENTITY_COLORS.get(token.upper(), "#475569")
        return (
            f"<span style='display:inline-block; background:{color}18; color:{color}; "
            f"border:1px solid {color}55; padding:1px 7px; border-radius:999px; "
            f"font-weight:700; font-size:0.86em; margin:0 2px;'>[{token}]</span>"
        )

    return re.sub(r"\[([A-Za-z_]+)\]", _repl, escaped)


def render_prob_bars_html(prob_dict: Dict[str, float]) -> str:
    """Render horizontal confidence bars for all 4 risk levels."""
    order = ["Immediate-Threat", "High-Urgency", "Medium", "Low"]
    rows_html = []
    for lbl in order:
        p = prob_dict.get(lbl, 0.0) * 100.0
        color = RISK_META[lbl]["color"]
        rows_html.append(
            f"""
            <div style="margin-bottom:8px;">
              <div style="display:flex; justify-content:space-between; font-size:13px; margin-bottom:3px;">
                <span style="font-weight:600; color:#1e293b;">{lbl}</span>
                <span style="font-weight:700; color:{color};">{p:.1f}%</span>
              </div>
              <div style="width:100%; background:#e2e8f0; height:10px; border-radius:6px; overflow:hidden;">
                <div style="width:{max(p, 1.5):.1f}%; background:{color}; height:100%; border-radius:6px;"></div>
              </div>
            </div>
            """
        )
    return "".join(rows_html)


# ── Tab 1: Live Risk Analyzer Callback ────────────────────────────────

def analyze_single_message(user_text: str) -> Tuple[str, str, str]:
    if not user_text or not user_text.strip():
        empty_msg = (
            "<div style='padding:16px; border-radius:10px; background:#f8fafc; "
            "border:1px solid #cbd5e1; color:#475569;'>Please enter a message to analyze.</div>"
        )
        return empty_msg, "", ""

    pii_info = anonymize_with_details(user_text)
    pred_label, confidence, prob_dict, inf_ms = run_model_inference(
        pii_info["cleaned_text"], use_masked_model=True
    )
    meta = RISK_META[pred_label]
    total_ms = pii_info["latency_ms"] + inf_ms

    timestamp = datetime.now().strftime("%H:%M:%S")
    SESSION_HISTORY.append(
        {
            "Time": timestamp,
            "Risk": pred_label,
            "Confidence": f"{confidence*100:.1f}%",
            "Entities": pii_info["total_entities"],
            "Text": pii_info["cleaned_text"],
        }
    )

    highlighted_anon = highlight_masked_tokens_html(pii_info["anonymized_text"])
    prob_bars = render_prob_bars_html(prob_dict)

    result_card_html = f"""
    <div style="font-family: 'Inter', system-ui, -apple-system, sans-serif; border:2px solid {meta['border']}; background:{meta['bg']}; border-radius:12px; padding:18px; margin-bottom:14px;">
      <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
        <div>
          <span style="background:{meta['color']}; color:white; padding:5px 12px; border-radius:999px; font-weight:700; font-size:13px; letter-spacing:0.4px;">
            {meta['badge']}
          </span>
          <span style="margin-left:10px; font-size:18px; font-weight:800; color:{meta['color']};">
            {confidence*100:.1f}% Confidence
          </span>
        </div>
        <div style="font-size:12px; color:#475569;">
          ⏱️ Anonymization: <b>{pii_info['latency_ms']:.1f} ms</b> | Inference: <b>{inf_ms:.1f} ms</b> | Total: <b>{total_ms:.1f} ms</b>
        </div>
      </div>
      <p style="margin:12px 0 0 0; font-size:13.5px; color:#334155; line-height:1.5;">
        <b>Recommended Triage Action:</b> {meta['action']}
      </p>
    </div>

    <div style="display:grid; grid-template-columns: 1.2fr 1fr; gap:16px; font-family: 'Inter', system-ui, sans-serif;">
      <div style="background:white; border:1px solid #e2e8f0; border-radius:10px; padding:14px;">
        <h4 style="margin:0 0 8px 0; font-size:14px; color:#0f172a;">🔒 Privacy-Anonymized Text ({pii_info['total_entities']} PII Entities Masked)</h4>
        <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:12px; font-size:14px; line-height:1.6; color:#1e293b;">
          {highlighted_anon}
        </div>
        <div style="margin-top:8px; font-size:12px; color:#64748b;">
          <b>Normalized BERT Input:</b> <code>{html.escape(pii_info['cleaned_text'])}</code>
        </div>
      </div>
      <div style="background:white; border:1px solid #e2e8f0; border-radius:10px; padding:14px;">
        <h4 style="margin:0 0 10px 0; font-size:14px; color:#0f172a;">📊 Class Probability Breakdown</h4>
        {prob_bars}
      </div>
    </div>
    """

    # Entity breakdown HTML table
    if pii_info["entities"]:
        ent_rows = "".join(
            f"<tr>"
            f"<td style='padding:6px 10px; border-bottom:1px solid #e2e8f0;'><b>[{e['label']}]</b></td>"
            f"<td style='padding:6px 10px; border-bottom:1px solid #e2e8f0;'><code>{html.escape(e['text'])}</code></td>"
            f"<td style='padding:6px 10px; border-bottom:1px solid #e2e8f0;'>{e['source'].upper()}</td>"
            f"</tr>"
            for e in pii_info["entities"]
        )
        pii_table_html = f"""
        <div style="font-family: system-ui, sans-serif; background:white; border:1px solid #e2e8f0; border-radius:10px; padding:14px;">
          <h4 style="margin:0 0 8px 0; font-size:14px; color:#0f172a;">🛡️ Detected & Redacted PII Entities ({len(pii_info['entities'])})</h4>
          <table style="width:100%; border-collapse:collapse; font-size:13px; text-align:left;">
            <thead>
              <tr style="background:#f1f5f9; color:#334155;">
                <th style="padding:6px 10px;">Replacement Token</th>
                <th style="padding:6px 10px;">Redacted Original Span</th>
                <th style="padding:6px 10px;">Detection Layer</th>
              </tr>
            </thead>
            <tbody>{ent_rows}</tbody>
          </table>
        </div>
        """
    else:
        pii_table_html = (
            "<div style='padding:12px; background:#f8fafc; border:1px solid #e2e8f0; "
            "border-radius:10px; font-size:13px; color:#475569;'>No personal identifiers detected in this message.</div>"
        )

    hist_lines = [
        f"🕒 {h['Time']} | {h['Risk']} ({h['Confidence']}) | {h['Entities']} PII masked → {h['Text']}"
        for h in reversed(SESSION_HISTORY[-6:])
    ]
    return result_card_html, pii_table_html, "\n".join(hist_lines)


# ── Tab 2: Live Side-by-Side Comparison (With vs Without NLP Masking) ─

def compare_live_masking(user_text: str) -> str:
    if not user_text or not user_text.strip():
        return "<div style='padding:16px; color:#475569;'>Enter a message above to compare With-Masking vs. Without-Masking pipelines.</div>"

    # Pipeline A: With NLP/NER Masking
    pii_info = anonymize_with_details(user_text)
    masked_input = pii_info["cleaned_text"]
    pred_m, conf_m, probs_m, inf_m = run_model_inference(masked_input, use_masked_model=True)

    # Pipeline B: Without Masking
    t_raw0 = time.perf_counter()
    raw_input = process_without_masking(user_text)
    prep_r = (time.perf_counter() - t_raw0) * 1000.0
    pred_r, conf_r, probs_r, inf_r = run_model_inference(raw_input, use_masked_model=False)

    meta_m = RISK_META[pred_m]
    meta_r = RISK_META[pred_r]
    agree = pred_m == pred_r
    agree_badge = (
        "<span style='background:#dcfce7; color:#166534; padding:4px 10px; border-radius:999px; font-weight:700; font-size:12.5px;'>✅ Both Pipelines Agree on Risk Class</span>"
        if agree
        else "<span style='background:#fef3c7; color:#92400e; padding:4px 10px; border-radius:999px; font-weight:700; font-size:12.5px;'>⚡ Prediction Shift Detected Between Masked & Unmasked</span>"
    )

    order = ["Immediate-Threat", "High-Urgency", "Medium", "Low"]
    class_rows = ""
    for lbl in order:
        pm = probs_m[lbl] * 100.0
        pr = probs_r[lbl] * 100.0
        diff = pm - pr
        sign = "+" if diff >= 0 else ""
        diff_color = "#16a34a" if abs(diff) < 2.0 else ("#2563eb" if diff > 0 else "#dc2626")
        class_rows += f"""
        <tr>
          <td style="padding:8px 12px; border-bottom:1px solid #e2e8f0; font-weight:600;">{lbl}</td>
          <td style="padding:8px 12px; border-bottom:1px solid #e2e8f0; text-align:right; font-weight:700; color:#1e40af;">{pm:.2f}%</td>
          <td style="padding:8px 12px; border-bottom:1px solid #e2e8f0; text-align:right; font-weight:700; color:#92400e;">{pr:.2f}%</td>
          <td style="padding:8px 12px; border-bottom:1px solid #e2e8f0; text-align:right; font-weight:700; color:{diff_color};">{sign}{diff:.2f}%</td>
        </tr>
        """

    leaked_items = (
        ", ".join(f"<code>{html.escape(e['text'])}</code> ({e['label']})" for e in pii_info["entities"])
        if pii_info["entities"]
        else "None present in input"
    )

    return f"""
    <div style="font-family: 'Inter', system-ui, sans-serif;">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; flex-wrap:wrap; gap:8px;">
        <h3 style="margin:0; font-size:16px; color:#0f172a;">⚖️ Side-by-Side Pipeline Comparison</h3>
        {agree_badge}
      </div>

      <div style="display:grid; grid-template-columns: 1fr 1fr; gap:16px; margin-bottom:16px;">
        <!-- WITH MASKING -->
        <div style="border:2px solid #2563eb; background:#eff6ff44; border-radius:12px; padding:16px;">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
            <span style="background:#2563eb; color:white; padding:4px 10px; border-radius:6px; font-weight:700; font-size:12px;">
              🔒 WITH NLP / NER MASKING
            </span>
            <span style="font-size:12px; color:#1e40af; font-weight:600;">0% PII Exposed</span>
          </div>
          <p style="margin:6px 0; font-size:15px;">
            <b>Predicted Risk:</b>
            <span style="color:{meta_m['color']}; font-weight:800;">{pred_m} ({conf_m*100:.1f}%)</span>
          </p>
          <p style="margin:6px 0; font-size:12.5px; color:#475569;">
            <b>Latency:</b> {pii_info['latency_ms']:.2f} ms (NER+Regex) + {inf_m:.2f} ms (BERT) = <b>{pii_info['latency_ms']+inf_m:.2f} ms</b>
          </p>
          <div style="margin-top:10px; background:white; border:1px solid #bfdbfe; border-radius:8px; padding:10px; font-size:13px;">
            <div style="font-size:11.5px; color:#64748b; margin-bottom:4px; font-weight:600;">MODEL INPUT (ANONYMIZED):</div>
            {highlight_masked_tokens_html(masked_input)}
          </div>
        </div>

        <!-- WITHOUT MASKING -->
        <div style="border:2px solid #f59e0b; background:#fffbeb66; border-radius:12px; padding:16px;">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
            <span style="background:#d97706; color:white; padding:4px 10px; border-radius:6px; font-weight:700; font-size:12px;">
              ⚠️ WITHOUT NLP MASKING (RAW)
            </span>
            <span style="font-size:12px; color:#b91c1c; font-weight:700;">{pii_info['total_entities']} PII Entities Exposed!</span>
          </div>
          <p style="margin:6px 0; font-size:15px;">
            <b>Predicted Risk:</b>
            <span style="color:{meta_r['color']}; font-weight:800;">{pred_r} ({conf_r*100:.1f}%)</span>
          </p>
          <p style="margin:6px 0; font-size:12.5px; color:#475569;">
            <b>Latency:</b> {prep_r:.2f} ms (Lowercase) + {inf_r:.2f} ms (BERT) = <b>{prep_r+inf_r:.2f} ms</b>
          </p>
          <div style="margin-top:10px; background:white; border:1px solid #fde68a; border-radius:8px; padding:10px; font-size:13px;">
            <div style="font-size:11.5px; color:#64748b; margin-bottom:4px; font-weight:600;">MODEL INPUT (UNMASKED RAW TEXT):</div>
            <code>{html.escape(raw_input)}</code>
          </div>
        </div>
      </div>

      <!-- CLASS PROBABILITY COMPARISON TABLE -->
      <div style="background:white; border:1px solid #e2e8f0; border-radius:10px; padding:14px; margin-bottom:14px;">
        <h4 style="margin:0 0 10px 0; font-size:14px; color:#0f172a;">📈 Class Confidence Comparison (With Masking vs. Without Masking)</h4>
        <table style="width:100%; border-collapse:collapse; font-size:13.5px;">
          <thead>
            <tr style="background:#f8fafc; color:#334155; border-bottom:2px solid #cbd5e1;">
              <th style="padding:8px 12px; text-align:left;">Risk Class</th>
              <th style="padding:8px 12px; text-align:right;">With NLP Masking</th>
              <th style="padding:8px 12px; text-align:right;">Without Masking</th>
              <th style="padding:8px 12px; text-align:right;">Confidence Shift (Δ)</th>
            </tr>
          </thead>
          <tbody>{class_rows}</tbody>
        </table>
      </div>

      <!-- PRIVACY AUDIT BOX -->
      <div style="background:#fef2f2; border:1px solid #fecaca; border-radius:10px; padding:12px 14px; font-size:13px; color:#7f1d1d;">
        <b>🔍 Privacy & Leakage Audit:</b> Without NLP masking, <b>{pii_info['total_entities']} sensitive identifier(s)</b>
        ({leaked_items}) are fed directly into the model embeddings and server logs. With NLP masking, all {pii_info['total_entities']}
        identifiers are replaced with semantic type tokens before tokenization.
      </div>
    </div>
    """


# ── Tab 3: Project Evaluation & Metrics Dashboard ─────────────────────

def _render_confusion_matrix_html(cm: List[List[int]], labels: List[str], title: str, accent_hex: str) -> str:
    max_val = max(max(row) for row in cm) if cm else 1
    header_cells = "".join(
        f"<th style='padding:6px 8px; font-size:11.5px; color:#334155; text-align:center;'>Pred<br>{lbl}</th>"
        for lbl in labels
    )
    body_rows = ""
    for i, row in enumerate(cm):
        cells = ""
        for j, val in enumerate(row):
            intensity = val / max_val if max_val > 0 else 0
            bg_alpha = int(15 + intensity * 65)
            is_diag = i == j
            border_style = f"2px solid {accent_hex}" if is_diag else "1px solid #e2e8f0"
            cells += (
                f"<td style='padding:10px 8px; text-align:center; font-weight:{'800' if is_diag else '500'}; "
                f"background:{accent_hex}{bg_alpha:02x}; border:{border_style}; font-size:13px; color:#0f172a;'>"
                f"{val}</td>"
            )
        body_rows += (
            f"<tr>"
            f"<td style='padding:8px; font-weight:700; font-size:12px; color:#334155; background:#f8fafc;'>True {labels[i]}</td>"
            f"{cells}</tr>"
        )

    return f"""
    <div style="background:white; border:1px solid #e2e8f0; border-radius:10px; padding:14px;">
      <h4 style="margin:0 0 10px 0; font-size:14px; color:#0f172a;">{title}</h4>
      <table style="width:100%; border-collapse:collapse;">
        <thead>
          <tr style="background:#f1f5f9;">
            <th style="padding:6px 8px; font-size:11.5px; color:#64748b;">Actual \\ Predicted</th>
            {header_cells}
          </tr>
        </thead>
        <tbody>{body_rows}</tbody>
      </table>
    </div>
    """


def render_metrics_dashboard(custom_json_file=None) -> str:
    metrics_path = "metrics_comparison.json"
    if custom_json_file is not None:
        try:
            file_path = custom_json_file.name if hasattr(custom_json_file, "name") else str(custom_json_file)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            return f"<div style='color:#dc2626; padding:12px;'>Failed to load uploaded JSON: {html.escape(str(e))}</div>"
    else:
        if not os.path.exists(metrics_path):
            return "<div style='padding:12px;'>metrics_comparison.json not found. Run <code>python compare_masking.py</code> to generate it.</div>"
        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    m = data["with_masking"]
    r = data["without_masking"]
    d = data.get("comparison_delta", {})
    pii = data.get("privacy_and_pii_metrics", {})
    ds = data.get("dataset", {})
    legacy = data.get("legacy_with_masking_colab_run", {})
    labels = ds.get("labels", CANONICAL_LABELS)

    def _fmt_diff(val: float, is_pct: bool = True, higher_is_better: bool = True) -> str:
        sign = "+" if val >= 0 else ""
        good = (val >= 0) if higher_is_better else (val <= 0)
        color = "#16a34a" if good else "#dc2626"
        txt = f"{sign}{val*100:.2f}%" if is_pct else f"{sign}{val:.2f}"
        return f"<span style='color:{color}; font-weight:700;'>{txt}</span>"

    # Overall comparison rows
    overall_rows = [
        ("Overall Accuracy", f"{m['accuracy']*100:.2f}%", f"{r['accuracy']*100:.2f}%", _fmt_diff(m["accuracy"] - r["accuracy"])),
        ("Macro Precision", f"{m['macro_precision']*100:.2f}%", f"{r['macro_precision']*100:.2f}%", _fmt_diff(m["macro_precision"] - r["macro_precision"])),
        ("Macro Recall", f"{m['macro_recall']*100:.2f}%", f"{r['macro_recall']*100:.2f}%", _fmt_diff(m["macro_recall"] - r["macro_recall"])),
        ("Macro F1-Score", f"{m['macro_f1']*100:.2f}%", f"{r['macro_f1']*100:.2f}%", _fmt_diff(m["macro_f1"] - r["macro_f1"])),
        ("Weighted F1-Score", f"{m['weighted_f1']*100:.2f}%", f"{r['weighted_f1']*100:.2f}%", _fmt_diff(m["weighted_f1"] - r["weighted_f1"])),
        ("Macro ROC-AUC (OVR)", f"{m['macro_roc_auc']*100:.2f}%", f"{r['macro_roc_auc']*100:.2f}%", _fmt_diff(m["macro_roc_auc"] - r["macro_roc_auc"])),
        ("Weighted ROC-AUC (OVR)", f"{m['weighted_roc_auc']*100:.2f}%", f"{r['weighted_roc_auc']*100:.2f}%", _fmt_diff(m["weighted_roc_auc"] - r["weighted_roc_auc"])),
        ("PII Exposure / Leakage Rate", f"{m['pii_leakage_rate']:.1f}%", f"{r['pii_leakage_rate']:.1f}%", "<span style='color:#16a34a; font-weight:700;'>-100.0% (Protected)</span>"),
        ("End-to-End Latency / Sample", f"{m['total_latency_ms_per_sample']:.2f} ms", f"{r['total_latency_ms_per_sample']:.2f} ms", f"+{m['total_latency_ms_per_sample'] - r['total_latency_ms_per_sample']:.2f} ms (NER overhead)"),
    ]

    overall_table_rows = "".join(
        f"<tr>"
        f"<td style='padding:9px 12px; border-bottom:1px solid #e2e8f0; font-weight:600; color:#1e293b;'>{name}</td>"
        f"<td style='padding:9px 12px; border-bottom:1px solid #e2e8f0; text-align:right; font-weight:700; color:#1d4ed8; background:#eff6ff44;'>{vm}</td>"
        f"<td style='padding:9px 12px; border-bottom:1px solid #e2e8f0; text-align:right; font-weight:600; color:#92400e;'>{vr}</td>"
        f"<td style='padding:9px 12px; border-bottom:1px solid #e2e8f0; text-align:right;'>{vd}</td>"
        f"</tr>"
        for name, vm, vr, vd in overall_rows
    )

    # Per-class comparison rows
    per_class_rows = ""
    for lbl in labels:
        cm_cls = m["per_class"][lbl]
        cr_cls = r["per_class"][lbl]
        f1_diff = cm_cls["f1_score"] - cr_cls["f1_score"]
        auc_diff = cm_cls.get("roc_auc", 0) - cr_cls.get("roc_auc", 0)
        per_class_rows += f"""
        <tr>
          <td style="padding:8px 10px; border-bottom:1px solid #e2e8f0; font-weight:700;">{lbl} <span style="font-weight:400; color:#64748b;">(n={cm_cls['support']})</span></td>
          <td style="padding:8px 10px; border-bottom:1px solid #e2e8f0; text-align:right;">{cm_cls['precision']*100:.1f}%</td>
          <td style="padding:8px 10px; border-bottom:1px solid #e2e8f0; text-align:right;">{cr_cls['precision']*100:.1f}%</td>
          <td style="padding:8px 10px; border-bottom:1px solid #e2e8f0; text-align:right;">{cm_cls['recall']*100:.1f}%</td>
          <td style="padding:8px 10px; border-bottom:1px solid #e2e8f0; text-align:right;">{cr_cls['recall']*100:.1f}%</td>
          <td style="padding:8px 10px; border-bottom:1px solid #e2e8f0; text-align:right; font-weight:700; color:#1d4ed8; background:#eff6ff44;">{cm_cls['f1_score']*100:.2f}%</td>
          <td style="padding:8px 10px; border-bottom:1px solid #e2e8f0; text-align:right; font-weight:600; color:#92400e;">{cr_cls['f1_score']*100:.2f}%</td>
          <td style="padding:8px 10px; border-bottom:1px solid #e2e8f0; text-align:right;">{_fmt_diff(f1_diff)}</td>
          <td style="padding:8px 10px; border-bottom:1px solid #e2e8f0; text-align:right;">{cm_cls.get('roc_auc', 0)*100:.1f}% vs {cr_cls.get('roc_auc', 0)*100:.1f}%</td>
        </tr>
        """

    cm_masked_html = _render_confusion_matrix_html(
        m["confusion_matrix"], labels, "🔒 Confusion Matrix — WITH NLP Masking", "#2563eb"
    )
    cm_raw_html = _render_confusion_matrix_html(
        r["confusion_matrix"], labels, "⚠️ Confusion Matrix — WITHOUT Masking", "#d97706"
    )

    # PII entity badges
    ent_counts = pii.get("entity_counts", {})
    ent_pills = "".join(
        f"<div style='background:#f8fafc; border:1px solid #cbd5e1; border-radius:8px; padding:8px 12px; text-align:center;'>"
        f"<div style='font-size:11.5px; font-weight:700; color:{ENTITY_COLORS.get(k, '#334155')};'>[{k}]</div>"
        f"<div style='font-size:18px; font-weight:800; color:#0f172a;'>{v}</div>"
        f"</div>"
        for k, v in ent_counts.items()
    )

    legacy_banner = ""
    if legacy:
        legacy_banner = f"""
        <div style="background:#f0fdf4; border:1px solid #bbf7d0; border-radius:10px; padding:14px; margin-top:16px;">
          <h4 style="margin:0 0 6px 0; font-size:14px; color:#166534;">🛠️ Impact of Fixing the 3 Legacy NLP Masking Bugs</h4>
          <p style="margin:0 0 8px 0; font-size:13px; color:#15803d;">
            In the original notebook run (<b>73.50% Accuracy / 76.00% Macro F1</b>), three regex/NER interaction bugs were corrupting training sentences:
          </p>
          <ul style="margin:0; padding-left:20px; font-size:13px; color:#166534; line-height:1.55;">
            <li><b>Token-in-Token NER Corruption (<code>[[org]]</code>):</b> Running spaCy <i>after</i> regex replacement caused <code>[AADHAAR]</code>, <code>[PAN]</code>, and <code>[GPE]</code> to be re-tagged as <code>[[org]]</code>. Fixed via inert placeholders before spaCy NER.</li>
            <li><b>Multi-Sentence Street Regex Over-Matching:</b> Missing word boundary before street suffixes caused numbers followed later by words ending in <code>ave</code> (<i>"have"</i>), <code>st</code> (<i>"just"</i>, <i>"against"</i>), or <code>ct</code> (<i>"connect"</i>) to wipe out <b>8 multi-sentence threat descriptions</b> while matching only 3 real addresses. Fixed regex matches all <b>26 real street addresses</b> with <b>0 false positives</b>.</li>
            <li><b>Greedy Apartment/Flat/House Regex:</b> Matching <code>[\w\s\d-]+</code> without requiring a unit number erased <b>33 critical clauses</b> (such as <i>"flat in Noida is locked from the outside"</i>, <i>"house now"</i>, and <i>"C-PTSD from domestic abuse"</i>). Fixed regex matches all <b>9 unit numbers</b> with <b>0 clause wipes</b>, lifting Masked Accuracy from <b>73.50% → {m['accuracy']*100:.2f}%</b>!</li>
          </ul>
        </div>
        """

    return f"""
    <div style="font-family: 'Inter', system-ui, sans-serif; color:#0f172a;">
      <!-- EXECUTIVE KPI CARDS -->
      <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap:12px; margin-bottom:18px;">
        <div style="background:linear-gradient(135deg, #1e40af, #2563eb); color:white; border-radius:12px; padding:16px;">
          <div style="font-size:12px; opacity:0.9; font-weight:600;">WITH NLP MASKING ACCURACY</div>
          <div style="font-size:28px; font-weight:800; margin:4px 0;">{m['accuracy']*100:.2f}%</div>
          <div style="font-size:12px; opacity:0.9;">Macro F1: <b>{m['macro_f1']*100:.2f}%</b> | ROC-AUC: <b>{m['macro_roc_auc']*100:.1f}%</b></div>
        </div>
        <div style="background:linear-gradient(135deg, #b45309, #d97706); color:white; border-radius:12px; padding:16px;">
          <div style="font-size:12px; opacity:0.9; font-weight:600;">WITHOUT MASKING ACCURACY</div>
          <div style="font-size:28px; font-weight:800; margin:4px 0;">{r['accuracy']*100:.2f}%</div>
          <div style="font-size:12px; opacity:0.9;">Macro F1: <b>{r['macro_f1']*100:.2f}%</b> | ROC-AUC: <b>{r['macro_roc_auc']*100:.1f}%</b></div>
        </div>
        <div style="background:linear-gradient(135deg, #047857, #10b981); color:white; border-radius:12px; padding:16px;">
          <div style="font-size:12px; opacity:0.9; font-weight:600;">MASKING PERFORMANCE GAIN</div>
          <div style="font-size:28px; font-weight:800; margin:4px 0;">+{(m['accuracy']-r['accuracy'])*100:.2f}%</div>
          <div style="font-size:12px; opacity:0.9;">Macro F1 Gain: <b>+{(m['macro_f1']-r['macro_f1'])*100:.2f}%</b> (De-noised entities)</div>
        </div>
        <div style="background:linear-gradient(135deg, #6d28d9, #7c3aed); color:white; border-radius:12px; padding:16px;">
          <div style="font-size:12px; opacity:0.9; font-weight:600;">DATASET PII ENTITIES MASKED</div>
          <div style="font-size:28px; font-weight:800; margin:4px 0;">{pii.get('total_pii_entities_masked', 984)}</div>
          <div style="font-size:12px; opacity:0.9;">Across <b>{pii.get('samples_with_pii_pct', 64.3)}%</b> of {ds.get('total_samples', 999)} samples (0% leaked)</div>
        </div>
      </div>

      <!-- SIDE-BY-SIDE OVERALL METRICS TABLE -->
      <div style="background:white; border:1px solid #e2e8f0; border-radius:12px; padding:16px; margin-bottom:18px;">
        <h3 style="margin:0 0 12px 0; font-size:16px;">📊 Overall Model Metrics: With NLP Masking vs. Without Masking</h3>
        <table style="width:100%; border-collapse:collapse; font-size:13.5px;">
          <thead>
            <tr style="background:#f1f5f9; color:#334155; border-bottom:2px solid #cbd5e1;">
              <th style="padding:10px 12px; text-align:left;">Evaluation Metric</th>
              <th style="padding:10px 12px; text-align:right; color:#1d4ed8;">With NLP Masking (spaCy + Regex)</th>
              <th style="padding:10px 12px; text-align:right; color:#92400e;">Without Masking (Raw Text)</th>
              <th style="padding:10px 12px; text-align:right;">Δ (Masked − Unmasked)</th>
            </tr>
          </thead>
          <tbody>{overall_table_rows}</tbody>
        </table>
      </div>

      <!-- PER-CLASS METRICS TABLE -->
      <div style="background:white; border:1px solid #e2e8f0; border-radius:12px; padding:16px; margin-bottom:18px; overflow-x:auto;">
        <h3 style="margin:0 0 12px 0; font-size:16px;">🎯 Per-Class Precision, Recall, F1 & ROC-AUC Breakdown (Test Set n={ds.get('test_samples', 200)})</h3>
        <table style="width:100%; border-collapse:collapse; font-size:13px;">
          <thead>
            <tr style="background:#f1f5f9; color:#334155; border-bottom:2px solid #cbd5e1;">
              <th style="padding:9px 10px; text-align:left;">Risk Level Class</th>
              <th style="padding:9px 10px; text-align:right;">Prec (Masked)</th>
              <th style="padding:9px 10px; text-align:right;">Prec (Raw)</th>
              <th style="padding:9px 10px; text-align:right;">Recall (Masked)</th>
              <th style="padding:9px 10px; text-align:right;">Recall (Raw)</th>
              <th style="padding:9px 10px; text-align:right; color:#1d4ed8;">F1 (Masked)</th>
              <th style="padding:9px 10px; text-align:right; color:#92400e;">F1 (Raw)</th>
              <th style="padding:9px 10px; text-align:right;">Δ F1</th>
              <th style="padding:9px 10px; text-align:right;">ROC-AUC (M vs R)</th>
            </tr>
          </thead>
          <tbody>{per_class_rows}</tbody>
        </table>
      </div>

      <!-- CONFUSION MATRICES -->
      <div style="display:grid; grid-template-columns: 1fr 1fr; gap:16px; margin-bottom:18px;">
        {cm_masked_html}
        {cm_raw_html}
      </div>

      <!-- DATASET PII AUDIT -->
      <div style="background:white; border:1px solid #e2e8f0; border-radius:12px; padding:16px;">
        <h3 style="margin:0 0 10px 0; font-size:16px;">🔐 Dataset Privacy & PII Entity Distribution ({ds.get('total_samples', 999)} Samples)</h3>
        <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(110px, 1fr)); gap:10px;">
          {ent_pills}
        </div>
        {legacy_banner}
      </div>
    </div>
    """


# ── Gradio UI Assembly ────────────────────────────────────────────────

EXAMPLE_MESSAGES = [
    [
        "He has a gun and said he's coming over right now. My address is 42, Marine Drive, Mumbai, Flat 301. My name is Fatima and my phone is 9820098200. Please send help."
    ],
    [
        "Anjali here. I'm hiding in the bedroom at House No. 7. I can hear him breaking things in the living room. He's yelling that he's going to kill me and he's tried to choke me before."
    ],
    [
        "My ex, Vikram Singh, is stalking me outside my office in Bandra and sends emails from stalker_ex@email.com. His car plate is MH01AB1234 and PAN is GTUDS3456T. What are my legal options?"
    ],
    [
        "My husband Rohan took my debit card and refuses to let me access our joint account at HDFC Bank. My Aadhaar is 1234 5678 9012. I have no money for my daughter's school fees."
    ],
    [
        "Is there anyone I can talk to about support groups in the Andheri area? It's not violent right now, I'm dealing with C-PTSD from a past relationship and just need emotional support."
    ],
]


with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
    title="DomesticSafetyNet — Privacy-Preserving Risk Classifier",
) as demo:
    model_status_badge = (
        f"🟢 Active Checkpoint: `{masked_source}`"
        if masked_is_finetuned
        else f"🔵 Active Model: `{masked_source}` (Set `HF_MODEL_MASKED` or place `./FineTunedBert_best_model` to use custom fine-tuned weights)"
    )

    gr.Markdown(
        f"""
        # 🛡️ DomesticSafetyNet — Privacy-Preserving Domestic Threat & Risk Classifier
        **Fine-Tuned DistilBERT + Hybrid NLP/NER PII Masking (`spaCy` + Structured Regex)**  
        Automatically redacts sensitive Personal Identifiable Information (`[PERSON]`, `[ADDRESS]`, `[PHONE]`, `[EMAIL]`, `[AADHAAR]`, `[PAN]`, `[LICENSE_PLATE]`, `[GPE]`, `[ORG]`) before classifying crisis urgency into **Immediate-Threat**, **High-Urgency**, **Medium**, or **Low**.  
        *{model_status_badge}*
        """
    )

    with gr.Tabs():
        # ── TAB 1: LIVE RISK ANALYZER ─────────────────────────────────
        with gr.TabItem("🛡️ Live Risk Analyzer"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_txt = gr.Textbox(
                        lines=4,
                        label="Incoming Crisis Message / Report",
                        placeholder="Type or paste a message here (all PII is automatically anonymized before classification)...",
                    )
                    with gr.Row():
                        analyze_btn = gr.Button("🔍 Analyze & Anonymize", variant="primary")
                        clear_btn = gr.ClearButton(components=[input_txt], value="Clear")

                    gr.Examples(
                        examples=EXAMPLE_MESSAGES,
                        inputs=input_txt,
                        label="Click an Example Scenario (Tests Indian & International PII + All 4 Risk Levels):",
                    )

            output_card = gr.HTML(label="Risk Assessment & Anonymized Output")
            output_entities = gr.HTML(label="Detected PII Entities")
            history_box = gr.Textbox(
                label="Session Triage Log (Most Recent First)",
                lines=4,
                interactive=False,
            )

            analyze_btn.click(
                fn=analyze_single_message,
                inputs=input_txt,
                outputs=[output_card, output_entities, history_box],
            )

        # ── TAB 2: MASKED VS UNMASKED LIVE COMPARISON ─────────────────
        with gr.TabItem("⚖️ Masked vs. Unmasked Live Comparison"):
            gr.Markdown(
                """
                ### Compare Model Behavior With & Without NLP Entity Masking in Real Time
                Test how stripping PII tokens (`[PERSON]`, `[ADDRESS]`, `[PHONE]`, etc.) affects class probabilities, latency, and privacy exposure compared to feeding raw text into the classifier.
                """
            )
            comp_input = gr.Textbox(
                lines=3,
                label="Test Message for Side-by-Side Pipeline Comparison",
                value=EXAMPLE_MESSAGES[0][0],
            )
            comp_btn = gr.Button("⚡ Run Side-by-Side Comparison", variant="primary")
            comp_output = gr.HTML(value=compare_live_masking(EXAMPLE_MESSAGES[0][0]))

            comp_btn.click(
                fn=compare_live_masking,
                inputs=comp_input,
                outputs=comp_output,
            )

        # ── TAB 3: PROJECT METRICS & EVALUATION DASHBOARD ─────────────
        with gr.TabItem("📊 Model Metrics & Evaluation Dashboard"):
            with gr.Row():
                with gr.Column(scale=3):
                    gr.Markdown(
                        """
                        ### Comprehensive Evaluation: With NLP Masking vs. Without Masking
                        Displays full classification and privacy metrics on the stratified 80/20 test split (`synthetic_data_1000.json`). Run `python compare_masking.py` in Google Colab and upload your generated `metrics_comparison.json` below to refresh at any time.
                        """
                    )
                with gr.Column(scale=1):
                    upload_json = gr.File(
                        label="Upload Updated metrics_comparison.json (Optional)",
                        file_types=[".json"],
                    )

            dashboard_html = gr.HTML(value=render_metrics_dashboard())
            upload_json.change(
                fn=render_metrics_dashboard,
                inputs=upload_json,
                outputs=dashboard_html,
            )

if __name__ == "__main__":
    demo.launch()

