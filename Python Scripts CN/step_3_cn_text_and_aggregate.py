"""
CN Step 3: Text features and event-level variables for SSE 50 (CN).
Inputs:
  - events_cn_collapsed.xlsx  (sheets: 'EC_collapsed', 'QR_raw')
Outputs:
  - text_features_by_event_sections_cn.xlsx (sheet 'Features' + 'Quality')
  - text_eventvars_cn.csv     (event-level variables used by Step 4+)
Notes:
  * Uses InlineText when present; for QR with paths, reads PDF via pdfminer.
  * Chinese tokenization via jieba if available; else simple fallback.
  * Chinese AI dictionary + forward-looking and realized patterns.
"""

from __future__ import annotations
from pathlib import Path
import re
import math
import logging
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

ROOT      = Path.cwd()
EVENTS_XL = ROOT / "events_cn_collapsed.xlsx"
OUT_XLSX  = ROOT / "text_features_by_event_sections_cn.xlsx"
OUT_CSV   = ROOT / "text_eventvars_cn.csv"

logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("pdfminer.pdfinterp").setLevel(logging.ERROR)

from pdfminer.high_level import extract_text as pdf_extract_text

def read_text_any(path_or_text: str, inline: bool) -> str:
    if inline:
        return str(path_or_text or "").strip()
    p = Path(str(path_or_text))
    if not p.exists():
        return ""
    if p.suffix.lower() == ".pdf":
        try:
            return pdf_extract_text(str(p)) or ""
        except Exception:
            return ""
    # fallback plain text
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        try:
            return p.read_bytes().decode("utf-8", errors="ignore")
        except Exception:
            return ""

try:
    import jieba
    def tokenize_cn(s: str) -> List[str]:
        if not s:
            return []
        # cut_all=False preserves better word boundaries
        return [t.strip() for t in jieba.cut(s, cut_all=False) if t.strip()]
except Exception:
    def tokenize_cn(s: str) -> List[str]:
        # very simple fallback: split on non-CJK and single chars
        return [c for c in re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", s) if c.strip()]

_use_tone = False
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    _tone_model_id = "IDEA-CCNL/Erlangshen-Roberta-110M-Financial"
    _tok = AutoTokenizer.from_pretrained(_tone_model_id, local_files_only=True)
    _mdl = AutoModelForSequenceClassification.from_pretrained(_tone_model_id, local_files_only=True)
    _mdl.eval()
    _use_tone = True
except Exception:
    _use_tone = False

def tone_score_cn(spans: List[str]) -> float | None:
    if not spans:
        return None
    if not _use_tone:
        return None
    vals = []
    for t in spans:
        if not t.strip():
            continue
        with torch.no_grad():
            inputs = _tok(t[:256], return_tensors="pt", truncation=True)
            logits = _mdl(**inputs).logits  # assume 3-way or similar
            # map to a rough (-1..+1) tone: (pos - neg) / sum
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            if len(probs) == 3:
                neg, neu, pos = probs
                score = float(pos - neg)
            else:
                # binary fallback: treat index 1 as positive
                if len(probs) == 2:
                    score = float(probs[1] - probs[0])
                else:
                    score = float(probs.mean() - 0.5)
        vals.append(score)
    return float(np.mean(vals)) if vals else None

AI_PATTERNS = [
    r"人工智能", r"机器学习", r"深度学习", r"生成式\s*AI", r"生成式人工智能", r"AIGC",
    r"大模型", r"大语言模型", r"大型语言模型", r"\bLLM\b",
    r"神经网络", r"向量数据库", r"向量库", r"RAG", r"检索增强",
    r"智能体", r"智能助手", r"Copilot", r"ChatGPT", r"Transformer"
]
AI_RE = re.compile("|".join(AI_PATTERNS), re.I)

# Forward-looking (Talk) cues within AI windows
FWD_WORDS = re.compile(r"(计划|将|拟|预计|目标|推进|布局|探索|打算|有望|力争|路线图|规划)", re.I)

# Realized (deployment/operations) cues
REAL_VERBS = re.compile(r"(已(经)?(上线|发布|落地|投产|集成|部署|应用|商用|开通|运行|量产)|部署|上线|落地|投产|集成|接入|迁移|商用|试运行)", re.I)

# Numeric / operational cues
NUM_CUES = re.compile(
    r"(\b\d+(\.\d+)?\s*(%|％|人|位|家|台|套|个|TFlops|TFLOPS|TOPS|TB|GB|亿元|亿|万元|万|千|百万|千万)\b|"
    r"\b20\d{2}\b|Q[1-4]|上(线|市)时间|\d+\s*(模型|用户|服务器|节点|算力))", re.I
)

# Named partners / vendors (add/trim as needed)
VENDORS = re.compile(
    r"(华为|阿里云|腾讯云|天翼云|百度|科大讯飞|OpenAI|微软|Azure|亚马逊|AWS|谷歌|Google|NVIDIA|英伟达|寒武纪|海光|浪潮|SAP|Salesforce|Snowflake|Databricks)",
    re.I
)

NEG_WORDS = re.compile(r"(不|未|没有|无|难以|尚未)", re.I)

def ai_hit_indices(tokens: List[str]) -> List[int]:
    idx = []
    for i, t in enumerate(tokens):
        if AI_RE.search(t):
            idx.append(i)
    return idx

def build_windows(tokens: List[str], hits: List[int], win: int = 20) -> List[Tuple[int,int]]:
    spans = []
    n = len(tokens)
    for i in hits:
        a = max(0, i - win)
        b = min(n, i + win + 1)
        spans.append((a, b))
    if not spans:
        return []
    spans.sort()
    merged = [spans[0]]
    for s in spans[1:]:
        a,b = s; la,lb = merged[-1]
        if a <= lb:
            merged[-1] = (la, max(lb, b))
        else:
            merged.append((a,b))
    return merged

def slice_text(tokens: List[str], span: Tuple[int,int]) -> str:
    a,b = span
    return "".join(tokens[a:b])

def section_features(text: str, section: str, lang: str = "zh") -> Dict[str, object]:
    txt = (text or "").strip()
    if not txt:
        return {
            "Section": section,
            "AI_disclosure_dummy_section": 0,
            "AI_context_spans": 0,
            "AI_intensity_section": 0.0,
            "AI_negated_share_section": 0.0,
            "AI_sentiment_AIctx_section": np.nan,
            "AI_specificity_index_section": 0,
            "AI_Talk_intensity_section": 0.0,
            "AI_Realized_intensity_section": 0.0,
            "Section_Tokens": 0
        }

    tokens = tokenize_cn(txt)
    T = len(tokens)
    if T == 0:
        return {
            "Section": section,
            "AI_disclosure_dummy_section": 0,
            "AI_context_spans": 0,
            "AI_intensity_section": 0.0,
            "AI_negated_share_section": 0.0,
            "AI_sentiment_AIctx_section": np.nan,
            "AI_specificity_index_section": 0,
            "AI_Talk_intensity_section": 0.0,
            "AI_Realized_intensity_section": 0.0,
            "Section_Tokens": 0
        }

    hits = ai_hit_indices(tokens)
    wins = build_windows(tokens, hits, win=20)

    ai_intensity = len(hits) / float(T)

    neg_cnt = 0
    has_num = False; has_vendor = False; has_ops = False
    spans_text = []
    fwd_count = 0
    realized_count = 0

    for sp in wins:
        s_txt = slice_text(tokens, sp)
        spans_text.append(s_txt)
        if NEG_WORDS.search(s_txt):
            neg_cnt += 1
        if NUM_CUES.search(s_txt):     has_num = True
        if VENDORS.search(s_txt):      has_vendor = True
        if REAL_VERBS.search(s_txt):   has_ops = True
        # talk vs realized cues
        if FWD_WORDS.search(s_txt):    fwd_count += 1
        if REAL_VERBS.search(s_txt) or NUM_CUES.search(s_txt): realized_count += 1

    neg_share = (neg_cnt / len(wins)) if wins else 0.0
    spec_idx  = int(has_num) + int(has_vendor) + int(has_ops)

    # Talk / Realized intensities (scaled variants)
    # Talk: AI share × (1 + λ * forward-looking density)
    lam = 0.30
    talk_int = ai_intensity * (1.0 + lam * (fwd_count / max(1, len(wins))))
    real_int = ai_intensity * (1.0 + 0.50 * (realized_count / max(1, len(wins))))

    tone = tone_score_cn(spans_text)

    return {
        "Section": section,
        "AI_disclosure_dummy_section": 1 if wins else 0,
        "AI_context_spans": len(wins),
        "AI_intensity_section": ai_intensity,
        "AI_negated_share_section": neg_share,
        "AI_sentiment_AIctx_section": tone if tone is not None else np.nan,
        "AI_specificity_index_section": spec_idx,
        "AI_Talk_intensity_section": talk_int,
        "AI_Realized_intensity_section": real_int,
        "Section_Tokens": T
    }

# --------- Main run ----------
if __name__ == "__main__":
    if not EVENTS_XL.exists():
        raise FileNotFoundError(f"Cannot find {EVENTS_XL}")

    xls = pd.ExcelFile(EVENTS_XL)

    ec = pd.read_excel(xls, sheet_name="EC_collapsed")
    qr = pd.read_excel(xls, sheet_name="QR_raw")

    for d in (ec, qr):
        d.columns = [c.strip() for c in d.columns]

    ec["Ticker"]    = ec["Ticker"].astype(str).str.upper().str.replace(".SS",".SH",regex=False)
    ec["EventDate"] = pd.to_datetime(ec["EventDate"], errors="coerce")
    ec["Source"]    = ec.get("Source", "EC").astype(str)
    ec["EventType"] = "EC"
    ec["Extractor"] = "inline"

    qr["Ticker"]    = qr["Ticker"].astype(str).str.upper().str.replace(".SS",".SH",regex=False)
    qr["EventDate"] = pd.to_datetime(qr["EventDate"], errors="coerce")
    qr["Source"]    = qr.get("Source", "QR").astype(str)
    qr["EventType"] = "QR"
    qr["Extractor"] = np.where(qr["File Path"].notna() & (qr["File Path"].astype(str).str.len() > 4), "pdf", "inline")

    events = pd.concat([ec[["Ticker","EventDate","Source","File Path","InlineText","EventType","Extractor"]],
                        qr[["Ticker","EventDate","Source","File Path","InlineText","EventType","Extractor"]]],
                       ignore_index=True)
    events = events.dropna(subset=["Ticker","EventDate"]).sort_values(["Ticker","EventDate","EventType"]).reset_index(drop=True)

    rows = []
    for _, e in events.iterrows():
        inline = False
        content = ""
        if pd.notna(e.get("InlineText","")) and str(e.get("InlineText","")).strip():
            content = str(e["InlineText"])
            inline = True
        else:
            content = read_text_any(str(e.get("File Path","")), inline=False)
        feats = section_features(content, section=("Report" if e["EventType"]=="QR" else "Transcript"), lang="zh")
        base = {
            "Ticker": e["Ticker"], "EventDate": e["EventDate"],
            "Source": e["Source"], "EventType": e["EventType"],
            "Extractor": e["Extractor"], "IsScannedGuess": 1 if (not inline and len(content.strip()) < 100) else 0,
            "Lang": "zh",
            "File Path": e.get("File Path", None) if not inline else "INLINE_TEXT",
        }
        base.update(feats)
        rows.append(base)

    feat = pd.DataFrame(rows)
    feat = feat.sort_values(["Ticker","EventDate","EventType","Section"]).reset_index(drop=True)

    qual = (feat.groupby(["EventType","Section","Extractor","IsScannedGuess","Lang"], dropna=False)
                .agg(docs=("Ticker","size"),
                     med_chars=("Section_Tokens", "median"),
                     ai_hit_rate=("AI_disclosure_dummy_section","mean"),
                     mean_talk=("AI_Talk_intensity_section","mean"),
                     mean_real=("AI_Realized_intensity_section","mean"),
                     mean_neg_share=("AI_negated_share_section","mean"))
                .reset_index())

    with pd.ExcelWriter(OUT_XLSX, engine="xlsxwriter", datetime_format="yyyy-mm-dd") as w:
        feat.to_excel(w, index=False, sheet_name="Features")
        qual.to_excel(w, index=False, sheet_name="Quality")

    ev = (feat.groupby(["Ticker","EventDate","EventType"], as_index=False)
              .agg(
                  AI_Talk_Intensity=("AI_Talk_intensity_section","mean"),
                  AI_Realized_Index=("AI_Realized_intensity_section","mean"),
                  Tone=("AI_sentiment_AIctx_section", "mean"),
                  AI_Specificity=("AI_specificity_index_section","max"),
                  AI_Windows=("AI_context_spans","sum"),
                  Has_AI=("AI_disclosure_dummy_section","max")
              ))

    ev["Talk_Flag"]     = (ev["AI_Talk_Intensity"] > 0).astype(int)
    ev["Realized_Flag"] = (ev["AI_Realized_Index"] > 0).astype(int)

    ev = ev.sort_values(["Ticker","EventDate","EventType"]).reset_index(drop=True)
    ev.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print(f"✅ Saved: {OUT_XLSX} (rows={len(feat):,})")
    print(f"✅ Saved: {OUT_CSV}  (events={len(ev):,}, AI events={ev['Has_AI'].sum():,})")
