# step_3_text_and_aggregate.py
from __future__ import annotations
from pathlib import Path
import re, numpy as np, pandas as pd
import spacy, nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from docx import Document as DocxDocument
from pdfminer.high_level import extract_text as pdf_extract_text

# ---------------- Config ----------------
try:
    ROOT = Path(__file__).resolve().parent
except NameError:
    ROOT = Path.cwd()

EVENTS_XLSX = ROOT / "clean_events.xlsx"
SECT_XLSX   = ROOT / "text_features_by_event_sections.xlsx"
EV_OUT_CSV  = ROOT / "text_eventvars.csv"

LAMBDA_M = 0.30
A1 = 1.0
A2 = 2.0
REALIZED_THRESHOLD = 2

# ---------------- NLP init (FIX: add sentencizer) ----------------
nlp = spacy.load("en_core_web_sm", disable=["lemmatizer", "parser"])
if "senter" in nlp.pipe_names:
    nlp.enable_pipe("senter")
elif "sentencizer" not in nlp.pipe_names:
    nlp.add_pipe("sentencizer")

try:
    sia = SentimentIntensityAnalyzer()
except Exception:
    nltk.download('vader_lexicon'); sia = SentimentIntensityAnalyzer()

# ---------------- Patterns ----------------
AI_KEYWORDS_EN = [
    r"\bAI\b", r"\bA\.I\.\b", r"artificial intelligence",
    r"machine learning", r"deep learning",
    r"generative\s*AI", r"\bgen\s*AI\b", r"foundation model",
    r"\bLLM\b", r"large language model", r"transformer(s)?",
    r"neural network(s)?", r"vector (db|database)", r"\bRAG\b"
]
AI_PAT = re.compile("|".join(AI_KEYWORDS_EN), re.I)
FWD_PAT = re.compile(r"\b(plan|intend|aim|target|explore|pilot|roadmap|will|would|going to|expect|hope|ramp up)\b", re.I)
NUM_PAT   = re.compile(r"(\b\d+([\.,]\d+)?\s?%|\b\$?\d+([\.,]\d+)?\s?(bn|b|m|k|million|billion)\b|\b20\d{2}\b|\bQ[1-4]\b)", re.I)
OPS_WORDS = re.compile(r"\b(deploy|roll( |-)?out|integrat(e|ion)|migrat(e|ion)|automate|scale|go live|ship|production|launch|in production)\b", re.I)
PARTNER_WORDS = re.compile(r"\b(partner(ship)?|collaborat(e|ion)|integrat(e|ion) with|powered by)\b", re.I)
NAMED_TECH = re.compile(r"\b(OpenAI|Azure|AWS|Amazon Web Services|Google Cloud|Vertex AI|NVIDIA|SAP|Salesforce|Copilot|Snowflake|Databricks|Hugging Face)\b", re.I)

# ---------------- Utils ----------------
def read_text_any(path_str: str) -> tuple[str, str]:
    p = Path(path_str)
    if not p.exists(): return "", "missing"
    ext = p.suffix.lower()
    try:
        if ext in (".txt",".log",".md"):
            return p.read_text(encoding="utf-8", errors="ignore"), "txt"
        if ext == ".docx":
            doc = DocxDocument(p); return "\n".join(par.text for par in doc.paragraphs), "docx"
        if ext == ".pdf":
            return pdf_extract_text(str(p)) or "", "pdf"
        return p.read_text(encoding="utf-8", errors="ignore"), "other"
    except Exception:
        try: return p.read_bytes().decode("utf-8", errors="ignore"), "bytes"
        except Exception: return "", "error"

def normalize_ws(s: str) -> str:
    return re.sub(r"[ \t]+", " ", re.sub(r"\u00A0", " ", s)).strip()

def section_split_ec(text: str) -> tuple[str, str]:
    markers = re.compile(r"(question(?:s)?\s*and\s*answer(?:s)?|Q&A|Q\&A|questions?\s*from|^operator:?)", re.I|re.M)
    m = markers.search(text)
    if m: return text[:m.start()], text[m.start():]
    alt = re.search(r"(^|\n)(Operator|Question|Analyst)\s*:?", text, flags=re.I)
    if alt: 
        cut = max(1000, alt.start()); return text[:cut], text[cut:]
    return text, ""

def find_ai_token_indices(doc) -> list[int]:
    idxs = []
    for i, tok in enumerate(doc):
        if AI_PAT.search(tok.text): idxs.append(i)
    for i in range(len(doc)-2):
        if AI_PAT.search(doc[i:i+2].text): idxs.append(i)
        if AI_PAT.search(doc[i:i+3].text): idxs.append(i)
    return sorted(set(idxs))

def build_context_windows(doc, hit_idxs, window=10):
    spans=[]; n=len(doc)
    for i in hit_idxs:
        a=max(0,i-window); b=min(n,i+window+1)
        spans.append(doc[a:b])
    if not spans: return []
    spans=sorted(spans,key=lambda s:(s.start,s.end))
    merged=[spans[0]]
    for s in spans[1:]:
        last=merged[-1]
        if s.start<=last.end: merged[-1]=doc[last.start:max(last.end,s.end)]
        else: merged.append(s)
    return merged

def safe_sent_count(span) -> int:
    try:
        return sum(1 for _ in span.sents)
    except Exception:
        return max(1, round(len(span)/20))

def features_for_section(raw_text: str, section_label: str, extractor: str) -> dict:
    text = normalize_ws(raw_text)
    is_scanned_guess = int(extractor == "pdf" and len(text) < 50)
    lang = "en"

    if not text:
        return {
            "Section": section_label, "Extractor": extractor, "IsScannedGuess": is_scanned_guess, "Lang": lang,
            "AI_disclosure_dummy_section": 0, "AI_intensity_section": 0.0, "AI_context_spans": 0,
            "AI_sentiment_AIctx_section": np.nan, "AI_specificity_index_section": 0,
            "AI_tokens": 0, "Section_Tokens": 0,
            "Fwd_count": 0, "Deploy_count": 0, "Numeric_count": 0, "Vendor_count": 0, "Sent_in_windows": 0,
            "AI_Talk_intensity_section": 0.0, "AI_Realized_intensity_section": 0.0,
        }

    total_tokens=0; total_ai_hits=0; ctx_count=0
    sent_scores=[]; sent_in_windows=0
    any_num=any_partner=any_ops=False
    fwd_cnt=dep_cnt=num_cnt=ven_cnt=0

    for piece in [text[i:i+200_000] for i in range(0,len(text),200_000)]:
        doc = nlp(piece)
        total_tokens += len(doc)
        hit_idxs = find_ai_token_indices(doc)
        total_ai_hits += len(hit_idxs)
        windows = build_context_windows(doc, hit_idxs, window=10)
        ctx_count += len(windows)

        for sp in windows:
            t = sp.text.strip()
            if not t: continue
            sent_scores.append(sia.polarity_scores(t)["compound"])
            sent_in_windows += safe_sent_count(sp)

            fwd_cnt += len(FWD_PAT.findall(t))
            dep_cnt += len(OPS_WORDS.findall(t))
            num_cnt += len(NUM_PAT.findall(t))
            ven_cnt += len(PARTNER_WORDS.findall(t)) + len(NAMED_TECH.findall(t))

            if not any_num and NUM_PAT.search(t): any_num=True
            if not any_partner and (PARTNER_WORDS.search(t) or NAMED_TECH.search(t)): any_partner=True
            if not any_ops and OPS_WORDS.search(t): any_ops=True
            if not any_partner:
                for ent in sp.ents:
                    if ent.label_ in ("ORG","PRODUCT"):
                        any_partner=True; break

    ai_share = (total_ai_hits/total_tokens) if total_tokens else 0.0
    tone = float(np.mean(sent_scores)) if sent_scores else np.nan
    spec = int(any_num) + int(any_partner) + int(any_ops)

    talk_sec = ai_share * (1.0 + LAMBDA_M * (fwd_cnt / (1.0 + sent_in_windows)))
    realized_sec = (A1 * dep_cnt) + (A2 * (1 if num_cnt>0 else 0)) + (1 if ven_cnt>0 else 0)

    return {
        "Section": section_label, "Extractor": extractor, "IsScannedGuess": is_scanned_guess, "Lang": lang,
        "AI_disclosure_dummy_section": int(ctx_count>0),
        "AI_intensity_section": ai_share,
        "AI_context_spans": ctx_count,
        "AI_sentiment_AIctx_section": tone,
        "AI_specificity_index_section": spec,
        "AI_tokens": int(total_ai_hits),
        "Section_Tokens": int(total_tokens),
        "Fwd_count": int(fwd_cnt),
        "Deploy_count": int(dep_cnt),
        "Numeric_count": int(num_cnt),
        "Vendor_count": int(ven_cnt),
        "Sent_in_windows": int(sent_in_windows),
        "AI_Talk_intensity_section": float(talk_sec),
        "AI_Realized_intensity_section": float(realized_sec),
    }

# ---------------- Load events ----------------
if not EVENTS_XLSX.exists():
    raise FileNotFoundError(f"Cannot find {EVENTS_XLSX}")

qr = pd.read_excel(EVENTS_XLSX, sheet_name="QR", parse_dates=["EventDate"])
ec = pd.read_excel(EVENTS_XLSX, sheet_name="EC", parse_dates=["EventDate"])
for d in (qr, ec):
    d["Ticker"] = d["Ticker"].astype(str).str.strip()
    d["File Path"] = d["File Path"].astype(str).str.strip()
qr["EventType"]="QR"; ec["EventType"]="EC"
events = pd.concat([qr, ec], ignore_index=True).sort_values(
    ["Ticker","EventDate","EventType"]).reset_index(drop=True)

# ---------------- Process ----------------
rows=[]
for _, e in events.iterrows():
    text, extractor = read_text_any(e["File Path"])
    text = normalize_ws(text)
    base = {
        "Country": e.get("Country",""),
        "Ticker": e["Ticker"],
        "Company": e.get("Company",""),
        "Source": e.get("Source",""),
        "EventDate": e["EventDate"],
        "File Path": e["File Path"],
        "EventType": e["EventType"],
    }
    if e["EventType"]=="EC":
        prepared, qa = section_split_ec(text)
        for part,label in [(prepared,"Prepared"), (qa,"QA")]:
            feats = features_for_section(part, label, extractor)
            r = base.copy(); r.update(feats); rows.append(r)
    else:
        feats = features_for_section(text, "Report", extractor)
        r = base.copy(); r.update(feats); rows.append(r)

features = pd.DataFrame(rows).sort_values(
    ["Ticker","EventDate","EventType","Section"]).reset_index(drop=True)

# ---------------- Quality sheet ----------------
if not features.empty:
    quality = (features
        .groupby(["EventType","Section","Extractor","IsScannedGuess","Lang"], dropna=False)
        .agg(docs=("Ticker","count"),
             med_chars=("Section_Tokens","median"),
             med_tokens=("AI_tokens","median"),
             ai_hit_rate=("AI_disclosure_dummy_section","mean"),
             mean_intensity_pp=("AI_Talk_intensity_section","mean"),
             mean_neg_share=("AI_specificity_index_section","mean"))
        .reset_index())
else:
    quality = pd.DataFrame(columns=["EventType","Section","Extractor","IsScannedGuess","Lang",
                                    "docs","med_chars","med_tokens","ai_hit_rate",
                                    "mean_intensity_pp","mean_neg_share"])

with pd.ExcelWriter(SECT_XLSX, engine="xlsxwriter", datetime_format="yyyy-mm-dd") as w:
    features.to_excel(w, index=False, sheet_name="Features")
    quality.to_excel(w, index=False, sheet_name="Quality")

# ---------------- Aggregate -> event level ----------------
keys = ["Ticker","EventDate","EventType","Source"]

def agg_event(g: pd.DataFrame) -> pd.Series:
    wt = g["Section_Tokens"].clip(lower=1)
    talk = np.average(g["AI_Talk_intensity_section"], weights=wt)
    realized = g["AI_Realized_intensity_section"].sum()
    if "AI_sentiment_AIctx_section" in g and "AI_context_spans" in g:
        tone = np.average(g["AI_sentiment_AIctx_section"].fillna(0), weights=g["AI_context_spans"].clip(lower=1))
    else:
        tone = g.get("AI_sentiment_AIctx_section", pd.Series([np.nan]*len(g))).mean()
    return pd.Series({
        "AI_Talk_Intensity": talk,
        "AI_Realized_Index": realized,
        "Tone": tone,
        "Tokens_total": g["Section_Tokens"].sum(),
        "Has_Prepared": int((g["Section"]=="Prepared").any()),
        "Has_QA": int((g["Section"]=="QA").any()),
        "Has_Report": int((g["Section"]=="Report").any())
    })

ev = features.groupby(keys, as_index=False).apply(agg_event).reset_index(drop=True)

ev["Realized_Flag"] = ev["AI_Realized_Index"] >= REALIZED_THRESHOLD
ev["Talk_Flag"]     = (~ev["Realized_Flag"]) & (ev["AI_Talk_Intensity"] > 0)
ev["Primary_Label"] = np.where(ev["Realized_Flag"], "Realized",
                          np.where(ev["Talk_Flag"], "Talk", "None"))

def z_by_channel(x: pd.Series) -> pd.Series:
    sd = x.std(ddof=1)
    return (x - x.mean()) / (sd if sd and not np.isnan(sd) and sd > 0 else 1.0)

ev["z_AI_Talk_Intensity"] = ev.groupby("EventType")["AI_Talk_Intensity"].transform(z_by_channel)
ev["z_AI_Realized_Index"] = ev.groupby("EventType")["AI_Realized_Index"].transform(z_by_channel)
ev["z_Tone"]              = ev.groupby("EventType")["Tone"].transform(z_by_channel)

ev.to_csv(EV_OUT_CSV, index=False)

print(f"✅ Step 3 complete.")
print(f"Saved sections  -> {SECT_XLSX} (Features={len(features):,} rows)")
print(f"Saved event vars -> {EV_OUT_CSV} (Events={len(ev):,}; Realized share={ev['Realized_Flag'].mean():.2%})")
