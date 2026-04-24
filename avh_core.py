import os
import sys
import json
import glob
import re
import requests
import urllib.parse
import time
import math
import html
import subprocess
import collections
from datetime import datetime
import zhconv

# ==============================================================================
# AVH Genesis Engine (V53.0 多視角直讀版 - AI直讀全文、8探針外發、全域能勢池收斂)
# ==============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MANIFEST_PATH = os.path.join(BASE_DIR, "avh_manifest.json")

OLLAMA_MODEL_NAME = "gemma4"
OLLAMA_API_URL = "http://localhost:11434/api/chat"

PRIMARY_STATEMENT_COUNT = 8
RETRIEVAL_ROWS_PER_PROBE = 8
PROBE_WORD_MIN = 6
PROBE_WORD_MAX = 22
PROBE_SIM_THRESHOLD = 0.05

print(f"🧠 [載入本地觀測核心] 啟動 V53.0 多視角直讀版 (引擎: {OLLAMA_MODEL_NAME})...")

if not os.path.exists(MANIFEST_PATH):
    print(f"⚠️ 遺失底層定義檔：{MANIFEST_PATH}，系統終止觀測。")
    sys.exit(1)

with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
    MANIFEST = json.load(f)

DIMENSION_KEYS = list(MANIFEST["dimensions"].keys())

# ==============================================================================
# 語義向量比對引擎 (Probe ↔ Crossref 文獻)
# ==============================================================================

STOP_WORDS = {
    "the", "and", "of", "to", "a", "in", "for", "is", "on", "that", "by", "this",
    "with", "i", "you", "it", "not", "or", "be", "are", "from", "at", "as", "your",
    "all", "have", "new", "we", "an", "was", "can", "will", "via", "using", "based",
    "proposing", "propose", "study", "approach", "method"
}

def get_text_vector(text):
    words = re.findall(r'\b[a-zA-Z]{3,}\b', str(text).lower())
    filtered = [w for w in words if w not in STOP_WORDS]
    return dict(collections.Counter(filtered))

def compute_dict_cosine(d1, d2):
    intersection = set(d1.keys()) & set(d2.keys())
    numerator = sum(d1[x] * d2[x] for x in intersection)
    sum1 = sum(v ** 2 for v in d1.values())
    sum2 = sum(v ** 2 for v in d2.values())
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    if not denominator:
        return 0.0
    return float(numerator) / denominator

# ==============================================================================
# 核心通訊層 (Local LLM)
# ==============================================================================

def call_local_llm(messages, json_mode=False, temperature=0.0, num_ctx=8192):
    payload = {
        "model": OLLAMA_MODEL_NAME,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "num_ctx": num_ctx}
    }
    if json_mode:
        payload["format"] = "json"

    payload_size = len(json.dumps(payload, ensure_ascii=False))
    print(f"   ↳ ⚡ [物理探測] 即將注入資訊熵：{payload_size} 字元。上下文視窗：{num_ctx}...")
    start_time = time.time()

    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=300)
        response.raise_for_status()
        elapsed = time.time() - start_time
        print(f"   ↳ 🟢 [觀測完成] 耗時 {elapsed:.1f} 秒，實體質量成功顯化。")
        return response.json()["message"]["content"]
    except requests.exceptions.Timeout:
        print("\n❌ [邊界破裂] 運算超過 300 秒！資訊熵過載導致引擎超時。")
        sys.exit(1)
    except requests.exceptions.ConnectionError:
        print("\n❌ [邊界破裂] 無法連線至 Ollama。模型可能因 VRAM 溢出而崩潰。")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 本地推演發生網路或通訊錯誤: {e}")
        raise

def parse_llm_json(response_text):
    text = str(response_text).strip()
    if not text:
        raise ValueError("LLM 回傳空白。")

    fence = chr(96) * 3
    pattern = fence + r"(?:json)?\s*(.*?)\s*" + fence
    match = re.search(pattern, text, re.DOTALL)
    if match:
        text = match.group(1).strip()

    start_idx = text.find("{")
    if start_idx == -1:
        raise ValueError("找不到 JSON 起始符號 '{'")

    end_idx, depth, in_string, escape = -1, 0, False, False
    for i in range(start_idx, len(text)):
        ch = text[i]
        if ch == '"' and not escape:
            in_string = not in_string
        if not in_string:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end_idx = i
                    break
        escape = True if ch == "\\" and not escape else False

    if end_idx == -1:
        raise ValueError("找不到完整 JSON 結尾。")
    clean_json = text[start_idx:end_idx + 1]

    try:
        return json.loads(clean_json, strict=False)
    except Exception:
        return json.loads(re.sub(r"(?<!\\)\n", " ", clean_json), strict=False)

# ==============================================================================
# 工具與數學核心
# ==============================================================================

def normalize_whitespace(text):
    return re.sub(r"\s+", " ", str(text)).strip()

def clean_crossref_abstract(raw_abstract):
    return normalize_whitespace(html.unescape(re.sub(r"<[^>]+>", " ", raw_abstract or "")))

def clamp(value, low, high):
    return max(low, min(high, value))

def dim_label(key):
    return MANIFEST["dimensions"][key]["layer"]

def sign_to_binary(scores_by_key):
    return "".join("1" if scores_by_key[k] > 0 else "0" for k in DIMENSION_KEYS)

def signed_score_to_side(score):
    return "離群突破（虛部/sin）" if score > 0 else "合群守成（實部/cos）"

def enforce_score(value, field_name):
    try:
        return clamp(int(round(float(value))), -100, 100)
    except Exception:
        raise ValueError(f"{field_name} 分數異常：{value}")

def enforce_confidence(value, field_name):
    try:
        return clamp(int(round(float(value))), 0, 100)
    except Exception:
        raise ValueError(f"{field_name} 置信度異常：{value}")

def angle_from_cosine(cos_val):
    return math.degrees(math.acos(clamp(cos_val, -1.0, 1.0)))

def proximity_from_scores(user_score, background_score):
    diff = abs(user_score - background_score)
    return round(max(0.0, 100.0 - diff / 2.0), 1)

def classify_relation(user_score, background_score):
    if abs(background_score) < 10:
        return "弱耦合"
    if user_score == 0 and background_score == 0:
        return "中性"
    if user_score * background_score < 0:
        return "反向"
    mag_u, mag_b = abs(user_score), abs(background_score)
    if abs(mag_u - mag_b) <= 10:
        return "同向近似"
    return "同向"

def compact_title(title, max_len=72):
    title = normalize_whitespace(title)
    return title if len(title) <= max_len else title[: max_len - 1] + "…"

def simple_escape(text):
    if not text:
        return ""
    out = str(text)
    for src, dst in [
        ("\\", "\\textbackslash{}"),
        ("&", "\\&"),
        ("%", "\\%"),
        ("$", "\\$"),
        ("#", "\\#"),
        ("_", "\\_"),
        ("{", "\\{"),
        ("}", "\\}")
    ]:
        out = out.replace(src, dst)
    return out

def markdown_to_latex(text):
    out = []
    for line in str(text).splitlines():
        if line.startswith("### "):
            out.append(f"\\subsubsection{{{simple_escape(line[4:])}}}")
        elif line.startswith("## "):
            out.append(f"\\subsection{{{simple_escape(line[3:])}}}")
        elif line.startswith("# "):
            out.append(f"\\section{{{simple_escape(line[2:])}}}")
        else:
            out.append(simple_escape(line))
    return "\n".join(out)

def build_dimensions_prompt():
    payload = [
        {
            "key": k,
            "zh_label": MANIFEST["dimensions"][k]["layer"],
            "sin_def": MANIFEST["dimensions"][k]["sin_def"],
            "cos_def": MANIFEST["dimensions"][k]["cos_def"]
        }
        for k in DIMENSION_KEYS
    ]
    return json.dumps(payload, ensure_ascii=False)

def aggregate_background(scored_papers):
    mean_scores = {}
    peak_scores = {}
    peak_papers = {}
    for key in DIMENSION_KEYS:
        vals = [(p["scores"][key], p) for p in scored_papers]
        mean_scores[key] = round(sum(v for v, _ in vals) / len(vals), 1)
        peak_val, peak_paper = max(vals, key=lambda x: x[0])
        peak_scores[key] = peak_val
        peak_papers[key] = peak_paper
    background_hex = sign_to_binary({k: mean_scores[k] for k in DIMENSION_KEYS})
    return mean_scores, peak_scores, peak_papers, background_hex

def build_vector_logs(user_profile, scored_papers):
    user_scores = user_profile["scores"]
    mean_scores, peak_scores, peak_papers, background_hex = aggregate_background(scored_papers)
    user_vec = [user_scores[k] for k in DIMENSION_KEYS]
    bg_vec = [mean_scores[k] for k in DIMENSION_KEYS]

    cos_val = compute_dict_cosine(dict(enumerate(user_vec)), dict(enumerate(bg_vec)))
    angle = round(angle_from_cosine(cos_val), 1)

    avg_u = sum(user_vec) / 6
    avg_b = sum(bg_vec) / 6
    mean_diff_global = abs(avg_u - avg_b)
    global_proximity = round(max(0.0, 100.0 - ((angle / 1.8) * 0.4 + mean_diff_global * 0.6)), 1)

    if angle < 30:
        base_rel = "高度同向"
    elif angle < 60:
        base_rel = "中度同向"
    elif angle < 90:
        base_rel = "弱同向"
    elif angle == 90:
        base_rel = "正交"
    elif angle < 120:
        base_rel = "弱反向"
    else:
        base_rel = "明顯反向"

    if angle < 90:
        if avg_u > avg_b + 40:
            global_relation = f"{base_rel}（全域能勢大幅突破）"
        elif avg_b > avg_u + 40:
            global_relation = f"{base_rel}（全域能勢大幅覆蓋）"
        else:
            global_relation = f"{base_rel}（能勢共振相近）"
    else:
        global_relation = base_rel

    vector_logs = []
    for key in DIMENSION_KEYS:
        u = user_scores[key]
        b = mean_scores[key]
        peak = peak_scores[key]
        peak_paper = peak_papers[key]
        proximity = proximity_from_scores(u, b)
        relation = classify_relation(u, b)
        diff_mean = round(u - b, 1)
        diff_peak = round(u - peak, 1)
        peak_compare = "背景能勢覆蓋" if abs(peak) > abs(u) else "本體能勢突破"

        vector_logs.append({
            "key": key,
            "label": dim_label(key),
            "user_score": u,
            "background_mean": b,
            "background_peak": peak,
            "peak_title": compact_title(peak_paper["title"]),
            "relation": relation,
            "proximity": proximity,
            "diff_mean": diff_mean,
            "diff_peak": diff_peak,
            "peak_compare": peak_compare,
        })

    return {
        "background_hex": background_hex,
        "mean_scores": mean_scores,
        "peak_scores": peak_scores,
        "peak_papers": peak_papers,
        "global_angle": angle,
        "global_cosine": round(cos_val, 4),
        "global_proximity": global_proximity,
        "global_relation": global_relation,
        "vector_logs": vector_logs,
    }

def format_user_dimension_logs(user_profile):
    logs = []
    for key in DIMENSION_KEYS:
        label = dim_label(key)
        score = user_profile["scores"][key]
        conf = user_profile["confidences"][key]
        reason = user_profile["reasons"][key]
        side = signed_score_to_side(score)
        logs.append(f"* **{label}**：`{score:+d}` / 100 ｜ **{side}** ｜ 置信度 `{conf}` ｜ 觀測判定：{reason}")
    return logs

def format_vector_logs(vector_data):
    logs = []
    for item in vector_data["vector_logs"]:
        logs.append(
            f"* **{item['label']}**：本體 `{item['user_score']:+d}` ｜ 背景均值 `{item['background_mean']:+.1f}` ｜ "
            f"背景峰值 `{item['background_peak']:+d}`（{item['peak_title']}） ｜ "
            f"方向 `{item['relation']}` ｜ 相近度 `{item['proximity']}` / 100 ｜ "
            f"均值差 `{item['diff_mean']:+.1f}` ｜ 峰值差 `{item['diff_peak']:+.1f}` ｜ {item['peak_compare']}"
        )
    return logs

def format_reference_records(scored_papers):
    rows = []
    for p in scored_papers:
        doi_link = f"https://doi.org/{p['id']}" if p["id"] != "Unknown" else "#"
        abs_marker = "" if p.get("has_abs", True) else " 🪧*(標題降維捕獲)*"
        hit_marker = f" ｜多視角命中 `{p.get('source_count', 1)}`" if p.get("source_count", 1) > 1 else ""
        rows.append(f"- [DOI 連結]({doi_link}) **{p['title']}**{abs_marker}{hit_marker}")
    return rows

def generate_summary(raw_text, global_relation, global_angle, global_proximity):
    prompt = f"""
本理論在外部背景場中的整體關係為：{global_relation}。
整體相位角：約 {global_angle} 度。
整體語意相近度：約 {global_proximity} / 100。

請根據下文，撰寫 180-240 字中文理論導讀。第一句必須以「本理論架構...」開頭。客觀不神話化。
""".strip()
    try:
        res_text = call_local_llm(
            [{"role": "system", "content": prompt}, {"role": "user", "content": raw_text[:4000]}],
            temperature=0.2,
            num_ctx=8192
        )
        return zhconv.convert(res_text.strip(), "zh-tw")
    except Exception:
        return f"本理論架構目前與背景場的整體關係為{global_relation}，相位角約為 {global_angle} 度，語意相近度約為 {global_proximity} / 100。由於生成階段發生偏移，系統暫以保底敘述輸出。"

def is_similar_title(t1, t2):
    w1 = set(re.findall(r'\w+', str(t1).lower()))
    w2 = set(re.findall(r'\w+', str(t2).lower()))
    if not w1 or not w2:
        return False
    return (len(w1 & w2) / len(w1 | w2)) > 0.5

def normalize_statement(stmt):
    s = normalize_whitespace(stmt)
    s = s.strip("`\"' ")
    s = re.sub(r"\s+", " ", s)
    return s

def is_valid_probe_statement(stmt):
    s = normalize_statement(stmt)
    word_count = len(re.findall(r'\b[a-zA-Z]+\b', s))
    if word_count < PROBE_WORD_MIN or word_count > PROBE_WORD_MAX:
        return False
    bad_patterns = [
        r"^a study on\b",
        r"^proposing a new\b",
        r"^this paper\b",
        r"^we propose\b",
        r"^an approach to\b"
    ]
    lower = s.lower()
    if any(re.search(p, lower) for p in bad_patterns):
        return False
    return True

def dedupe_statements(statements):
    seen = set()
    out = []
    for stmt in statements:
        s = normalize_statement(stmt)
        if not s:
            continue
        key = s.lower()
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out

# ==============================================================================
# AVH 推演邏輯層
# ==============================================================================

def evaluate_user_profile(raw_text):
    sys_prompt = f"""
你是一台極度嚴謹的「學術本體論量化儀器」。
請根據文本內容，對 6 個維度做定量評估。絕對只能回傳合法的 JSON 格式。

維度定義：
{build_dimensions_prompt()}

量化規則：
1. 每維回傳 score，範圍 -100 到 +100 的整數。
2. 每維回傳 confidence (置信度)，範圍 0 到 100 的整數。
3. reason 必須是客觀的中文短語。絕對禁止使用神祕學或算命語彙。
4. keywords_26：請「消化過」作者的真實意圖後，提取恰好 26 個作者「絕對支持、建構或倡導」的高維專有名詞 (英文)。
   ⚠️ 意圖濾波器：絕對禁止提取作者「反對、批判或欲推翻」的舊有概念！
5. primary_statement：請直接給出你認為最能代表全文的英文核心論述。
6. core_statements_8：請直接根據全文內容提出 8 套英文核心論述。它們是外發探針，不需要再由程式用關鍵字二次組裝。
   ⚠️ 每套必須控制在 22 個英文單字以內，結合 [問題場域] 與 [機制]。嚴禁空泛廢話。

⚠️ 絕對禁止更改以下扁平 JSON 結構中的英文 Key 名稱：
{{
  "primary_statement": "最能代表全文的英文核心論述",
  "keywords_26": ["kw1", "kw2", "...", "kw26"],
  "core_statements_8": [
    "候選核心論述1", "候選核心論述2", "候選核心論述3", "候選核心論述4",
    "候選核心論述5", "候選核心論述6", "候選核心論述7", "候選核心論述8"
  ],
  "academic_fingerprint": "你的中文學術指紋",
  "value_intent_score": 85,
  "value_intent_confidence": 92,
  "value_intent_reason": "說明",
  "governance_score": 70,
  "governance_confidence": 88,
  "governance_reason": "說明",
  "cognition_score": 60,
  "cognition_confidence": 90,
  "cognition_reason": "說明",
  "architecture_score": 40,
  "architecture_confidence": 85,
  "architecture_reason": "說明",
  "expansion_score": 30,
  "expansion_confidence": 80,
  "expansion_reason": "說明",
  "application_score": -10,
  "application_confidence": 75,
  "application_reason": "說明"
}}
""".strip()

    print("🕸️ [階段 1] 多視角直讀：AI 直接閱讀全文，釋放 26 維關鍵字與 8 重外發探針...")
    user_prompt = f"【觀測目標質量開始】\n{raw_text[:4000]}\n【觀測目標質量結束】\n⚠️ 系統底層約束：只輸出 JSON。"

    res = parse_llm_json(
        call_local_llm(
            [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
            json_mode=True,
            num_ctx=8192
        )
    )

    raw_keywords = res.get("keywords_26", [])
    raw_candidates = res.get("core_statements_8", [])
    raw_primary = normalize_statement(res.get("primary_statement", ""))

    valid_statements = []
    if isinstance(raw_candidates, list):
        for cand in raw_candidates:
            cand_str = normalize_statement(cand)
            if is_valid_probe_statement(cand_str):
                valid_statements.append(cand_str)
            else:
                print(f"   ↳ [過濾剔除] Probe 格式不合：{cand_str}")

    valid_statements = dedupe_statements(valid_statements)

    if raw_primary and is_valid_probe_statement(raw_primary):
        if raw_primary.lower() not in {s.lower() for s in valid_statements}:
            valid_statements.insert(0, raw_primary)

    if not valid_statements:
        print("   ↳ ⚠️ [容錯介入] 大腦未回傳有效探針，啟動標題淬取...")
        match = re.search(r'(?:#+)\s*([A-Za-z0-9\-\s:]+)', raw_text)
        if not match:
            match = re.search(r'\(([A-Za-z0-9\-\s]{10,80})\)', raw_text)
        if not match:
            match = re.search(r'([A-Za-z0-9\-\s]{15,80})', raw_text)
        fallback = normalize_statement(match.group(1)) if match else "核心論述提取失敗"
        valid_statements = [fallback]

    primary_statement = raw_primary if raw_primary and is_valid_probe_statement(raw_primary) else valid_statements[0]

    print(f"   ↳ 🎯 [多視角展開] 成功釋放 {len(valid_statements)} 組有效探針：")
    for i, stmt in enumerate(valid_statements, 1):
        head = "Primary" if stmt == primary_statement else f"Probe {i}"
        print(f"      - [{head}] {stmt}")

    try:
        by_key = {}
        for k in DIMENSION_KEYS:
            if f"{k}_score" not in res or f"{k}_confidence" not in res:
                raise ValueError(f"缺少維度分數或置信度：{k}")
            by_key[k] = {
                "signed_score": res.get(f"{k}_score", 0),
                "confidence": res.get(f"{k}_confidence", 0),
                "reason": str(res.get(f"{k}_reason", "無"))
            }
    except Exception as e:
        print(f"⚠️ [維度破裂] LLM 拒絕執行標準量化或遺失必要欄位！原因：{e}")
        by_key = {
            k: {"signed_score": 0, "confidence": 0, "reason": "大腦認定理論已完備，拒絕量化"}
            for k in DIMENSION_KEYS
        }

    return {
        "primary_statement": normalize_whitespace(primary_statement),
        "valid_statements": valid_statements,
        "keywords": [str(k) for k in raw_keywords] if isinstance(raw_keywords, list) else [],
        "academic_fingerprint": normalize_whitespace(res.get("academic_fingerprint", "預設紀錄")),
        "scores": {k: enforce_score(by_key[k].get("signed_score"), k) for k in DIMENSION_KEYS},
        "confidences": {k: enforce_confidence(by_key[k].get("confidence"), k) for k in DIMENSION_KEYS},
        "reasons": {k: normalize_whitespace(by_key[k].get("reason", "")) for k in DIMENSION_KEYS},
        "hex_code": sign_to_binary({k: enforce_score(by_key[k].get("signed_score"), k) for k in DIMENSION_KEYS})
    }

def multi_perspective_retrieval_and_rerank(statements, raw_text):
    """
    V53 核心改動：
    1. 不再先用英文關鍵字 cosine 內部收斂成單句。
    2. 所有 AI 生成的有效 probe 都直接外發 Crossref。
    3. 文獻排序依 probe ↔ 文獻的匹配度，以及多視角命中次數聚合。
    """
    if not statements or statements[0] == "核心論述提取失敗":
        print("🌍 [階段 2] 核心論述失效，中斷 Crossref 打撈，系統將自然回歸無人區狀態。")
        return [], [], 0

    global_candidate_pool = []
    retrieval_logs = []
    raw_hits_count = 0

    print(f"🌍 [階段 2 & 3] 啟動多視角打撈與全域顯化 (共 {len(statements)} 組有效探針，最大搜索量 {len(statements) * RETRIEVAL_ROWS_PER_PROBE} 篇)...")

    for idx, stmt in enumerate(statements):
        print(f"   ↳ ⏳ [視角 {idx + 1}/{len(statements)}] 發射論述: {stmt}")
        stmt_vec = get_text_vector(stmt)

        url = (
            f"https://api.crossref.org/works?"
            f"query={urllib.parse.quote(stmt)}&select=DOI,title,abstract,author,issued&rows={RETRIEVAL_ROWS_PER_PROBE}"
        )

        try:
            response = requests.get(url, headers={"User-Agent": "AVH-Hologram-Engine/53.0"}, timeout=20)
            if response.status_code == 429:
                time.sleep(5)
                response = requests.get(url, headers={"User-Agent": "AVH-Hologram-Engine/53.0"}, timeout=20)
            response.raise_for_status()
        except Exception as e:
            print(f"      ⚠️ API 呼叫失敗 ({e})")
            retrieval_logs.append(f"* **視角 {idx + 1}** `{stmt}`\n  * ⚠️ 打撈落空：Crossref API 呼叫失敗或超時")
            continue

        items = response.json().get("message", {}).get("items", [])
        raw_hits_count += len(items)

        if not items:
            print("      ⚠️ API 回傳 0 篇")
            retrieval_logs.append(f"* **視角 {idx + 1}** `{stmt}`\n  * ⚠️ 打撈落空：Crossref 回傳 0 篇 (API Zero Items)")
            continue

        scored_for_this_stmt = []
        for paper in items:
            title_list = paper.get("title")
            title = normalize_whitespace(title_list[0] if title_list else "Unknown")
            doi = str(paper.get("DOI", "Unknown")).strip()
            abs_text = clean_crossref_abstract(paper.get("abstract", ""))

            if not abs_text:
                eval_text = title
                display_abs = "（此文獻於資料庫中無提供摘要，系統已觸發「標題降維比對」機制。）"
            else:
                eval_text = title + " " + abs_text
                display_abs = abs_text[:900]

            paper_vec = get_text_vector(eval_text)
            sim = compute_dict_cosine(stmt_vec, paper_vec)

            authors = paper.get("author", [])
            first_author = str(authors[0].get("family", "")).lower().strip() if authors else "unknown"
            try:
                year = int(paper.get("issued", {}).get("date-parts", [[0]])[0][0])
            except Exception:
                year = 0

            scored_for_this_stmt.append({
                "id": doi,
                "title": title,
                "abstract": display_abs,
                "author": first_author,
                "year": year,
                "source_statement": stmt,
                "similarity": sim,
                "has_abs": bool(abs_text)
            })

        scored_for_this_stmt.sort(key=lambda x: x["similarity"], reverse=True)

        top3_log = "  * 📊 **Top 3 探針命中度**：\n"
        for i, c in enumerate(scored_for_this_stmt[:3]):
            top3_log += f"    {i + 1}. `[{c['similarity']:.3f}]` {c['title'][:60]}...\n"

        effective_hits = [c for c in scored_for_this_stmt if c["similarity"] >= PROBE_SIM_THRESHOLD]

        if effective_hits:
            best = effective_hits[0]
            sim_str = f"{best['similarity']:.3f}"
            abs_marker = "" if best["has_abs"] else " 🪧*(標題降維捕獲)*"
            status_log = f"  * 🎯 **有效探針捕獲 (Cos >= {PROBE_SIM_THRESHOLD:.2f})**：[{best['title']}{abs_marker}](https://doi.org/{best['id']}) (Cos: `{sim_str}`)"
            print(f"      ✅ 視角最佳捕獲: {best['title'][:40]}... (Cosine: {sim_str})")
        else:
            status_log = f"  * ⚠️ **打撈落空**：此視角皆為低於 {PROBE_SIM_THRESHOLD:.2f} 門檻的弱命中，已被系統物理抹殺。"
            print(f"      ⚠️ 視角落空: 捕獲節點 Cosine 皆小於 {PROBE_SIM_THRESHOLD:.2f} 門檻，已被抹殺。")

        retrieval_logs.append(f"* **視角 {idx + 1}** `{stmt}`\n{top3_log}{status_log}")

        global_candidate_pool.extend(effective_hits)

    print(f"🌍 [全域顯化] 總計 {len(global_candidate_pool)} 篇有效文獻進入全域池，啟動多視角聚合排序...")

    if not global_candidate_pool:
        return [], retrieval_logs, raw_hits_count

    aggregated = {}
    for c in global_candidate_pool:
        key = c["id"] if c["id"] != "Unknown" else c["title"].lower()
        if key not in aggregated:
            aggregated[key] = {
                "id": c["id"],
                "title": c["title"],
                "abstract": c["abstract"],
                "author": c["author"],
                "year": c["year"],
                "has_abs": c["has_abs"],
                "max_similarity": c["similarity"],
                "sum_similarity": c["similarity"],
                "hit_count": 1,
                "source_statements": {c["source_statement"]},
            }
        else:
            agg = aggregated[key]
            agg["max_similarity"] = max(agg["max_similarity"], c["similarity"])
            agg["sum_similarity"] += c["similarity"]
            agg["hit_count"] += 1
            agg["source_statements"].add(c["source_statement"])
            if c["similarity"] > agg["max_similarity"]:
                agg["title"] = c["title"]
                agg["abstract"] = c["abstract"]
                agg["has_abs"] = c["has_abs"]

    merged_candidates = []
    for agg in aggregated.values():
        source_count = len(agg["source_statements"])
        avg_similarity = agg["sum_similarity"] / agg["hit_count"]
        global_score = agg["max_similarity"] * 0.7 + avg_similarity * 0.2 + min(source_count, 4) * 0.05 + min(agg["hit_count"], 4) * 0.05
        merged_candidates.append({
            "id": agg["id"],
            "title": agg["title"],
            "abstract": agg["abstract"],
            "author": agg["author"],
            "year": agg["year"],
            "has_abs": agg["has_abs"],
            "similarity": round(agg["max_similarity"], 4),
            "avg_similarity": round(avg_similarity, 4),
            "hit_count": agg["hit_count"],
            "source_count": source_count,
            "source_statements": sorted(list(agg["source_statements"])),
            "global_score": round(global_score, 4),
        })

    merged_candidates.sort(key=lambda x: (x["global_score"], x["similarity"], x["source_count"]), reverse=True)

    final_papers = []
    seen_dois = set()
    seen_titles = []

    for candidate in merged_candidates:
        if candidate["id"] in seen_dois:
            continue

        is_dup = False
        for st in seen_titles:
            if is_similar_title(candidate["title"], st):
                is_dup = True
                break

        if not is_dup:
            final_papers.append(candidate)
            seen_dois.add(candidate["id"])
            seen_titles.append(candidate["title"])

        if len(final_papers) >= 8:
            break

    print(f"🌍 全域收斂完成：從總池中萃取出 {len(final_papers)} 篇絕對最強文獻，準備進入六維量化...")
    return final_papers, retrieval_logs, raw_hits_count

def evaluate_background_papers(final_papers, core_statement):
    if not final_papers:
        return {"papers": [], "batch_log": "無背景文獻。"}

    print(f"📚 [階段 4] 啟動「切片吞吐」模式，逐篇量化 {len(final_papers)} 篇背景文獻以保護 VRAM...")
    scored_papers = []

    for i, paper in enumerate(final_papers):
        print(f"   ↳ ⏳ [切片吞吐 {i + 1}/{len(final_papers)}] 正在消化: {paper['title'][:30]}...")
        sys_prompt = f"""
觀測原點："{core_statement}"
請量化以下這【1】篇文獻。維度定義：
{build_dimensions_prompt()}

回傳扁平化 JSON 格式：
{{
  "note": "短中文",
  "value_intent_score": 25,
  "governance_score": 10,
  "cognition_score": 40,
  "architecture_score": 35,
  "expansion_score": 20,
  "application_score": -15
}}
""".strip()
        user_prompt = f"【待測背景文獻】\n{json.dumps(paper, ensure_ascii=False)}"

        try:
            res = parse_llm_json(
                call_local_llm(
                    [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
                    json_mode=True,
                    num_ctx=4096
                )
            )
            by_key = {k: {"signed_score": res.get(f"{k}_score", 0)} for k in DIMENSION_KEYS}
            scored_papers.append({
                "id": paper["id"],
                "title": paper["title"],
                "note": normalize_whitespace(res.get("note", "")),
                "scores": {k: enforce_score(by_key[k].get("signed_score"), k) for k in DIMENSION_KEYS},
                "has_abs": paper.get("has_abs", True),
                "source_count": paper.get("source_count", 1)
            })
        except Exception as e:
            print(f"      ⚠️ 該篇文獻消化失敗，觸發動態卸力，直接略過 ({e})")
            continue

        time.sleep(1)

    synthetic_batch_log = f"系統採用切片吞吐模式，成功量化全域最強的 {len(scored_papers)}/{len(final_papers)} 篇文獻。"
    return {"papers": scored_papers, "batch_log": synthetic_batch_log}

# ==============================================================================
# 檔案與輸出控制層
# ==============================================================================

def process_avh_manifestation(source_path):
    print(f"\n🌊 [波包掃描] 實體源碼：{os.path.basename(source_path)}")
    try:
        with open(source_path, "r", encoding="utf-8") as file:
            raw_text = file.read()
        if len(raw_text.strip()) < 100:
            return None

        user_profile = evaluate_user_profile(raw_text)
        user_hex = user_profile["hex_code"]
        state_info = MANIFEST["states"].get(user_hex, {"name": "未知狀態", "desc": "缺乏觀測紀錄"})

        try:
            final_papers, retrieval_logs, raw_hits_count = multi_perspective_retrieval_and_rerank(user_profile["valid_statements"], raw_text)
        except Exception as e:
            final_papers, retrieval_logs, raw_hits_count = [], [f"打撈或收斂失敗（{e}）"], 0

        scored_background = evaluate_background_papers(final_papers, user_profile["primary_statement"]) if final_papers else {"papers": []}

        if not scored_background["papers"]:
            if final_papers:
                baseline_status = f"Void（觀測破裂：{len(final_papers)} 篇背景文獻量化全數失敗）"
                background_batch_log = "打撈已保留文獻，但背景文獻逐篇量化時 LLM 發生崩潰，無法形成有效母體。"
            else:
                baseline_status = "Void（無人區：外部場域尚不足以形成可測量母體）"
                background_batch_log = "最終保留文獻為 0，系統判定當前外部場域不足以構成可測量背景母體。"

            background_hex = "000000"
            vector_logs = ["* **背景向量量化**：無人區狀態或量化失敗，暫無穩定背景向量可供干涉比較。"]
            global_angle = "無定義（Void）"
            global_cosine = "N/A"
            global_proximity = "N/A"
            global_relation = "無人區"
            summary = "本理論架構目前處於無人區狀態；外部鄰近文獻尚不足以形成穩定背景母體，因此與現有學界的方向關係暫時不可定義。"
        else:
            baseline_status = f"Background Field Established（全域能勢建構：{len(final_papers)} 鄰域節點）"
            background_batch_log = scored_background["batch_log"]

            vector_data = build_vector_logs(user_profile, scored_background["papers"])
            background_hex = vector_data["background_hex"]
            vector_logs = format_vector_logs(vector_data)
            global_angle = f"{vector_data['global_angle']} 度"
            global_cosine = vector_data["global_cosine"]
            global_proximity = vector_data["global_proximity"]
            global_relation = vector_data["global_relation"]

            summary = generate_summary(raw_text, vector_data["global_relation"], vector_data["global_angle"], vector_data["global_proximity"])

        return {
            "user_hex": user_hex,
            "baseline_hex": background_hex,
            "state_name": state_info["name"],
            "state_desc": state_info["desc"],
            "summary": summary,
            "full_text": raw_text,
            "meta_data": {
                "primary_statement": user_profile["primary_statement"],
                "keywords": user_profile["keywords"],
                "valid_statements": user_profile["valid_statements"],
                "academic_fingerprint": user_profile["academic_fingerprint"],
                "user_dimension_logs": format_user_dimension_logs(user_profile),
                "raw_hits": raw_hits_count,
                "final_hits": len(final_papers) if final_papers else 0,
                "retrieval_logs": retrieval_logs,
                "background_batch_log": background_batch_log,
                "paper_records": format_reference_records(scored_background["papers"]),
                "vector_logs": vector_logs,
                "baseline_status": baseline_status,
                "global_angle": global_angle,
                "global_cosine": global_cosine,
                "global_proximity": global_proximity,
                "global_relation": global_relation,
                "llm_model": OLLAMA_MODEL_NAME,
            }
        }
    except Exception as e:
        print(f"❌ 處理失敗: {e}")
        return None

def generate_trajectory_log(target_file, data):
    now = datetime.now().astimezone()
    tz_name = now.tzname() or "CST"
    timestamp = now.strftime(f"%Y-%m-%d %H:%M:%S {tz_name}")

    meta = data["meta_data"]
    user_logs_text = "\n\n".join(meta["user_dimension_logs"])
    retrieval_text = "\n".join(meta["retrieval_logs"])
    vector_logs_text = "\n\n".join(meta["vector_logs"])
    papers_text = "\n".join(meta["paper_records"])
    kw_str = ", ".join(meta["keywords"]) if meta["keywords"] else "未萃取"
    probe_str = " ｜ ".join(meta["valid_statements"]) if meta["valid_statements"] else "未釋放"

    return (
        f"## 📡 AVH 技術觀測日誌：`{target_file}`\n"
        f"* **觀測時間戳（{tz_name}）**：`{timestamp}`\n"
        f"* **高維算力引擎（本地純淨版）**：`{meta['llm_model']}`\n\n"
        f"---\n"
        f"### 1. 🌌 絕對本體觀測（Absolute Ontology）\n"
        f"* 🛡️ **本體論絕對指紋（Ontology Hex）**：`[{data['user_hex']}]` - **{data['state_name']}**\n"
        f"  * 📜 **演化實相（State Manifest）**：_{data['state_desc']}_\n"
        f"* **絕對核心論述（Primary Statement）**：`{meta['primary_statement']}`\n"
        f"* **多視角外發探針（Probe Set）**：`{probe_str}`\n"
        f"* **弦論二十六維純淨關鍵字（26 Keywords）**：`{kw_str}`\n\n"
        f"**學術指紋（Academic Fingerprint）**：\n"
        f"> {meta['academic_fingerprint']}\n\n"
        f"**詳細本體量化儀表板（Ontology Quantification Dashboard）**：\n\n"
        f"{user_logs_text}\n\n"
        f"---\n"
        f"### 2. 🎣 背景能勢打撈（Background Field Retrieval）\n"
        f"* **場域建構狀態（Field Status）**：`{meta['baseline_status']}` （多視角搜索共 {meta['raw_hits']} 篇 → 全域收斂至 {meta['final_hits']} 篇）\n"
        f"* **光譜透析多視角打撈日誌（Spectrum Dialysis Retrieval Log）**：\n"
        f"{retrieval_text}\n\n"
        f"* **背景批次量化摘要（Batch Quantification Log）**：_{meta['background_batch_log']}_\n"
        f"* **全域收斂之能勢節點（Top 8 Global Reference Nodes）**：\n"
        f"{papers_text}\n\n"
        f"---\n"
        f"### 3. 📐 向量干涉量化（Quantified Vector Interference）\n"
        f"* **背景絕對指紋（Background Hex）**：`[{data['baseline_hex']}]`\n"
        f"* **整體場域關係（Global Relation）**：**{meta['global_relation']}**\n"
        f"* **整體相位角（Global Angle）**：`{meta['global_angle']}`\n"
        f"* **全域餘弦相似（Global Cosine Similarity）**：`{meta['global_cosine']}`\n"
        f"* **整體語意相近度（Global Semantic Proximity）**：`{meta['global_proximity']}` / 100\n"
        f"* **量化公式（Quantification Rule）**：`Per-dimension proximity = 100 - |U - B| / 2; Global proximity = 100 - ((Angle / 1.8) * 0.4 + Global_Mean_Diff * 0.6)`\n\n"
        f"**維度向量干涉儀表板（Per-Dimension Vector Dashboard）**：\n\n"
        f"{vector_logs_text}\n\n"
        f"---\n"
        f"### 4. 🧾 系統導讀摘要（System Interpretation）\n"
        f"> {data['summary']}\n\n"
        f"---\n"
    )

def export_wordpress_html(basename, data):
    safe_full_text = html.escape(data["full_text"]).replace("\n", "<br>")
    safe_summary = html.escape(data["summary"])
    meta = data["meta_data"]

    now = datetime.now().astimezone()
    tz_name = now.tzname() or "CST"
    timestamp_str = now.strftime(f"%Y-%m-%d %H:%M:%S {tz_name}")

    html_output = (
        "<div class=\"avh-hologram-article\">\n"
        "  <div class=\"avh-content\">\n"
        f"    {safe_full_text}\n"
        "  </div>\n"
        "  <hr>\n"
        "  <div class=\"avh-seal\" style=\"border: 2px solid #333; padding: 20px; background: #fafafa; margin-top: 30px;\">\n"
        "    <h3>📡 學術價值全像儀（AVH）主權算力認證</h3>\n"
        f"    <p><strong>絕對核心論述：</strong>{html.escape(meta['primary_statement'])}</p>\n"
        f"    <p><strong>本體狀態：</strong>[ {html.escape(data['user_hex'])} ] - {html.escape(data['state_name'])}</p>\n"
        f"    <p><em>「{html.escape(data['state_desc'])}」</em></p>\n"
        f"    <p><strong>背景狀態：</strong>[ {html.escape(data['baseline_hex'])} ]</p>\n"
        f"    <p><strong>整體場域關係：</strong>{html.escape(str(meta['global_relation']))}</p>\n"
        f"    <p><strong>整體相位角：</strong>{html.escape(str(meta['global_angle']))}</p>\n"
        f"    <p><strong>整體語意相近度：</strong>{html.escape(str(meta['global_proximity']))} / 100</p>\n"
        f"    <p><strong>理論導讀摘要：</strong><br>{safe_summary}</p>\n"
        f"    <p>物理時間戳：{timestamp_str}</p>\n"
        "  </div>\n"
        "</div>\n"
    )
    with open(os.path.join(BASE_DIR, f"WP_Ready_{basename}.html"), "w", encoding="utf-8") as f:
        f.write(html_output)

def export_latex(basename, data):
    safe_text = markdown_to_latex(data["full_text"])
    meta = data["meta_data"]

    now = datetime.now().astimezone()
    tz_name = now.tzname() or "CST"
    timestamp_str = now.strftime(f"%Y-%m-%d %H:%M:%S {tz_name}")

    tex_output = (
        "\\documentclass{article}\n"
        "\\usepackage[utf8]{inputenc}\n"
        "\\usepackage{xeCJK}\n"
        f"\\title{{{simple_escape(basename)}}}\n"
        "\\author{Alaric Kuo}\n"
        f"\\date{{{timestamp_str}}}\n"
        "\\begin{document}\n"
        "\\maketitle\n"
        "\\begin{abstract}\n"
        f"核心論述：{simple_escape(meta['primary_statement'])}\n\n"
        f"本體狀態：[{data['user_hex']}] {simple_escape(data['state_name'])}\n\n"
        f"演化實相：{simple_escape(data['state_desc'])}\n\n"
        f"背景狀態：[{data['baseline_hex']}]\n\n"
        f"整體場域關係：{simple_escape(str(meta['global_relation']))}\n\n"
        f"整體相位角：{simple_escape(str(meta['global_angle']))}\n\n"
        f"整體語意相近度：{simple_escape(str(meta['global_proximity']))}/100\n"
        "\\end{abstract}\n\n"
        f"{safe_text}\n\n"
        "\\end{document}\n"
    )
    with open(os.path.join(BASE_DIR, f"{basename}_Archive.tex"), "w", encoding="utf-8") as f:
        f.write(tex_output)

def ensure_git_identity():
    try:
        name = subprocess.run(["git", "config", "--get", "user.name"], capture_output=True, text=True, cwd=BASE_DIR).stdout.strip()
        email = subprocess.run(["git", "config", "--get", "user.email"], capture_output=True, text=True, cwd=BASE_DIR).stdout.strip()

        if not name:
            subprocess.run(["git", "config", "user.name", "AVH Local Bot"], check=False, cwd=BASE_DIR)
        if not email:
            subprocess.run(["git", "config", "user.email", "avh-local-bot@example.com"], check=False, cwd=BASE_DIR)
    except Exception:
        pass

def run_git_automation():
    print("\n🚀 [本地自動化] 正在推送到 GitHub...")
    try:
        ensure_git_identity()
        subprocess.run(["git", "add", "."], check=False, cwd=BASE_DIR)
        if subprocess.run(["git", "diff", "--cached", "--quiet"], check=False, cwd=BASE_DIR).returncode == 1:
            now = datetime.now().astimezone()
            tz_name = now.tzname() or "CST"
            commit_msg = f"🌌 自動顯化：本地算力推演定錨 ({now.strftime('%Y-%m-%d %H:%M')} {tz_name})"
            subprocess.run(["git", "commit", "-m", commit_msg], check=True, cwd=BASE_DIR)
            subprocess.run(["git", "push"], check=True, cwd=BASE_DIR)
            print("✅ 推送完成！歷史已成功定錨。")
        else:
            print("ℹ️ 觀測結果無變動。")
    except Exception as e:
        print(f"❌ Git 同步失敗: {e}")

# ==============================================================================
# 主程式進入點
# ==============================================================================

if __name__ == "__main__":
    md_files = [
        f for f in glob.glob(os.path.join(BASE_DIR, "*.md"))
        if os.path.basename(f).lower() != "avh_observation_log.md"
    ]
    if not md_files:
        print("ℹ️ 未找到任何待測 Markdown 來源檔。")
        sys.exit(0)

    log_path = os.path.join(BASE_DIR, "AVH_OBSERVATION_LOG.md")
    success_count = 0

    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write("# 📡 AVH 學術價值全像儀：V53.0 多視角直讀版\n---\n")

        for i, source in enumerate(md_files):
            print(f"\n{'=' * 60}")
            print(f"🚀 [物理觀測啟動] 處理進度 {i + 1}/{len(md_files)}: {os.path.basename(source)}")
            print(f"{'=' * 60}")

            try:
                data = process_avh_manifestation(source)
                if data:
                    success_count += 1
                    log_file.write(generate_trajectory_log(os.path.basename(source), data))
                    basename = os.path.splitext(os.path.basename(source))[0]
                    export_wordpress_html(basename, data)
                    export_latex(basename, data)

                print("\n❄️ [物理散熱] 實體質量處理完畢，強制進入 5 秒冷卻期，釋放 GPU VRAM 壓力...")
                time.sleep(5)

            except Exception as e:
                print(f"❌ [系統級崩潰] 處理 {os.path.basename(source)} 時發生致命錯誤: {e}")
                print("❄️ [物理保護] 異常中止，啟動 10 秒強制散熱，避免連續熱當機...")
                time.sleep(10)

    if success_count > 0:
        run_git_automation()
    else:
        print("❌ 無檔案成功完成處理。")