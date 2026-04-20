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
from datetime import datetime
from openai import OpenAI
import zhconv

# ==============================================================================
# AVH Genesis Engine (V36.1 絕對歸一與防爆版)
# ==============================================================================

LLM_MODEL_NAME = "openai/gpt-4o"

print(f"🧠 [載入觀測核心] 啟動 V36.1 穩定回歸版 ({LLM_MODEL_NAME})...")

# 1. 唯一真理源：啟動時直接加載 Manifest
if not os.path.exists("avh_manifest.json"):
    print("工具調用失敗，原因為 遺失底層定義檔 avh_manifest.json")
    sys.exit(1)

with open("avh_manifest.json", "r", encoding="utf-8") as f:
    MANIFEST = json.load(f)

DIMENSION_KEYS = list(MANIFEST["dimensions"].keys())


def get_llm_client():
    token = os.environ.get("COPILOT_GITHUB_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        raise ValueError("遺失 GITHUB_TOKEN，無法啟動算力。")
    return OpenAI(base_url="https://models.github.ai/inference", api_key=token)




def ensure_json_keyword(messages):
    """GitHub Models 的 json_object 模式要求 messages 內必須明示 json。"""
    for m in messages:
        content = str(m.get("content", ""))
        if "json" in content.lower():
            return messages

    patched = list(messages)
    patched.insert(0, {
        "role": "system",
        "content": "Return valid JSON only. The response must be a single JSON object."
    })
    return patched

def call_llm_with_retry(client, messages, temperature=0.0, max_retries=4, json_mode=True):
    last_error = None
    for attempt in range(max_retries):
        try:
            effective_messages = ensure_json_keyword(messages) if json_mode else messages
            kwargs = {"messages": effective_messages, "model": LLM_MODEL_NAME, "temperature": temperature}
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            last_error = e
            wait_time = 2 ** attempt
            print(f"⚠️ 雲端連線異常 (嘗試 {attempt + 1}/{max_retries})，等待 {wait_time} 秒後重試...")
            if attempt < max_retries - 1:
                time.sleep(wait_time)
    raise ConnectionError(f"雲端算力請求超時或阻擋 ({last_error})")


def parse_llm_json(response_text):
    if response_text is None:
        raise ValueError("LLM 未回傳任何內容。")
    text = response_text.strip()

    # 使用 chr(96) 動態生成反引號，物理規避渲染器截斷 Bug
    fence = chr(96) * 3
    pattern = fence + r"(?:json)?\s*(.*?)\s*" + fence
    match = re.search(pattern, text, re.DOTALL)
    if match:
        text = match.group(1).strip()

    start_idx = text.find("{")
    if start_idx == -1:
        raise ValueError("找不到 JSON 起始符號 '{'")

    depth, in_string, escape, end_idx = 0, False, False, -1
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
        escape = (ch == "\\" and not escape)

    if end_idx == -1:
        raise ValueError("找不到完整 JSON 結尾。")

    clean_json = text[start_idx:end_idx + 1]
    try:
        return json.loads(clean_json, strict=False)
    except Exception:
        clean_json = re.sub(r'(?<!\\)\n', ' ', clean_json)
        return json.loads(clean_json, strict=False)


def normalize_whitespace(text):
    return re.sub(r"\s+", " ", str(text)).strip()


def clean_crossref_abstract(raw_abstract):
    if not raw_abstract:
        return ""
    text = re.sub(r"<[^>]+>", " ", raw_abstract)
    text = html.unescape(text)
    return normalize_whitespace(text)


def clamp(value, low, high):
    return max(low, min(high, value))


def dim_label(key):
    return MANIFEST["dimensions"][key]["layer"]


def sign_to_binary(scores_by_key):
    return "".join("1" if scores_by_key[k] > 0 else "0" for k in DIMENSION_KEYS)


def signed_score_to_side(score):
    return "離群突破（sin）" if score > 0 else "合群守成（cos）"


def enforce_score(value, field_name):
    try:
        score = int(round(float(value)))
    except Exception:
        raise ValueError(f"{field_name} 分數無法解析：{value}")
    return clamp(score, -100, 100)


def enforce_confidence(value, field_name):
    try:
        conf = int(round(float(value)))
    except Exception:
        raise ValueError(f"{field_name} 置信度無法解析：{value}")
    return clamp(conf, 0, 100)


def validate_dimension_entries(entries, field_prefix):
    if not isinstance(entries, list) or len(entries) != len(DIMENSION_KEYS):
        raise ValueError(f"{field_prefix} 維度資料數量異常")
    by_key = {str(item.get("key", "")).strip(): item for item in entries if isinstance(item, dict)}
    missing = [k for k in DIMENSION_KEYS if k not in by_key]
    if missing:
        raise ValueError(f"{field_prefix} 缺少維度：{missing}")
    return by_key


def cosine_similarity(vec_a, vec_b):
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def angle_from_cosine(cos_val):
    return math.degrees(math.acos(clamp(cos_val, -1.0, 1.0)))


def proximity_from_scores(user_score, background_score):
    diff = abs(user_score - background_score)
    return round(max(0.0, 100.0 - diff / 2.0), 1)


def classify_relation(user_score, background_score):
    if abs(background_score) < 10:
        return "弱耦合"
    if user_score * background_score < 0:
        return "反向干涉"
    mag_u, mag_b = abs(user_score), abs(background_score)
    if abs(mag_u - mag_b) <= 10:
        return "同向近似"
    return "同向演化 (本體能勢突破)" if mag_u > mag_b else "同向演化 (本體溢出)"


def compact_title(title, max_len=72):
    title = normalize_whitespace(title)
    return title if len(title) <= max_len else title[:max_len - 1] + "…"


def escape_latex(text):
    if text is None:
        return ""
    chars = [
        ("\\", "__BS__"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
        ("__BS__", r"\textbackslash{}"),
    ]
    out = str(text)
    for src, dst in chars:
        out = out.replace(src, dst)
    return out


def markdown_to_latex(text):
    lines = str(text).splitlines()
    out = []
    for line in lines:
        if line.startswith("### "):
            out.append(f"\\subsubsection{{{escape_latex(line[4:])}}}")
        elif line.startswith("## "):
            out.append(f"\\subsection{{{escape_latex(line[3:])}}}")
        elif line.startswith("# "):
            out.append(f"\\section{{{escape_latex(line[2:])}}}")
        else:
            out.append(escape_latex(line))
    return "\n".join(out)


def build_dimensions_prompt():
    payload = [
        {
            "key": k,
            "layer": MANIFEST["dimensions"][k]["layer"],
            "sin": MANIFEST["dimensions"][k]["sin_def"],
            "cos": MANIFEST["dimensions"][k]["cos_def"],
        }
        for k in DIMENSION_KEYS
    ]
    return json.dumps(payload, ensure_ascii=False)


def evaluate_user_profile(raw_text):
    client = get_llm_client()
    manifest_str = build_dimensions_prompt()
    sys_prompt = (
        "你是一台極度嚴謹的「學術本體論量化儀器」。"
        "【術語規範】嚴禁簡體字。必須使用台灣繁體學術語彙（如「資訊」、「網路」、「巨觀」）。"
        "量化規則：每一維回傳 signed_score (-100 到 +100)。"
        "core_statement 控制在 10-15 個英文單字。"
        f"維度定義：{manifest_str}"
    )
    print("🕸️ [階段 1] 量化本體強度向量...")
    response = call_llm_with_retry(
        client,
        messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": raw_text[:8000]}],
        temperature=0.0,
    )
    res = parse_llm_json(response.choices[0].message.content)
    by_key = validate_dimension_entries(res.get("dimensions", []), "本體量化")

    scores = {k: enforce_score(by_key[k].get("signed_score"), k) for k in DIMENSION_KEYS}
    confidences = {k: enforce_confidence(by_key[k].get("confidence"), k) for k in DIMENSION_KEYS}
    reasons = {k: normalize_whitespace(by_key[k].get("reason", "")) for k in DIMENSION_KEYS}

    return {
        "core_statement": normalize_whitespace(res.get("core_statement", "Academic Ontology")),
        "academic_fingerprint": normalize_whitespace(res.get("academic_fingerprint", "")),
        "scores": scores,
        "confidences": confidences,
        "reasons": reasons,
        "hex_code": sign_to_binary(scores),
    }


def fetch_broad_neighborhood_crossref(core_statement):
    headers = {"User-Agent": "AVH-Engine/36.1 (mailto:bot@example.com)"}
    encoded_query = urllib.parse.quote(core_statement)
    url = f"https://api.crossref.org/works?query={encoded_query}&select=DOI,title,abstract&rows=30"
    print(f"🌍 [階段 2] 投放核心宣告：『{core_statement}』")
    try:
        response = requests.get(url, headers=headers, timeout=20)
        if response.status_code == 429:
            time.sleep(5)
            response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        items = response.json().get("message", {}).get("items", [])
        raw_papers = []
        for paper in items:
            abs_text = clean_crossref_abstract(paper.get("abstract"))
            if not abs_text:
                continue
            raw_papers.append({
                "id": str(paper.get("DOI", "Unknown")),
                "title": normalize_whitespace((paper.get("title") or ["Unknown"])[0]),
                "abstract": abs_text[:900],
            })
            if len(raw_papers) >= 20:
                break
        return raw_papers
    except Exception as e:
        raise ConnectionError(f"Crossref 連線異常 ({e})")


def rerank_and_filter_papers(core_statement, raw_papers):
    if not raw_papers:
        return [], "無文獻。"
    client = get_llm_client()
    papers_json = json.dumps(raw_papers, ensure_ascii=False)
    sys_prompt = (
        f'你是一位客觀的高維度學術觀測員。唯一核心宣告："{core_statement}"。'
        "請篩選出底層邏輯同構或可深度對話的文獻，強制剔除撞單字但無關者。最多 8 篇。"
        "【術語規範】禁用「信息、網絡、宏觀」，必須使用「資訊、網路、巨觀」。"
    )
    print("⚖️ [階段 3] 啟動結構重排...")
    response = call_llm_with_retry(
        client,
        messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": papers_json}],
    )
    res = parse_llm_json(response.choices[0].message.content)
    selected_ids = {str(sid).strip() for sid in res.get("selected_ids", [])}
    return [p for p in raw_papers if p["id"] in selected_ids][:8], normalize_whitespace(res.get("filtering_log", ""))


def evaluate_background_papers(final_papers, core_statement):
    if not final_papers:
        return {"papers": [], "batch_log": "無背景文獻。"}
    client = get_llm_client()
    manifest_str = build_dimensions_prompt()
    papers_str = json.dumps(final_papers, ensure_ascii=False)
    sys_prompt = (
        f'你是一台「背景文獻量化儀」。核心宣告："{core_statement}"。'
        f"請用相同六維座標量化對話母體。維度：{manifest_str}。"
        "【術語規範】禁簡體，禁「信息、網絡、宏觀」。"
    )
    print("📚 [階段 4] 逐篇量化背景文獻...")
    response = call_llm_with_retry(
        client,
        messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": papers_str}],
    )
    res = parse_llm_json(response.choices[0].message.content)
    scored_papers = []
    valid_map = {p["id"]: p for p in final_papers}
    for item in res.get("papers", []):
        p_id = str(item.get("id", "")).strip()
        if p_id not in valid_map:
            continue
        by_k = validate_dimension_entries(item.get("scores", []), p_id)
        scored_papers.append({
            "id": p_id,
            "title": valid_map[p_id]["title"],
            "note": normalize_whitespace(item.get("note", "")),
            "scores": {k: enforce_score(by_k[k].get("signed_score"), k) for k in DIMENSION_KEYS},
        })
    return {"papers": scored_papers, "batch_log": normalize_whitespace(res.get("batch_log", ""))}


def build_vector_logs(user_profile, scored_papers):
    user_scores = user_profile["scores"]
    mean_scores, peak_scores, peak_papers = {}, {}, {}
    for key in DIMENSION_KEYS:
        vals = [(p["scores"][key], p) for p in scored_papers]
        mean_scores[key] = round(sum(v for v, _ in vals) / len(vals), 1)
        peak_val, peak_p = max(vals, key=lambda x: x[0])
        peak_scores[key], peak_papers[key] = peak_val, peak_p

    background_hex = sign_to_binary({k: mean_scores[k] for k in DIMENSION_KEYS})
    user_vec, bg_vec = [user_scores[k] for k in DIMENSION_KEYS], [mean_scores[k] for k in DIMENSION_KEYS]
    cos_val = cosine_similarity(user_vec, bg_vec)
    angle = round(angle_from_cosine(cos_val), 1)

    v_logs = []
    for k in DIMENSION_KEYS:
        u, b, pk = user_scores[k], mean_scores[k], peak_scores[k]
        v_logs.append({
            "label": dim_label(k),
            "user": u,
            "mean": b,
            "peak": pk,
            "pk_title": compact_title(peak_papers[k]["title"]),
            "relation": classify_relation(u, b),
            "proximity": proximity_from_scores(u, b),
            "diff_m": round(u - b, 1),
            "diff_p": round(u - pk, 1),
            "compare": "本體能勢突破" if abs(pk) > abs(u) else "本體能勢溢出 (Ontology Override)",
        })
    return {
        "background_hex": background_hex,
        "global_angle": angle,
        "global_cosine": round(cos_val, 4),
        "vector_logs": v_logs,
    }


def generate_summary(raw_text, rel, angle):
    client = get_llm_client()
    prompt = (
        f"關係為：{rel}。相位角：約 {angle} 度。"
        "請撰寫 180-240 字繁體中文導讀。第一句必須以「本理論架構...」開頭。客觀不神話化。"
        "【術語規範】絕對禁止使用「信息、網絡、宏觀」，必須使用「資訊、網路、巨觀」。"
    )
    response = call_llm_with_retry(
        client,
        messages=[{"role": "system", "content": prompt}, {"role": "user", "content": raw_text[:5000]}],
        temperature=0.2,
        json_mode=False,
    )
    return zhconv.convert((response.choices[0].message.content or "").strip(), "zh-tw")


def process_avh_manifestation(source_path):
    print(f"\n🌊 [波包掃描] 實體源碼：{source_path}")
    try:
        with open(source_path, "r", encoding="utf-8") as file:
            raw_text = file.read()
        if len(raw_text.strip()) < 100:
            return None

        user = evaluate_user_profile(raw_text)
        raw_papers = fetch_broad_neighborhood_crossref(user["core_statement"])
        final_papers, filtering_log = rerank_and_filter_papers(user["core_statement"], raw_papers)

        user_dimension_logs = [
            f"* **{dim_label(k)}**：`{user['scores'][k]:+d}` / 100 | {signed_score_to_side(user['scores'][k])} | 置信度 `{user['confidences'][k]}` | 判定：{user['reasons'][k]}"
            for k in DIMENSION_KEYS
        ]

        state = MANIFEST["states"].get(user["hex_code"], {})
        state_name = state.get("name", "")
        state_desc = state.get("desc", "")

        if not final_papers:
            return {
                "user_hex": user["hex_code"],
                "baseline_hex": "000000",
                "state_name": state_name,
                "state_desc": state_desc,
                "summary": "無人區狀態。",
                "full_text": raw_text,
                "meta_data": {
                    "core_statement": user["core_statement"],
                    "academic_fingerprint": user["academic_fingerprint"],
                    "user_dimension_logs": user_dimension_logs,
                    "raw_hits": len(raw_papers),
                    "final_hits": 0,
                    "filtering_log": filtering_log,
                    "baseline_status": "Void",
                    "global_angle": "無定義",
                    "global_relation": "無人區",
                    "llm_model": LLM_MODEL_NAME,
                },
            }

        scored_bg = evaluate_background_papers(final_papers, user["core_statement"])
        vec = build_vector_logs(user, scored_bg["papers"])
        rel = "高度同向" if vec["global_angle"] < 30 else "弱同向" if vec["global_angle"] < 90 else "反向"
        summary = generate_summary(raw_text, rel, vec["global_angle"])

        paper_records = [
            f"- [DOI 連結](https://doi.org/{p['id']}) **{p['title']}** ｜{p['note']}"
            for p in scored_bg["papers"]
        ]

        vector_logs = [
            f"* **{i['label']}**：本體 `{i['user']:+d}` | 背景均值 `{i['mean']:+.1f}` | 峰值 `{i['peak']:+d}`（{i['pk_title']}） | 方向 `{i['relation']}` | 均值差 `{i['diff_m']:+.1f}` | {i['compare']}"
            for i in vec["vector_logs"]
        ]

        return {
            "user_hex": user["hex_code"],
            "baseline_hex": vec["background_hex"],
            "state_name": state_name,
            "state_desc": state_desc,
            "summary": summary,
            "full_text": raw_text,
            "meta_data": {
                "core_statement": user["core_statement"],
                "academic_fingerprint": user["academic_fingerprint"],
                "user_dimension_logs": user_dimension_logs,
                "raw_hits": len(raw_papers),
                "final_hits": len(final_papers),
                "filtering_log": filtering_log,
                "background_batch_log": scored_bg["batch_log"],
                "paper_records": paper_records,
                "vector_logs": vector_logs,
                "baseline_status": "Established",
                "global_angle": f"{vec['global_angle']} 度",
                "global_relation": rel,
                "llm_model": LLM_MODEL_NAME,
            },
        }
    except Exception as e:
        print(f"❌ 失敗: {e}")
        return None


def generate_log_block(target, res):
    meta = res["meta_data"]
    lines = [
        f"## 📡 觀測日誌：`{target}`",
        f"* 引擎：`{meta['llm_model']}`",
        "---",
        "### 1. 🌌 絕對本體觀測",
        f"* 🛡️ 指紋：`[{res['user_hex']}]` - **{res['state_name']}**",
        f"* 核心宣告：`{meta['core_statement']}`",
        "",
        "**學術指紋**：",
        f"> **[基底狀態]** {res['state_desc']}",
        ">",
        f"> **[演化觀測]** {meta['academic_fingerprint']}",
        "",
        *meta["user_dimension_logs"],
        "",
        "---",
        "### 2. 🎣 背景能勢打撈",
        f"* 狀態：`{meta['baseline_status']}` ({meta['raw_hits']} -> {meta['final_hits']})",
        f"* 重排日誌：_{meta['filtering_log']}_",
    ]
    lines.extend(meta.get("paper_records", []))
    lines.extend([
        "",
        "---",
        "### 3. 📐 向量干涉量化",
        f"* 背景 Hex：`[{res['baseline_hex']}]` | 關係：**{meta['global_relation']}** | 相位角：`{meta['global_angle']}`",
        "",
    ])
    lines.extend(meta.get("vector_logs", []))
    lines.extend([
        "",
        "---",
        "### 4. 🧾 系統導讀",
        f"> {res['summary']}",
        "",
    ])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    source_files = [f for f in glob.glob("*.md") if f.lower() not in ["avh_observation_log.md"]]
    if not source_files:
        sys.exit(0)

    with open("AVH_OBSERVATION_LOG.md", "w", encoding="utf-8") as log_file:
        log_file.write("# 📡 AVH 學術價值全像儀：V36.1 穩定回歸日誌\n---\n")
        last_hex = ""

        for target in source_files:
            res = process_avh_manifestation(target)
            if res:
                last_hex = res["user_hex"]
                log_file.write(generate_log_block(target, res))

    if last_hex:
        with open(os.environ.get("GITHUB_ENV", "env.tmp"), "a", encoding="utf-8") as env:
            env.write(f"HEX_CODE={last_hex}\n")
