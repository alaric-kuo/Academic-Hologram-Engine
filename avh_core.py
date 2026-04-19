import os
import sys
import json
import glob
import re
import requests
import urllib.parse
import time
from datetime import datetime
from openai import OpenAI
import zhconv

# ==============================================================================
# AVH Genesis Engine (V30.2 自然母體對撞 - 柔性容錯與無人區宣告版)
# ==============================================================================

LLM_MODEL_NAME = 'openai/gpt-4o'
MD_FENCE = "`" * 3

print(f"🧠 [載入觀測核心] 啟動 V30.2 高維度大腦矩陣 ({LLM_MODEL_NAME})...")

def get_llm_client():
    token = os.environ.get("COPILOT_GITHUB_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        print("工具調用失敗，原因為 遺失 GITHUB_TOKEN")
        sys.exit(1)
    return OpenAI(base_url="https://models.github.ai/inference", api_key=token)

def call_llm_with_retry(client, messages, temperature=0.1, max_retries=4):
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                messages=messages,
                model=LLM_MODEL_NAME, 
                temperature=temperature
            )
        except Exception as e:
            wait_time = 2 ** attempt
            print(f"⚠️ 雲端連線異常 (嘗試 {attempt + 1}/{max_retries})，等待 {wait_time} 秒後重試... [{e}]")
            if attempt == max_retries - 1:
                print(f"工具調用失敗，原因為 雲端算力請求超時或阻擋 ({e})")
                sys.exit(1)
            time.sleep(wait_time)

def parse_llm_json(response_text):
    try:
        text = response_text.strip()
        pattern = f"{MD_FENCE}(?:json)?\\s*(\\{{.*?\\}})\\s*{MD_FENCE}"
        fence_match = re.search(pattern, text, re.DOTALL)
        if fence_match:
            return json.loads(fence_match.group(1))

        obj_match = re.search(r"(\{.*\})", text, re.DOTALL)
        if obj_match:
            return json.loads(obj_match.group(1))
        raise ValueError("找不到可解析的 JSON 區塊")
    except Exception as e:
        print(f"工具調用失敗，原因為 LLM JSON 解析失敗 ({e})")
        sys.exit(1)

def evaluate_user_text_and_compress(raw_text, manifest):
    client = get_llm_client()
    manifest_str = json.dumps(manifest["dimensions"], ensure_ascii=False)
    
    sys_prompt = f"""
你是一台極度嚴謹的「學術本體論觀測儀器」。請閱讀文本並評估 6 個維度(1=突破, 0=守成)。
維度定義：{manifest_str}

為了向外部圖譜發動「語意寬鬆檢索」，請將這篇文本的底層物理/治理/系統邏輯，壓縮成一句「精準的英文學術核心宣告 (Core Statement)」。
**這句話必須控制在 10 到 15 個英文單字之間。** 請回傳 JSON：
{MD_FENCE}json
{{
  "hex_code": "111111",
  "dim_logs": [
    "* **價值意圖**：離群突破 (sin) `[觀測判定：...]`",
    "* **治理維度**：離群突破 (sin) `[觀測判定：...]`",
    "* **認知深度**：離群突破 (sin) `[觀測判定：...]`",
    "* **描述架構**：離群突破 (sin) `[觀測判定：...]`",
    "* **擴張潛力**：離群突破 (sin) `[觀測判定：...]`",
    "* **應用實相**：離群突破 (sin) `[觀測判定：...]`"
  ],
  "core_statement": "Quantum topology approach to trust governance and anti-fragility systems"
}}
{MD_FENCE}
"""
    print("🕸️ [大腦運算 - 階段 1] 測量本體絕對指紋，執行核心塌陷 (12-word Core)...")
    response = call_llm_with_retry(
        client, 
        messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": raw_text[:4000]}],
        temperature=0.1
    )
    return parse_llm_json(response.choices[0].message.content)

def fetch_broad_neighborhood_crossref(core_statement):
    headers = {
        "User-Agent": "AVH-Hologram-Engine/30.2 (https://github.com/alaric-kuo; mailto:open-source-bot@example.com)"
    }
    encoded_query = urllib.parse.quote(core_statement)
    url = f"https://api.crossref.org/works?query={encoded_query}&select=DOI,title,abstract,is-referenced-by-count&rows=25"
    
    print(f"🌍 [實體觀測 - 階段 2] 投放核心宣告：『{core_statement}』\n🌍 正在 Crossref 禮貌池中打撈 Top 25 關聯文獻...")
    
    try:
        response = requests.get(url, headers=headers, timeout=20)
        if response.status_code == 429:
            print(f"⚠️ 遭遇 Crossref 瞬間限流，強制退避 5 秒...")
            time.sleep(5)
            response = requests.get(url, headers=headers, timeout=20)
            
        response.raise_for_status()
        data = response.json()
        
        items = data.get("message", {}).get("items", [])
        if not items:
            print(f"工具調用失敗，原因為 核心宣告在 Crossref 查無任何文獻")
            sys.exit(1)
            
        raw_papers = []
        for paper in items:
            raw_abstract = paper.get("abstract")
            if not raw_abstract: 
                continue
            clean_abstract = re.sub(r'<[^>]+>', '', raw_abstract) 
            title = paper.get("title", [""])[0] if paper.get("title") else "Unknown"
            
            raw_papers.append({
                "id": paper.get("DOI", "Unknown"),
                "title": title,
                "abstract": clean_abstract[:600],
                "citations": paper.get("is-referenced-by-count", 0)
            })
            
            if len(raw_papers) >= 20:
                break
                
        if len(raw_papers) < 3:
            print(f"工具調用失敗，原因為 撈取到的合格摘要過少，無法支撐重排 ({len(raw_papers)} 篇)")
            sys.exit(1)
            
        time.sleep(1)
        return raw_papers
        
    except Exception as e:
        print(f"工具調用失敗，原因為 Crossref 連線異常或超時 ({e})")
        sys.exit(1)

def rerank_and_filter_papers(core_statement, raw_papers):
    """【V30.2 核心】放寬絕對限制，接受低數量的母體，甚至允許大腦宣告 0 篇 (無人區)"""
    client = get_llm_client()
    papers_json = json.dumps(raw_papers, ensure_ascii=False)
    
    sys_prompt = f"""
你現在是一位客觀的學術觀測員。
我的核心理論宣告是："{core_statement}"
以下是從傳統搜尋引擎撈回來的 {len(raw_papers)} 篇初步相關文獻。

請利用你的高維度認知閱讀它們。剔除那些「只是撞字、核心邏輯毫無關聯」的雜訊。
挑選出「在現有學術範式中，最適合拿來作為該理論『參考座標』或『對話對象』」的文獻 (最多 8 篇)。

【絕對柔性指令】：如果該理論過於前沿（處於無人區），導致這 {len(raw_papers)} 篇文獻中「連一篇配得上作為參考座標的都沒有」，你完全可以回傳少量的 id，甚至回傳空陣列 `[]`。

請回傳 JSON：
{MD_FENCE}json
{{
  "selected_ids": ["id_1", "id_2"],
  "filtering_log": "簡述你保留了哪些文獻作為座標，或為何判定多數文獻皆不合適"
}}
{MD_FENCE}
"""
    print(f"⚖️ [大腦運算 - 階段 3] 啟動柔性重排 (Re-ranking)，尋找參考座標，容許無人區判定...")
    response = call_llm_with_retry(
        client,
        messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": papers_json}],
        temperature=0.0
    )
    res = parse_llm_json(response.choices[0].message.content)
    
    selected_ids = set(res.get("selected_ids", []))
    filtering_log = res.get("filtering_log", "執行標準過濾機制。")
    
    final_papers = [p for p in raw_papers if p["id"] in selected_ids]
    # 💥 移除舊版 sys.exit(1) 的死亡限制，讓 0 篇也能順利往下走
    return final_papers, filtering_log

def evaluate_baseline_papers(papers, manifest):
    client = get_llm_client()
    manifest_str = json.dumps(manifest["dimensions"], ensure_ascii=False)
    papers_str = json.dumps([{"title": p["title"], "abstract": p["abstract"]} for p in papers])
    
    sys_prompt = f"""
你正在測量當代學術的「真實背景能勢場」。以下是由系統精準篩選出的自然母體論文群。
請綜合判斷這個論文群體所構成的場域，在 6 個維度上的整體表現。
維度定義：{manifest_str} (1=突破, 0=守成)

請回傳 JSON：
{MD_FENCE}json
{{
  "baseline_hex": "010011",
  "vote_stats": [2, 3, 1, 4, 5, 4] 
}}
{MD_FENCE}
"""
    print("⚖️ [場域測量 - 階段 4] 計算真實自然母體之絕對張量...")
    response = call_llm_with_retry(
        client,
        messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": papers_str}],
        temperature=0.1
    )
    res = parse_llm_json(response.choices[0].message.content)
    return res.get("baseline_hex", "000000"), res.get("vote_stats", [0]*6)

def process_avh_manifestation(source_path, manifest):
    print(f"\n🌊 [波包掃描] 實體源碼：{source_path}")
    try:
        with open(source_path, 'r', encoding='utf-8') as file:
            raw_text = file.read()
            
        if len(raw_text.strip()) < 100:
            return None

        # 1. User Hex & 12-Word Core
        user_data = evaluate_user_text_and_compress(raw_text, manifest)
        user_hex = user_data["hex_code"]
        dim_logs = user_data["dim_logs"]
        core_statement = user_data.get("core_statement", "Academic Ontology Theory")
        user_state_info = manifest["states"].get(user_hex, {"name": "未知狀態", "desc": "缺乏觀測紀錄"})
        
        # 2. Broad Retrieval (Crossref Polite Pool)
        raw_papers = fetch_broad_neighborhood_crossref(core_statement)
        
        # 3. LLM Re-ranking (Dynamic Count)
        final_papers, filtering_log = rerank_and_filter_papers(core_statement, raw_papers)
        
        # 💥 4. 處理絕對無人區的情境
        paper_records = []
        if not final_papers:
            baseline_status = "Absolute Void (絕對無人區：大腦判定周遭毫無可對話之母體)"
            baseline_hex = "000000"
            vote_stats = [0]*6
            paper_records.append("- `[Void]` **大腦過濾宣告**：傳統引擎返回之文獻皆屬雜訊，本理論目前無直接學術鄰域。")
        else:
            baseline_status = f"Crossref Matrix Established (基礎設施母體建構完成：{len(final_papers)} 核心節點)"
            baseline_hex, vote_stats = evaluate_baseline_papers(final_papers, manifest)
            for p in final_papers:
                paper_records.append(f"- `[DOI:{p['id']}]` **{p['title']}** (Cited: {p['citations']})")

        # 5. Offset Calculation
        breakthrough_dims = []
        offset_score = 0
        for i in range(6):
            if user_hex[i] == "1" and baseline_hex[i] == "0":
                breakthrough_dims.append(manifest["dimensions"][list(manifest["dimensions"].keys())[i]]["layer"])
                offset_score += 1
            elif user_hex[i] == "0" and baseline_hex[i] == "1":
                offset_score -= 1
                
        breakthrough_str = "、".join(breakthrough_dims) if breakthrough_dims else "與場域同頻"
        offset_status = f"能勢偏移值 (Offset): {offset_score:+d} (正值為超越母體，負值為受迫於母體)"
        
        # 6. Summary Generation
        client = get_llm_client()
        summary_prompt = f"""
本理論在「外部場域觀測」中測得偏移值為 {offset_score:+d}，拓樸破缺維度：【{breakthrough_str}】。
請根據下文撰寫 200 字理論導讀，客觀描述其在學術生態中的定位(若是無人區請直接指出)。
第一句必須以「本理論架構...」開頭。
"""
        response = call_llm_with_retry(
            client,
            messages=[{"role": "system", "content": summary_prompt}, {"role": "user", "content": raw_text[:3000]}],
            temperature=0.2
        )
        clean_summary = zhconv.convert(response.choices[0].message.content.strip(), 'zh-tw')

        return {
            "user_hex": user_hex,
            "baseline_hex": baseline_hex,
            "breakthrough_str": breakthrough_str,
            "offset_status": offset_status,
            "state_name": user_state_info['name'],
            "dim_logs": dim_logs,
            "summary": clean_summary,
            "full_text": raw_text,
            "meta_data": {
                "core_statement": core_statement,
                "raw_hits": len(raw_papers),
                "final_hits": len(final_papers),
                "filtering_log": filtering_log,
                "paper_records": paper_records,
                "vote_stats": vote_stats,
                "baseline_status": baseline_status,
                "llm_model": LLM_MODEL_NAME
            }
        }
    except Exception as e:
        print(f"工具調用失敗，原因為 處理管線執行異常 ({e})")
        sys.exit(1)

def generate_trajectory_log(target_file, data):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")
    dim_logs_text = "\n    ".join(data['dim_logs'])
    meta = data['meta_data']
    papers_text = "\n".join(meta['paper_records'])
    
    if meta['final_hits'] > 0:
        vote_str = " | ".join([f"Dim{i+1}: {meta['vote_stats'][i]}/{meta['final_hits']}" for i in range(6)])
    else:
        vote_str = "無對照母體，張量坍縮為 0"

    log_output = (
        f"## 📡 AVH 技術觀測日誌：`{target_file}`\n"
        f"* **觀測時間戳 (CST)**：`{timestamp}`\n"
        f"* **高維算力引擎**：`{meta['llm_model']}`\n\n"
        f"---\n"
        f"### 1. 🌌 自然母體建構 (Crossref Natural Matrix)\n"
        f"* **本體核心宣告 (Core Statement)**：`{meta['core_statement']}`\n"
        f"* **場域建構狀態**：`{meta['baseline_status']}` (原始打撈 {meta['raw_hits']} 篇)\n"
        f"* **大腦重排日誌 (Re-ranking Filter)**：_{meta['filtering_log']}_\n"
        f"* **母體核心節點 (True Neighborhood)**：\n"
        f"{papers_text}\n\n"
        f"* **母體張量統計**：`[ {vote_str} ]`\n"
        f"* 🗺️ **母體絕對指紋 (Background Hex)**：`[{data['baseline_hex']}]`\n\n"
        f"### 2. ⚖️ 無人區干涉測量 (No Man's Land Triangulation)\n"
        f"* 🛡️ **本體論絕對指紋 (Ontology Hex)**：`[{data['user_hex']}]` - **{data['state_name']}**\n"
        f"* ⚡ **{data['offset_status']}**\n"
        f"* 🌌 **拓樸破缺維度 (相對於自然母體)**：**【{data['breakthrough_str']}】**\n\n"
        f"**詳細本體測量儀表板**：\n"
        f"    {dim_logs_text}\n\n"
        f"---\n"
        f"> *註：本報告採 V30.2 柔性容錯架構。若大腦判定無同頻母體，則誠實宣告無人區。*\n"
    )
    return log_output

def export_wordpress_html(basename, data):
    html_content = data['full_text'].replace('\n', '<br>')
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta = data['meta_data']
    
    html_output = (
        "<div class=\"avh-hologram-article\">\n"
        "    <div class=\"avh-content\">\n"
        f"        {html_content}\n"
        "    </div>\n"
        "    <hr>\n"
        "    <div class=\"avh-seal\" style=\"border: 2px solid #333; padding: 20px; background: #fafafa; margin-top: 30px;\">\n"
        "        <h3>📡 學術價值全像儀 (AVH) 穩定母體認證</h3>\n"
        f"        <p><strong>理論導讀摘要 (Generated by {meta['llm_model']})：</strong><br>{data['summary']}</p>\n"
        "        <hr>\n"
        f"        <p>場域建構狀態：{meta['baseline_status']}</p>\n"
        f"        <p><strong>{data['offset_status']}</strong></p>\n"
        f"        <p>突破維度：【 {data['breakthrough_str']} 】</p>\n"
        f"        <p>最終本體狀態：[ {data['user_hex']} ] - <strong>{data['state_name']}</strong></p>\n"
        f"        <p>物理時間戳：{timestamp_str}</p>\n"
        "    </div>\n"
        "</div>\n"
    )
    with open(f"WP_Ready_{basename}.html", "w", encoding="utf-8") as f:
        f.write(html_output)

def export_latex(basename, data):
    tex_content = data['full_text'].replace("#", "\\section")
    tex_output = (
        "\\documentclass{article}\n"
        "\\usepackage[utf8]{inputenc}\n"
        "\\usepackage{xeCJK}\n"
        f"\\title{{{basename}}}\n"
        "\\author{Alaric Kuo}\n"
        "\\date{\\today}\n"
        "\\begin{document}\n"
        "\\maketitle\n"
        "\\begin{abstract}\n"
        f"[{data['user_hex']}] {data['state_name']}。\n\n"
        f"{data['offset_status']}\n"
        "\\end{abstract}\n\n"
        f"{tex_content}\n\n"
        "\\end{document}\n"
    )
    with open(f"{basename}_Archive.tex", "w", encoding="utf-8") as f:
        f.write(tex_output)

if __name__ == "__main__":
    if not os.path.exists("avh_manifest.json"):
        print("工具調用失敗，原因為 遺失底層定義檔")
        sys.exit(1)
        
    with open("avh_manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    source_files = [f for f in glob.glob("*.md") if f.lower() not in ["avh_observation_log.md"]]
    if not source_files:
        sys.exit(0)
        
    with open("AVH_OBSERVATION_LOG.md", "w", encoding="utf-8") as log_file:
        log_file.write("# 📡 AVH 學術價值全像儀：V30.2 容錯母體觀測日誌\n---\n")
        last_hex_code = ""
        for target_source in source_files:
            result_data = process_avh_manifestation(target_source, manifest)
            if result_data:
                last_hex_code = result_data['user_hex']
                log_file.write(generate_trajectory_log(target_source, result_data))
                basename = os.path.splitext(target_source)[0]
                export_wordpress_html(basename, result_data)
                export_latex(basename, result_data) 

    if last_hex_code:
        with open(os.environ.get("GITHUB_ENV", "env.tmp"), "a") as env_file:
            env_file.write(f"HEX_CODE={last_hex_code}\n")
