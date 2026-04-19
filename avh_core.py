import os
import sys
import json
import glob
import numpy as np
import networkx as nx
import re
import requests
import xml.etree.ElementTree as ET
import urllib.parse
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import zhconv

# ==============================================================================
# AVH Genesis Engine (V24.4 終極審計版：分艙隔離與觀測樣本透明化)
# ==============================================================================

EMBEDDING_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
LLM_MODEL_NAME = 'openai/gpt-4o'

print(f"🧠 [載入觀測核心] 正在啟動 IQD 物理差分矩陣 ({EMBEDDING_MODEL_NAME})...")
try:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
except Exception as e:
    print(f"工具調用失敗，原因為 本地向量模型載入錯誤 ({e})")
    sys.exit(1)

def call_arxiv_scholar(query, limit=5):
    print(f"🌍 [實體觀測] 轉向 arXiv 資料庫抓取前沿文獻，關鍵字：[{query}]...")
    terms = re.findall(r'[A-Za-z]{4,}', query)[:3]
    if not terms:
        terms = ["System", "Engineering", "Entropy"]

    search_query = "+AND+".join([f"all:{t}" for t in terms])
    encoded_query = urllib.parse.quote(search_query, safe=":+")
    
    url = (
        f"https://export.arxiv.org/api/query?"
        f"search_query={encoded_query}&start=0&max_results={limit}"
        f"&sortBy=submittedDate&sortOrder=descending"
    )
    
    headers = {"User-Agent": "AVH-Hologram/24.4 (GitHub Actions)"}
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        root = ET.fromstring(response.content)
        namespace = {'atom': 'http://www.w3.org/2005/Atom'}
        
        papers = []
        for entry in root.findall('atom:entry', namespace):
            title_node = entry.find('atom:title', namespace)
            abstract_node = entry.find('atom:summary', namespace)
            id_node = entry.find('atom:id', namespace) # 新增：抓取 arXiv ID 供審計
            
            if title_node is None or abstract_node is None:
                continue
                
            title = title_node.text.strip().replace('\n', ' ')
            abstract = abstract_node.text.strip().replace('\n', ' ')
            paper_id = id_node.text.split('/')[-1] if id_node is not None else "Unknown"
            
            papers.append({"id": paper_id, "title": title, "abstract": abstract})
            
        if not papers:
            print("⚠️ [警告] arXiv 未找到相關文獻，將退回預設保守基準。")
            
        return papers, search_query.replace("+AND+", " AND ")

    except Exception as e:
        print(f"工具調用失敗，原因為 arXiv API 遭遇超時或阻擋 ({e})")
        sys.exit(1)

def call_copilot_brain(system_prompt, user_prompt):
    token = os.environ.get("COPILOT_GITHUB_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        print("工具調用失敗，原因為 遺失 GITHUB_TOKEN，無法驗證 GitHub Models")
        sys.exit(1)
        
    print(f"🧠 [雲端大腦] 正在連線 GitHub Models Inference ({LLM_MODEL_NAME})...")
    
    client = OpenAI(
        base_url="https://models.github.ai/inference",
        api_key=token,
    )
    
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model=LLM_MODEL_NAME, 
            temperature=0.2, # 降低溫度，強迫它更客觀
            max_tokens=800
        )
        return zhconv.convert(response.choices[0].message.content.strip(), 'zh-tw')
    except Exception as e:
        print(f"工具調用失敗，原因為 GitHub Models API 呼叫被阻擋或發生錯誤 ({e})")
        sys.exit(1)

def compute_iqd_hex(text_vec, manifest):
    ordered_dimensions = ["value_intent", "governance", "cognition", "architecture", "expansion", "application"]
    hex_code = ""
    dim_logs = []
    
    for key in ordered_dimensions:
        dim = manifest["dimensions"][key]
        v_sin = embedding_model.encode([dim["sin_def"]])[0]
        v_cos = embedding_model.encode([dim["cos_def"]])[0]
        
        sim_sin = float(np.dot(text_vec, v_sin) / (np.linalg.norm(text_vec) * np.linalg.norm(v_sin)))
        sim_cos = float(np.dot(text_vec, v_cos) / (np.linalg.norm(text_vec) * np.linalg.norm(v_cos)))
        
        diff = sim_sin - sim_cos
        bit = "1" if diff > 0.0 else "0" 
        hex_code += bit
        
        winner = "離群突破 (sin)" if bit == "1" else "守成合群 (cos)"
        dim_logs.append(f"* **{dim['layer']}**：{winner} `[Δ Diff: {diff:+.4f}]`")
        
    return hex_code, dim_logs

def process_avh_manifestation(source_path, manifest):
    print(f"\n🌊 [波包掃描] 正在讀取源碼：{source_path}")
    try:
        with open(source_path, 'r', encoding='utf-8') as file:
            raw_text = file.read()
            
        paragraphs = [p.strip() for p in raw_text.split('\n') if len(p.strip()) > 30]
        if len(paragraphs) < 3:
            return None
            
        embeddings = embedding_model.encode(paragraphs)
        sim_matrix = cosine_similarity(embeddings)
        np.fill_diagonal(sim_matrix, 0)
        
        nx_graph = nx.from_numpy_array(sim_matrix)
        scores = nx.pagerank(nx_graph)
        
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        peak_embeddings = [embeddings[i] for i in ranked_indices[:3]]
        psi_peak = np.mean(peak_embeddings, axis=0) 
        
        top_sentence = paragraphs[ranked_indices[0]]

        # 1. 測量絕對指紋
        print("🕸️ [IQD 差分] 計算文本巔峰之物理絕對指紋...")
        user_hex, dim_logs = compute_iqd_hex(psi_peak, manifest)
        user_state_info = manifest["states"].get(user_hex, {"name": "未知狀態", "desc": "缺乏觀測紀錄"})

        # 2. arXiv 真實基準與觀測樣本清單
        keywords = " ".join(re.findall(r'\b[A-Za-z]{4,}\b', top_sentence)[:3]) 
        if not keywords:
            keywords = "System Engineering Entropy"
            
        papers, actual_query = call_arxiv_scholar(keywords)
        baseline_hex_votes = []
        vote_stats = [0]*6 
        paper_records = [] # 新增：存放用於 Log 顯示的文獻清單
        
        if papers:
            for p in papers:
                p_vec = embedding_model.encode([p["abstract"]])[0]
                p_hex, _ = compute_iqd_hex(p_vec, manifest)
                baseline_hex_votes.append(p_hex)
                paper_records.append(f"- `[{p['id']}]` {p['title']}")
                for i in range(6):
                    if p_hex[i] == '1':
                        vote_stats[i] += 1
                
            baseline_hex = ""
            for i in range(6):
                baseline_hex += "1" if vote_stats[i] > (len(papers) / 2) else "0"
        else:
            baseline_hex = "000000"
            actual_query = "Fallback (無網路或無結果)"
            paper_records.append("- (無實體文獻參與觀測)")

        # 3. 雲端大腦：只負責產生「導讀摘要」(保留給 HTML，Log 內只留客觀解讀)
        breakthrough_dims = [manifest["dimensions"][list(manifest["dimensions"].keys())[i]]["layer"] 
                             for i in range(6) if user_hex[i] == "1" and baseline_hex[i] == "0"]
        breakthrough_str = "、".join(breakthrough_dims) if breakthrough_dims else "與傳統基準同頻"
        
        sys_prompt = f"""
你是一位專注於系統工程與物理本體論的客觀解讀者。
本理論經過 arXiv 實體文獻對比，已確認在以下維度突破傳統工程框架：【{breakthrough_str}】。
理論狀態定位為：{user_state_info['name']}。

請根據下文撰寫約 200 字的「理論導讀摘要」。
要求：
1. 語氣客觀，禁止使用任何宣傳式修辭（如革新、顛覆、偉大）。
2. 專注解釋理論核心如何運作。
3. 第一句話必須以「本理論架構...」開頭。
"""
        clean_summary = call_copilot_brain(sys_prompt, raw_text[:3000])

        return {
            "user_hex": user_hex,
            "baseline_hex": baseline_hex,
            "breakthrough_str": breakthrough_str,
            "state_name": user_state_info['name'],
            "dim_logs": dim_logs,
            "summary": clean_summary,
            "full_text": raw_text,
            "meta_data": {
                "top_sentence": top_sentence[:100] + "..." if len(top_sentence) > 100 else top_sentence,
                "arxiv_query": actual_query,
                "papers_hit": len(papers) if papers else 0,
                "paper_records": paper_records,
                "vote_stats": vote_stats,
                "embedding_model": EMBEDDING_MODEL_NAME,
                "llm_model": LLM_MODEL_NAME
            }
        }
    except Exception as e:
        print(f"工具調用失敗，原因為 處理管線發生未預期錯誤 ({e})")
        sys.exit(1)

def generate_trajectory_log(target_file, data):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")
    dim_logs_text = "\n    ".join(data['dim_logs'])
    meta = data['meta_data']
    papers_text = "\n".join(meta['paper_records'])
    vote_str = " | ".join([f"Dim{i+1}: {meta['vote_stats'][i]}/{meta['papers_hit']}" for i in range(6)])

    # 【核心改動】：拔除 LLM 旁白，補上可審計的 arXiv 論文清單
    log_output = (
        f"## 📡 AVH 技術觀測日誌：`{target_file}`\n"
        f"* **觀測時間戳 (CST)**：`{timestamp}`\n"
        f"* **觀測儀器 (Vector)**：`{meta['embedding_model']}`\n\n"
        f"---\n"
        f"### 1. 🔬 觀測參數與基準建立 (Observation Parameters)\n"
        f"* **原著核心波包 (Peak Sentence)**：\n  > *\"{meta['top_sentence']}\"*\n"
        f"* **arXiv 檢索條件 (Query)**：`{meta['arxiv_query']}`\n"
        f"* **arXiv 基準樣本 (Hits: {meta['papers_hit']})**：\n"
        f"{papers_text}\n\n"
        f"* **基準矩陣投票統計**：`[ {vote_str} ]`\n"
        f"* 🗺️ **外部真實基準 (Baseline Hex)**：`[{data['baseline_hex']}]`\n\n"
        f"### 2. ⚖️ IQD 差分干涉結果 (Interference Results)\n"
        f"* 🛡️ **本體論絕對指紋**：`[{data['user_hex']}]` - **{data['state_name']}**\n"
        f"* 🌌 **拓樸破缺維度 (相對於 arXiv)**：**【{data['breakthrough_str']}】**\n\n"
        f"**詳細差分儀表板**：\n"
        f"    {dim_logs_text}\n\n"
        f"---\n"
        f"> *註：本報告為系統自動化向量觀測紀錄，不包含生成式 AI 之主觀詮釋。*\n"
    )
    return log_output

def export_wordpress_html(basename, data):
    html_content = data['full_text'].replace('\n', '<br>')
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta = data['meta_data']
    
    # 【核心改動】：將 LLM 的「導讀摘要」專門放在這裡，做對外展示用
    html_output = (
        "<div class=\"avh-hologram-article\">\n"
        "    <div class=\"avh-content\">\n"
        f"        {html_content}\n"
        "    </div>\n"
        "    <hr>\n"
        "    <div class=\"avh-seal\" style=\"border: 2px solid #333; padding: 20px; background: #fafafa; margin-top: 30px;\">\n"
        "        <h3>📡 學術價值全像儀 (AVH) 實體觀測認證</h3>\n"
        f"        <p><strong>理論導讀摘要 (Generated by {meta['llm_model']})：</strong><br>{data['summary']}</p>\n"
        "        <hr>\n"
        f"        <p>觀測基準：arXiv 前沿文獻 (樣本數：{meta['papers_hit']})</p>\n"
        f"        <p>突破維度：【 {data['breakthrough_str']} 】</p>\n"
        f"        <p>最終演化狀態：[ {data['user_hex']} ] - <strong>{data['state_name']}</strong></p>\n"
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
        f"突破傳統維度：【{data['breakthrough_str']}】\n"
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
        log_file.write("# 📡 AVH 學術價值全像儀：純淨物理觀測日誌\n---\n")
        
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
