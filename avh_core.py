import os
import sys
import json
import glob
import numpy as np
import networkx as nx
import re
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import zhconv

# ==============================================================================
# AVH Genesis Engine (V16.0 霸權回歸：AI 自鎖顯化與全數值儀表板版)
# ==============================================================================

print("🧠 [載入觀測核心] 正在啟動多語系拓樸網路 (paraphrase-multilingual-MiniLM)...")
try:
    embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
except Exception as e:
    print("模型載入失敗：" + str(e))
    sys.exit(1)

print("✨ [載入造物核心] 正在喚醒具備全文識讀能力之 LLM (Qwen2.5-0.5B-Instruct)...")
try:
    llm_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    llm_model = AutoModelForCausalLM.from_pretrained(llm_name, torch_dtype="auto")
except Exception as e:
    print("生成大腦載入失敗：" + str(e))
    sys.exit(1)

def ask_llm(system_prompt, user_prompt, max_tokens=800, temp=0.3):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(llm_model.device)
    
    generated_ids = llm_model.generate(
        model_inputs.input_ids,
        max_new_tokens=max_tokens,
        temperature=temp,
        repetition_penalty=1.15
    )
    
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    raw_response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return zhconv.convert(raw_response, 'zh-tw')

def calculate_euler_parameters(text, manifest):
    """計算並回傳華麗的數學物理參數儀表板 (僅供觀測，不干涉 AI 決定)"""
    paragraphs = [p.strip() for p in text.split('\n') if len(p.strip()) > 30]
    if len(paragraphs) < 3:
        return None
        
    embeddings = embedding_model.encode(paragraphs)
    sim_matrix = cosine_similarity(embeddings)
    np.fill_diagonal(sim_matrix, 0)
    
    nx_graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(nx_graph)
    
    # 計算全域矩陣特徵
    psi_global = np.mean(embeddings, axis=0)
    vec_stats = {
        "node_count": len(paragraphs),
        "mean": float(np.mean(psi_global)),
        "std": float(np.std(psi_global)),
        "norm": float(np.linalg.norm(psi_global))
    }
    
    # 計算尤拉顯化數值 (sin / cos 引力)
    ordered_dimensions = ["value_intent", "governance", "cognition", "architecture", "expansion", "application"]
    dim_logs = []
    
    for key in ordered_dimensions:
        dim = manifest["dimensions"][key]
        v_sin = embedding_model.encode([dim["sin_def"]])[0]
        v_cos = embedding_model.encode([dim["cos_def"]])[0]
        
        sim_sin = float(np.dot(psi_global, v_sin) / (np.linalg.norm(psi_global) * np.linalg.norm(v_sin)))
        sim_cos = float(np.dot(psi_global, v_cos) / (np.linalg.norm(psi_global) * np.linalg.norm(v_cos)))
        
        dim_logs.append(f"* **{dim['layer']}**：`[sin: {sim_sin:+.4f} | cos: {sim_cos:+.4f}]`")
        
    return vec_stats, dim_logs

def process_avh_manifestation(source_path, manifest):
    print(f"\n🌊 [波包掃描] 正在讀取源碼：{source_path}")
    try:
        with open(source_path, 'r', encoding='utf-8') as file:
            raw_text = file.read()
            
        if len(raw_text) < 100:
            print(f"⚠️ {source_path} 文本過短。")
            return None
            
        # ---------------------------------------------------------
        # 步驟 1：取得尤拉數學參數 (還原你最愛的版面數值)
        # ---------------------------------------------------------
        print("🕸️ [矩陣測量] 正在計算尤拉相位數值與原波包特徵...")
        math_data = calculate_euler_parameters(raw_text, manifest)
        if not math_data:
            return None
        vec_stats, dim_logs = math_data

        # ---------------------------------------------------------
        # 步驟 2：AI 第一次讀取 -> 產出學術指紋 (上上版的最強邏輯)
        # ---------------------------------------------------------
        print("👁️ [脈絡識讀] 由 AI 閱讀全文並判定高維價值指紋...")
        dimension_prompts = ""
        ordered_keys = ["value_intent", "governance", "cognition", "architecture", "expansion", "application"]
        for idx, key in enumerate(ordered_keys):
            dim = manifest["dimensions"][key]
            dimension_prompts += f"維度 {idx+1} ({dim['layer']}):\n - [1] 突破: {dim['sin_def']}\n - [0] 守成: {dim['cos_def']}\n\n"
            
        eval_sys_prompt = (
            "你是一個高維度觀測儀。閱讀全文，根據以下六個維度判定文章屬性。\n"
            f"{dimension_prompts}"
            "【絕對指令】：請在回覆中包含一組 6 個數字的代碼（只包含 0 或 1），例如 111111。"
        )
        
        raw_source_hex = ask_llm(eval_sys_prompt, f"請判定文本的 6 位元指紋：\n\n{raw_text[:3000]}", max_tokens=30, temp=0.1)
        
        # 暴力正則提取，保證絕不出現 000000 崩潰
        match = re.search(r'[01]{6}', raw_source_hex)
        source_hex = match.group(0) if match else "000000"
        state_info = manifest["states"].get(source_hex, {"name": "未知狀態", "desc": "缺乏觀測紀錄"})
        print(f"✅ AI 識讀指紋: [{source_hex}] - {state_info['name']}")

        # ---------------------------------------------------------
        # 步驟 3：狀態順向自鎖 -> 顯化摘要
        # ---------------------------------------------------------
        print(f"🛡️ [自鎖顯化] 注入天命評語，強制 AI 進行方向性論述...")
        summary_sys_prompt = f"""
你是一個精準的學術本體論大腦。
本系統已經判定這篇理論的學術指紋為：[{source_hex}] - {state_info['name']}。
核心精神評語：「{state_info['desc']}」

【絕對任務】：
閱讀使用者的全文，寫出一段氣勢磅礴的摘要。
你必須在摘要中，精準展現上述「核心精神評語」所描述的價值與方向。
【格式警告】：直接輸出段落文字。嚴禁使用 Markdown 分隔線、嚴禁產生條列式清單。
論述完畢請強制輸出『[顯化完畢]』。
"""
        generated_summary = ask_llm(summary_sys_prompt, f"請在指紋鎖定下進行思想顯化：\n\n{raw_text[:3000]}", max_tokens=900, temp=0.35)
        
        # 清理結尾標記與可能殘留的 Markdown 亂碼
        if "[顯化完畢]" in generated_summary:
            generated_summary = generated_summary.split("[顯化完畢]")[0]
        elif "顯化完畢" in generated_summary:
            generated_summary = generated_summary.split("顯化完畢")[0]
            
        clean_summary = re.sub(r'^[#*\-\s]+', '', generated_summary)
        clean_summary = re.sub(r'[#*\-\s]+$', '', clean_summary).strip()
        print("✅ [顯化完成] AI 已成功在自鎖狀態下完成論述！")

        return {
            "hex_code": source_hex,
            "state_name": state_info['name'],
            "state_desc": state_info['desc'],
            "vec_stats": vec_stats,
            "dim_logs": dim_logs,
            "summary": clean_summary,
            "full_text": raw_text
        }
    except Exception as e:
        print(f"工具調用失敗，原因為 邏輯顯化過程錯誤 ({e})")
        sys.exit(1)

def generate_trajectory_log(target_file, data):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")
    dim_logs_text = "\n    ".join(data['dim_logs'])
    
    log_output = (
        f"## 📡 演化顯化軌跡：`{target_file}`\n"
        f"* **物理時間戳**：`{timestamp}`\n\n"
        f"### 1. 🧠 核心邏輯拓樸萃取 (Semantic Graph Abstraction)\n"
        f"* **邏輯節點數**：從全文提煉出 `{data['vec_stats']['node_count']}` 個核心邏輯段落。\n"
        f"* **原文矩陣特徵**：\n"
        f"    * `均值 (Mean)`：{data['vec_stats']['mean']:.8f}\n"
        f"    * `標準差 (Std)`：{data['vec_stats']['std']:.8f}\n"
        f"    * `模長 (L2 Norm)`：{data['vec_stats']['norm']:.8f}\n\n"
        f"### 2. 🧬 尤拉相位顯化 (Euler Phase Manifestation)\n"
        f"* **系統說明**：系統透過高維神經網絡進行全脈絡識讀，並輔以數學矩陣計算底層尤拉相位的引力數值。\n"
        f"* **狀態張量**：`[{data['hex_code']}]` (由 AI 識讀定錨)\n"
        f"* **物理相變**：**{data['state_name']}**\n"
        f"* **尤拉數值觀測**：\n"
        f"    {dim_logs_text}\n"
        f"* **學術指紋評語**：\n"
        f"    > {data['state_desc']}\n\n"
        f"---\n"
        f"### 3. 🎯 狀態自鎖顯化摘要 (Auto-Locked Synthesis)\n"
        f"*(系統將上述測得之絕對指紋化為「鋼鐵模具」，強制約束 AI 進行全脈絡的摘要生成，確保論述與高維價值完美接地。)*\n\n"
        f"> **{data['summary']}**\n\n"
        f"---\n"
    )
    return log_output

def export_wordpress_html(basename, data):
    html_content = data['full_text'].replace('\n', '<br>')
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    html_output = (
        "<div class=\"avh-hologram-article\">\n"
        "    <div class=\"avh-content\">\n"
        f"        {html_content}\n"
        "    </div>\n"
        "    <hr>\n"
        "    <div class=\"avh-seal\" style=\"border: 2px solid #333; padding: 20px; background: #fafafa; margin-top: 30px;\">\n"
        "        <p><strong>📡 本理論已完成 學術價值全像儀 (AVH) 狀態自鎖顯化</strong></p>\n"
        f"        <p>當下演化狀態：[ {data['hex_code']} ] - <strong>{data['state_name']}</strong></p>\n"
        f"        <p>物理時間戳：{timestamp_str}</p>\n"
        "        <p><em>V16.0 霸權回歸協議 | 本體論底層保護 | AJ Consulting</em></p>\n"
        "    </div>\n"
        "</div>\n"
    )
    with open("WP_Ready_" + basename + ".html", "w", encoding="utf-8") as f:
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
        f"本文章經由 AVH 學術價值全像儀觀測，當下演化狀態顯化為 [{data['hex_code']}] {data['state_name']}。\n\n"
        f"{data['summary'][:200]}...\n"
        "\\end{abstract}\n\n"
        f"{tex_content}\n\n"
        "\\end{document}\n"
    )
    with open(basename + "_Archive.tex", "w", encoding="utf-8") as f:
        f.write(tex_output)

if __name__ == "__main__":
    if not os.path.exists("avh_manifest.json"):
        print("工具調用失敗，原因為 遺失底層定義檔 (avh_manifest.json)")
        sys.exit(1)
        
    with open("avh_manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    source_files = [f for f in glob.glob("*.md") if f.lower() not in ["avh_observation_log.md"]]
    
    if not source_files:
        print("系統休眠：未偵測到有效理論源碼波包。")
        sys.exit(0)
        
    print(f"\n🚀 啟動 AVH 造物引擎 (V16.0 霸權回歸版)，共偵測到 {len(source_files)} 個波包等待觀測...")
    
    with open("AVH_OBSERVATION_LOG.md", "w", encoding="utf-8") as log_file:
        log_file.write("# 📡 AVH 學術價值全像儀：本體論顯化軌跡\n")
        log_file.write("*本文件詳實紀錄系統如何利用 AI 進行高維度脈絡識讀定錨，並透過底層尤拉矩陣計算引力數值。系統最終將『學術指紋』轉化為絕對約束枷鎖，強制 AI 產出具備高維度脈絡且邏輯接地的完美摘要。*\n\n---\n")
        
        last_hex_code = ""
        for target_source in source_files:
            result_data = process_avh_manifestation(target_source, manifest)
            if result_data:
                last_hex_code = result_data['hex_code']
                report = generate_trajectory_log(target_source, result_data)
                log_file.write(report)
                
                basename = os.path.splitext(target_source)[0]
                export_wordpress_html(basename, result_data)
                export_latex(basename, result_data)

    if last_hex_code:
        with open(os.environ.get("GITHUB_ENV", "env.tmp"), "a") as env_file:
            env_file.write(f"HEX_CODE={last_hex_code}\n")
