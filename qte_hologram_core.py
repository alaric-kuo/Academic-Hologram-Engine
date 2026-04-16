import os
import sys
import json
import glob
import numpy as np
import requests
from openai import OpenAI
from datetime import datetime

# ==============================================================================
# QTE Academic Hologram Core Engine (嚴謹本體論版)
# ==============================================================================

# 1. 算力通道掛載
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
S2_API_KEY = os.environ.get("S2_API_KEY", "")

def get_embedding(text):
    try:
        response = client.embeddings.create(input=[text], model="text-embedding-3-small", timeout=15)
        return np.array(response.data[0].embedding)
    except Exception as e:
        print(f"工具調用失敗，原因為 OpenAI API 拒絕連線或超時狀態 ({str(e)})")
        sys.exit(1)

def extract_continuous_fingerprint(source_path):
    """【純文本拓樸連續積分】揚棄 PDF 視覺排版雜訊，直指資訊熵本體"""
    print(f"🌊 [波包坍縮] 正在讀取源碼：{source_path}")
    try:
        with open(source_path, 'r', encoding='utf-8') as file:
            full_text = " ".join(file.read().split())
    except Exception as e:
        print(f"工具調用失敗，原因為 原始碼讀取異常 ({str(e)})")
        sys.exit(1)
        
    # 【第一層防禦】最小有效質量檢驗
    if len(full_text) < 500:
        print("工具調用失敗，原因為 文本資訊熵過低 (內容過短或無效檔案)")
        sys.exit(1)
    
    # 拓樸重疊視窗 (Overlap Windows) - 保持語意連續演化
    window_size, stride = 1500, 1000
    trajectories = [full_text[i:i+window_size] for i in range(0, len(full_text), stride) if len(full_text[i:i+window_size]) > 100]
    
    print(f"🌌 [空間展開] 文本映射為 {len(trajectories)} 個連續相位，正在生成希爾伯特軌跡...")
    try:
        response = client.embeddings.create(input=trajectories, model="text-embedding-3-small", timeout=30)
        wave_functions = [np.array(data.embedding) for data in response.data]
        return full_text[:200], np.mean(wave_functions, axis=0) # 回傳探針與全局波包
    except Exception as e:
        print(f"工具調用失敗，原因為 向量轉換過程超時或遭到阻擋 ({str(e)})")
        sys.exit(1)

def scan_background_field(query_text):
    """【背景能勢探測】掛載嚴格超時與阻擋防禦"""
    print("📡 [場域探測] 正在掃描全球學術網格的既有能勢分佈...")
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    headers = {"x-api-key": S2_API_KEY} if S2_API_KEY else {}
    params = {"query": query_text[:100], "limit": 10, "fields": "citationCount"}
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"工具調用失敗，原因為 Semantic Scholar API 連線超時或阻擋 ({str(e)})")
        sys.exit(1)
        
    return sum(p.get('citationCount', 0) for p in response.json().get('data', []))

def validate_semantic_coherence(psi_global):
    """【第二層防禦】資訊熵濾網：檢驗文本是否邏輯崩潰"""
    baseline_text = "Academic research paper, theoretical framework, engineering application, scientific methodology, ontology, topology."
    baseline_vector = get_embedding(baseline_text)
    sim = np.dot(psi_global, baseline_vector) / (np.linalg.norm(psi_global) * np.linalg.norm(baseline_vector))
    
    if sim < 0.15: 
        print(f"工具調用失敗，原因為 文本邏輯崩潰或不具備基礎學理語意 (相似度 {sim:.3f} 低於極限閾值)")
        sys.exit(1)

def calculate_hexagram(psi_global, manifest):
    """【多維正交投影】計算學術六爻"""
    validate_semantic_coherence(psi_global)
    hex_bits = ""
    for i in range(5, -1, -1):
        dim = manifest['dimensions'][list(manifest['dimensions'].keys())[5-i]]
        v_pos = get_embedding(dim['pos_def'])
        v_neg = get_embedding(dim['neg_def'])
        
        sim_pos = np.dot(psi_global, v_pos) / (np.linalg.norm(psi_global) * np.linalg.norm(v_pos))
        sim_neg = np.dot(psi_global, v_neg) / (np.linalg.norm(psi_global) * np.linalg.norm(v_neg))
        hex_bits += "1" if sim_pos > sim_neg else "0"
    return hex_bits

def generate_hologram_report(target_file, hex_code, energy, manifest):
    """【全息顯化】生成 README 報告"""
    hex_info = manifest['states'].get(hex_code, {"name": "未定義拓樸", "desc": "系統偵測到未知演化路徑，資訊熵溢出。"})
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")
    
    return f"""
## 📡 QTE 學術價值全像儀 觀測報告
**源碼載體：** `{target_file}` | **語意時間戳：** `{timestamp}`

### 🌌 本體論六爻投影 (Hexagram Projection)
> **狀態陣列：`[{hex_code}]`** > **物理相變：{hex_info['name']}**
> **全局能勢：{energy}** (背景引用質量總和)

### 🧬 QTE 演化軌跡判讀
**{hex_info['desc']}**

### 📐 拓樸維度解析 (天地人)
* **[天] 價值意圖與治理維度:** `[{hex_code[0]}, {hex_code[1]}]`
* **[人] 認知深度與描述架構:** `[{hex_code[2]}, {hex_code[3]}]`
* **[地] 擴張潛力與應用實相:** `[{hex_code[4]}, {hex_code[5]}]`

---
*本 Repository 由 QTE 底層協議自動守護。每一次 Commit 皆伴隨純文本波包坍縮、自動 PDF 封存與全息狀態紀錄。*
"""

if __name__ == "__main__":
    if not os.path.exists('qte_academic_manifest.json'):
        print("工具調用失敗，原因為 遺失底層定義檔 (qte_academic_manifest.json)")
        sys.exit(1)
        
    with open('qte_academic_manifest.json', 'r', encoding='utf-8') as f:
        manifest = json.load(f)
        
    # 尋找目標源碼 (排除 README.md)
    md_files = [f for f in glob.glob("*.md") if f.lower() != 'readme.md']
    if not md_files:
        print("系統休眠：未偵測到 Markdown 源碼波包。")
        sys.exit(0)
        
    target_source = md_files[0] 
    
    print(f"\n🚀 啟動 QTE 引擎，鎖定目標: {target_source}")
    probe_text, psi = extract_continuous_fingerprint(target_source)
    bg_energy = scan_background_field(probe_text)
    
    print("⚖️ 正在進行狀態坍縮與資訊熵檢核...")
    hex_code = calculate_hexagram(psi, manifest)
    report = generate_hologram_report(target_source, hex_code, bg_energy, manifest)
    
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write("# 瀚菱顧問 (AJ Consulting) - 信任系統底層協議\n\n")
        f.write(report)
    
    # 輸出一個讓 GitHub Actions 抓取的環境變數，告知目標檔案名稱
    with open(os.environ['GITHUB_ENV'], 'a') as env_file:
        env_file.write(f"TARGET_MD={target_source}\n")
        env_file.write(f"HEX_CODE={hex_code}\n")
    
    print("✅ README.md 物理投影完成！")
