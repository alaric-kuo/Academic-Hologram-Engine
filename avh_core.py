import os
import sys
import json
import glob
import re
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer

# ==============================================================================
# AVH Genesis Engine (V4.0.0 純粹自我定位版)
# ==============================================================================

print("🧠 [載入核心] 正在啟動開源神經網路模型 (all-MiniLM-L6-v2)...")
try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"模型載入失敗：{str(e)}")
    sys.exit(1)

def extract_ontological_trajectory(source_path):
    print(f"🌊 [波包坍縮] 正在讀取源碼：{source_path}")
    try:
        with open(source_path, 'r', encoding='utf-8') as file:
            raw_text = file.read()
            full_text = " ".join(raw_text.split())
            
        # 🛡️ [絕對裝甲啟動] 正則表達式截斷 (Regex Truncation)
        # 只要偵測到「第五章」與「64」的組合，或者直接看到「[000000]」的陣列開頭，直接物理斬斷！
        cutoff_patterns = [
            r"第五章.*?64.*?實相",
            r"\[000000\] 絕對剛性基石",
            r"## 附錄",
            r"\[AVH-IGNORE\]"
        ]
        
        for pattern in cutoff_patterns:
            match = re.search(pattern, full_text)
            if match:
                full_text = full_text[:match.start()]
                print(f"🛡️ [裝甲啟動] 成功辨識字典邊界特徵 '{pattern}'，已徹底斬除後方雜訊。")
                break
                
    except Exception as e:
        print(f"檔案讀取異常，跳過此文件 ({str(e)})")
        return None
        
    if len(full_text) < 300:
        print(f"⚠️ {source_path} 淨化後有效文本資訊熵過低，忽略觀測。")
        return None
    
    window_size, stride = 1500, 800
    trajectories = [full_text[i:i+window_size] for i in range(0, len(full_text), stride) if len(full_text[i:i+window_size]) > 50]
    
    try:
        wave_functions = [embedding_model.encode([chunk])[0] for chunk in trajectories]
        
        # 🛡️ 幾何中位數離群值剔除
        median_center = np.median(wave_functions, axis=0)
        distances = [np.linalg.norm(wf - median_center) for wf in wave_functions]
        threshold = np.percentile(distances, 85)
        cohesive_wfs = [wf for wf, d in zip(wave_functions, distances) if d <= threshold]
        
        psi_global = np.mean(cohesive_wfs, axis=0)
        
        print(f"🛡️ [拓樸過濾] 已移除 {len(wave_functions) - len(cohesive_wfs)} 個異常離群視窗。")
        
        vec_stats = {
            "dim": len(psi_global),
            "mean": float(np.mean(psi_global)),
            "std": float(np.std(psi_global)),
            "norm": float(np.linalg.norm(psi_global))
        }
        
        similarities = [np.dot(wf, psi_global) / (np.linalg.norm(wf) * np.linalg.norm(psi_global)) for wf in wave_functions]
        centroid_index = np.argmax(similarities)
        semantic_probe = trajectories[centroid_index][:200]
        
        return {
            "psi_global": psi_global,
            "vec_stats": vec_stats,
            "probe_text": semantic_probe,
            "window_count": len(cohesive_wfs),
            "centroid_sim": float(similarities[centroid_index]),
            "full_text": raw_text
        }
    except Exception as e:
        print(f"工具調用失敗，原因為 向量轉換過程錯誤 ({str(e)})")
        sys.exit(1)

def generate_trajectory_log(target_file, trajectory_data, hex_code, manifest):
    """拔除外部碰撞廢物，回歸純粹的自我演化軌跡紀錄"""
    hex_info = manifest['states'].get(hex_code, {"name": "未定義拓樸", "desc": "未知路徑。"})
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")
    stats = trajectory_data['vec_stats']
    
    return f"""
## 📡 觀測軌跡：`{target_file}`
* **物理時間戳**：`{timestamp}`

### 1. 🌌 全文能勢集成 (Wave Function Integration)
* **淨化後窗格數**：`{trajectory_data['window_count']} 視窗 (1500/800 Overlap)`
* **384維重心矩陣特徵** (本體論防禦啟動)：
    * `均值 (Mean)`：{stats['mean']:.8f}
    * `標準差 (Std)`：{stats['std']:.8f}
    * `模長 (L2 Norm)`：{stats['norm']:.8f}

### 2. 🎯 語意重心提取 (Semantic Centroid Probe)
* **重心凝聚度**：`{trajectory_data['centroid_sim']:.4f}`
* **理論核心探針**：
    > "{trajectory_data['probe_text']}..."

### 3. 🧬 最終狀態坍縮 (Topological Collapse)
* **狀態陣列**：`[{hex_code}]`
* **物理相變**：**{hex_info['name']}**
* **學術指紋**：
    > {hex_info['desc']}

---
"""

def export_wordpress_html(basename, content, hex_code, state_name):
    html_content = content.replace('\n', '<br>')
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html_template = f"""
<div class="avh-hologram-article">
    <div class="avh-content">
        {html_content}
    </div>
    <hr>
    <div class="avh-seal" style="border: 2px solid #333; padding: 20px; background: #fafafa; margin-top: 30px;">
        <p><strong>📡 本理論已通過 學術價值全像儀 (AVH) 認證</strong></p>
        <p>當下演化狀態：[ {hex_code} ] - <strong>{state_name}</strong></p>
        <p>語意時間戳：{timestamp_str}</p>
        <p><em>本體論底層協議保護 | 純粹自我定位矩陣 | AJ Consulting</em></p>
    </div>
</div>
"""
    with open(f'WP_Ready_{basename}.html', 'w', encoding='utf-8') as f:
        f.write(html_template)

def export_latex(basename, content, hex_code, state_name):
    tex_content = content.replace('#', '\\section')
    tex_template = (
        "\\documentclass{article}\n"
        "\\usepackage[utf8]{inputenc}\n"
        "\\usepackage{xeCJK}\n"
        "\\title{" + basename + "}\n"
        "\\author{Alaric Kuo}\n"
        "\\date{\\today}\n"
        "\\begin{document}\n"
        "\\maketitle\n"
        "\\begin{abstract}\n"
        "本文章經由 AVH 學術價值全像儀觀測，當下演化狀態為 [" + hex_code + "] " + state_name + "。\n"
        "\\end{abstract}\n\n"
        + tex_content + "\n\n"
        "\\end{document}\n"
    )
    with open(f'{basename}_Archive.tex', 'w', encoding='utf-8') as f:
        f.write(tex_template)

if __name__ == "__main__":
    if not os.path.exists('avh_manifest.json'):
        print("工具調用失敗，原因為 遺失底層定義檔 (avh_manifest.json)")
        sys.exit(1)
        
    with open('avh_manifest.json', 'r', encoding='utf-8') as f:
        manifest = json.load(f)
        
    source_files = []
    for ext in ["*.md", "*.tex"]:
        source_files.extend([f for f in glob.glob(ext) if f.lower() not in ['avh_observation_log.md']])
    
    if not source_files:
        print("系統休眠：未偵測到有效理論源碼波包。")
        sys.exit(0)
        
    print(f"\n🚀 啟動 AVH 引擎 (純粹自我定位模式)，共偵測到 {len(source_files)} 個波包等待坍縮...")
    
    with open('AVH_OBSERVATION_LOG.md', 'w', encoding='utf-8') as log_file:
        log_file.write("# 📡 AVH 學術價值全像儀：純粹自我定位軌跡\n")
        log_file.write("*本文件詳實紀錄方法論實作過程中，知識波包從高維向量到三維投影的每一處相變。我們拒絕外部網格的無效引用評估，將觀測權利絕對收斂於本體。*\n\n---\n")
        
        last_hex_code = ""
        for target_source in source_files:
            trajectory_data = extract_ontological_trajectory(target_source)
            if not trajectory_data: continue
            
            ordered_dimensions = ["value_intent", "governance", "cognition", "architecture", "expansion", "application"]
            hex_bits = ""
            psi = trajectory_data['psi_global']
            
            for key in ordered_dimensions:
                dim = manifest['dimensions'][key]
                v_pos = embedding_model.encode([dim['pos_def']])[0]
                v_neg = embedding_model.encode([dim['neg_def']])[0]
                
                sim_pos = np.dot(psi, v_pos) / (np.linalg.norm(psi) * np.linalg.norm(v_pos))
                sim_neg = np.dot(psi, v_neg) / (np.linalg.norm(psi) * np.linalg.norm(v_neg))
                hex_bits += "1" if sim_pos > sim_neg else "0"
            
            last_hex_code = hex_bits
            state_name = manifest['states'][hex_bits]['name']
            
            # 拔除 bg_energy 與 collisions 的呼叫，直接生成日誌
            report = generate_trajectory_log(target_source, trajectory_data, hex_bits, manifest)
            log_file.write(report)
            
            basename = os.path.splitext(target_source)[0]
            export_wordpress_html(basename, trajectory_data['full_text'], hex_bits, state_name)
            export_latex(basename, trajectory_data['full_text'], hex_bits, state_name)
            
            print(f"✅ {target_source} 理論收斂完成！ [{hex_bits}]")

    if last_hex_code:
        with open(os.environ.get('GITHUB_ENV', 'env.tmp'), 'a') as env_file:
            env_file.write(f"HEX_CODE={last_hex_code}\n")
