import os
import sys
import json
import glob
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from sentence_transformers import SentenceTransformer
import zhconv

# ==============================================================================
# AVH Genesis Engine (V7.1.0 絕對寂靜：包立不相容・原波包多樣性版)
# ==============================================================================

print("🧠 [載入觀測核心] 正在啟動多語系拓樸網路 (paraphrase-multilingual-MiniLM)...")
try:
    embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
except Exception as e:
    print("模型載入失敗：" + str(e))
    sys.exit(1)

def extract_pure_ontology(source_path):
    print("🌊 [波包顯化] 正在讀取源碼：" + source_path)
    try:
        with open(source_path, 'r', encoding='utf-8') as file:
            raw_text = file.read()
            
        paragraphs = [p.strip() for p in raw_text.split('\n') if len(p.strip()) > 30]
        if len(paragraphs) < 3:
            print("⚠️ " + source_path + " 文本結構過於單一，資訊熵不足。")
            return None
            
        print("🕸️ [邏輯建構] 正在建立語意拓樸網格並計算引力權重...")
        embeddings = embedding_model.encode(paragraphs)
        
        sim_matrix = cosine_similarity(embeddings)
        np.fill_diagonal(sim_matrix, 0)
        
        # 使用圖論計算各段落的 PageRank 分數，作為「語意引力」的指標
        nx_graph = nx.from_numpy_array(sim_matrix)
        scores = nx.pagerank(nx_graph)
        
        # 排序段落，抓取引力最強的節點
        ranked_paragraphs = sorted(((scores[i], paragraphs[i], embeddings[i], i) for i in range(len(paragraphs))), reverse=True)
        
        core_size = max(3, min(5, int(len(paragraphs) * 0.35)))
        core_fragments = []
        
        print("🌌 [相容性檢查] 啟動包立不相容原理，排除語意塌陷黑洞...")
        # [V7.1.0 包立不相容原理 (Pauli Exclusion Principle)]
        # 確保萃取出的波包具有多樣性，不掉入同義反覆的字典黑洞
        for item in ranked_paragraphs:
            if len(core_fragments) >= core_size:
                break
            
            is_duplicate = False
            for selected in core_fragments:
                # 計算與已選波包的餘弦相似度
                sim = np.dot(item[2], selected[2]) / (np.linalg.norm(item[2]) * np.linalg.norm(selected[2]))
                if sim > 0.85:  # 相似度 > 85% 視為同一量子態，觸發物理斥力，丟棄！
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                core_fragments.append(item)
                
        # 萬一斥力太強導致碎片不足，無條件補齊至最低限度 3 段
        if len(core_fragments) < 3:
            for item in ranked_paragraphs:
                if len(core_fragments) >= 3:
                    break
                if item not in core_fragments:
                    core_fragments.append(item)
        
        # 依照原始順序重新排列萃取出的原話
        core_fragments_sorted = sorted(core_fragments, key=lambda x: x[3])
        extracted_pure_text = "\n\n".join([f"> {item[1]}" for item in core_fragments_sorted])
        
        # 計算核心意志的平均向量 (Will-Manifestation Vector)
        psi_global = np.mean([item[2] for item in core_fragments], axis=0)
        
        print("🛡️ [意志顯化] 已成功透過圖論萃取具備全域視野的純粹原波包。")
        
        vec_stats = {
            "dim": len(psi_global),
            "mean": float(np.mean(psi_global)),
            "std": float(np.std(psi_global)),
            "norm": float(np.linalg.norm(psi_global))
        }
        
        return {
            "psi_global": psi_global,
            "vec_stats": vec_stats,
            "pure_fragments": zhconv.convert(extracted_pure_text, 'zh-tw'),
            "fragment_count": len(core_fragments),
            "full_text": raw_text
        }
    except Exception as e:
        print("工具調用失敗，原因為 邏輯顯化過程錯誤 (" + str(e) + ")")
        sys.exit(1)

def generate_trajectory_log(target_file, trajectory_data, hex_code, manifest):
    hex_info = manifest['states'].get(hex_code, {"name": "未定義拓樸", "desc": "未知路徑。"})
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")
    stats = trajectory_data['vec_stats']
    
    log_output = (
        "## 📡 演化顯化軌跡：`" + target_file + "`\n"
        "* **物理時間戳**：`" + timestamp + "`\n\n"
        "### 1. 🧠 核心邏輯拓樸萃取 (Semantic Gravity Extraction)\n"
        "* **邏輯節點數**：從全文提煉出 `" + str(trajectory_data['fragment_count']) + "` 個具備相容性之核心論述碎片。\n"
        "* **原波包矩陣特徵** (經包立不相容原理過濾後之全域本體)：\n"
        "    * `均值 (Mean)`：" + f"{stats['mean']:.8f}" + "\n"
        "    * `標準差 (Std)`：" + f"{stats['std']:.8f}" + "\n"
        "    * `模長 (L2 Norm)`：" + f"{stats['norm']:.8f}" + "\n\n"
        "### 2. 🧬 尤拉相位顯化 (Euler Phase Manifestation)\n"
        "* **系統說明**：*系統已徹底廢除 LLM 生成層，改以具備全域多樣性之「原話波包」直接撞擊觀測矩陣中的尤拉相位(sin/cos)，實現 0 雜訊之實相顯化。*\n"
        "* **狀態張量**：`[" + hex_code + "]`\n"
        "* **物理相變**：**" + hex_info['name'] + "**\n"
        "* **學術指紋**：\n"
        "    > " + hex_info['desc'] + "\n\n"
        "---\n"
        "### 🔗 附錄：系統提煉之「原始本體顯化」\n"
        "*(本段落為圖論演算法從全文中直接萃取之最高引力論述，已排除同義塌陷，保持原創意志全貌)*\n\n"
        + trajectory_data['pure_fragments'] + "\n\n"
        "---\n"
    )
    return log_output

def export_wordpress_html(basename, content, hex_code, state_name):
    html_content = content.replace('\n', '<br>')
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    html_output = (
        "<div class=\"avh-hologram-article\">\n"
        "    <div class=\"avh-content\">\n"
        "        " + html_content + "\n"
        "    </div>\n"
        "    <hr>\n"
        "    <div class=\"avh-seal\" style=\"border: 2px solid #333; padding: 20px; background: #fafafa; margin-top: 30px;\">\n"
        "        <p><strong>📡 本理論已完成 學術價值全像儀 (AVH) 絕對邏輯顯化</strong></p>\n"
        "        <p>當下演化狀態：[ " + hex_code + " ] - <strong>" + state_name + "</strong></p>\n"
        "        <p>物理時間戳：" + timestamp_str + "</p>\n"
        "        <p><em>V7.1.0 絕對寂靜協議 | 本體論底層保護 | AJ Consulting</em></p>\n"
        "    </div>\n"
        "</div>\n"
    )
    with open("WP_Ready_" + basename + ".html", "w", encoding="utf-8") as f:
        f.write(html_output)

def export_latex(basename, content, hex_code, state_name):
    tex_content = content.replace("#", "\\section")
    tex_output = (
        "\\documentclass{article}\n"
        "\\usepackage[utf8]{inputenc}\n"
        "\\usepackage{xeCJK}\n"
        "\\title{" + basename + "}\n"
        "\\author{Alaric Kuo}\n"
        "\\date{\\today}\n"
        "\\begin{document}\n"
        "\\maketitle\n"
        "\\begin{abstract}\n"
        "本文章經由 AVH 學術價值全像儀觀測，當下演化狀態顯化為 [" + hex_code + "] " + state_name + "。\n"
        "\\end{abstract}\n\n"
        + tex_content + "\n\n"
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
        
    source_files = []
    for ext in ["*.md", "*.tex"]:
        source_files.extend([f for f in glob.glob(ext) if f.lower() not in ["avh_observation_log.md"]])
    
    if not source_files:
        print("系統休眠：未偵測到有效理論源碼波包。")
        sys.exit(0)
        
    print("\n🚀 啟動 AVH 造物引擎 (V7.1.0 絕對寂靜模式)，共偵測到 " + str(len(source_files)) + " 個波包等待顯化...")
    
    with open("AVH_OBSERVATION_LOG.md", "w", encoding="utf-8") as log_file:
        log_file.write("# 📡 AVH 學術價值全像儀：本體論顯化軌跡\n")
        log_file.write("*本文件詳實紀錄知識波包透過圖論萃取出的絕對核心邏輯碎片。系統已封印語言模型，並導入量子斥力（包立不相容原理）避免語意塌陷，確保作者的「全域原始波包」能無雜訊地撞擊觀測矩陣中的尤拉相位(sin/cos)。*\n\n---\n")
        
        last_hex_code = ""
        for target_source in source_files:
            trajectory_data = extract_pure_ontology(target_source)
            if not trajectory_data: continue
            
            ordered_dimensions = ["value_intent", "governance", "cognition", "architecture", "expansion", "application"]
            hex_bits = ""
            
            pure_will_psi = trajectory_data["psi_global"]
            
            for key in ordered_dimensions:
                dim = manifest["dimensions"][key]
                v_sin = embedding_model.encode([dim["sin_def"]])[0]
                v_cos = embedding_model.encode([dim["cos_def"]])[0]
                
                # 計算原始意志波包與尤拉相位的相似度
                sim_sin = np.dot(pure_will_psi, v_sin) / (np.linalg.norm(pure_will_psi) * np.linalg.norm(v_sin))
                sim_cos = np.dot(pure_will_psi, v_cos) / (np.linalg.norm(pure_will_psi) * np.linalg.norm(v_cos))
                
                hex_bits += "1" if sim_sin > sim_cos else "0"
            
            last_hex_code = hex_bits
            state_name = manifest["states"][hex_bits]["name"]
            
            report = generate_trajectory_log(target_source, trajectory_data, hex_bits, manifest)
            log_file.write(report)
            
            basename = os.path.splitext(target_source)[0]
            export_wordpress_html(basename, trajectory_data["full_text"], hex_bits, state_name)
            export_latex(basename, trajectory_data["full_text"], hex_bits, state_name)
            
            print("✅ " + target_source + " 原始意志顯化完成！ [" + hex_bits + "]")

    if last_hex_code:
        with open(os.environ.get("GITHUB_ENV", "env.tmp"), "a") as env_file:
            env_file.write("HEX_CODE=" + last_hex_code + "\n")
