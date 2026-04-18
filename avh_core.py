import os
import sys
import json
import glob
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
import zhconv

# ==============================================================================
# AVH Genesis Engine (V14.0 絕對自鎖：AI 識讀與華麗顯化版)
# ==============================================================================

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

def process_auto_locked_synthesis(source_path, manifest):
    print(f"\n🌊 [大腦啟動] 正在讀取源碼：{source_path}")
    try:
        with open(source_path, 'r', encoding='utf-8') as file:
            raw_text = file.read()
            
        if len(raw_text) < 100:
            print(f"⚠️ {source_path} 文本過短，無法進行脈絡識讀。")
            return None
            
        dimension_prompts = ""
        ordered_keys = ["value_intent", "governance", "cognition", "architecture", "expansion", "application"]
        for idx, key in enumerate(ordered_keys):
            dim = manifest["dimensions"][key]
            dimension_prompts += f"維度 {idx+1} ({dim['layer']}):\n"
            dimension_prompts += f" - [1] 離群突破: {dim['sin_def']}\n"
            dimension_prompts += f" - [0] 守成合群: {dim['cos_def']}\n\n"
            
        # ---------------------------------------------------------
        # 步驟 1：AI 全文識讀 (AI Determines the State)
        # ---------------------------------------------------------
        print("👁️ [脈絡識讀] 由 AI 閱讀全文並判定高維價值指紋...")
        
        eval_sys_prompt = (
            "你是一個高維度的學術價值觀測儀。請閱讀使用者的全文，並根據以下六個維度進行嚴格判定。\n\n"
            f"{dimension_prompts}"
            "【絕對指令】：只輸出一串 6 個數字的代碼（只包含 0 或 1）。嚴禁輸出任何廢話。"
        )
        eval_user_prompt = f"請判定以下文本的 6 位元指紋：\n\n{raw_text[:3000]}"
        
        raw_source_hex = ask_llm(eval_sys_prompt, eval_user_prompt, max_tokens=10, temp=0.1)
        source_hex = ''.join(filter(lambda x: x in ['0', '1'], raw_source_hex))
        
        if len(source_hex) != 6:
            print(f"⚠️ AI 判定指紋格式錯誤 ({raw_source_hex})，強制設為 000000。")
            source_hex = "000000"
            
        state_info = manifest["states"].get(source_hex, {"name": "未知狀態", "desc": "缺乏觀測紀錄"})
        
        print(f"✅ 測得學術指紋: [{source_hex}] - {state_info['name']}")

        # ---------------------------------------------------------
        # 步驟 2：順向自鎖顯化 (Auto-Locked Synthesis)
        # ---------------------------------------------------------
        print(f"🛡️ [自鎖顯化] 正在將 [{source_hex}] 的天命評語化為鋼鐵模具，強制 AI 進行論述...")
        
        summary_sys_prompt = f"""
你是一個精準的學術本體論大腦。
本系統已經判定，這篇理論的「學術演化狀態」為：[{source_hex}] - {state_info['name']}。
核心精神評語為：「{state_info['desc']}」

【絕對任務】：
請閱讀使用者的全文，並以繁體中文寫出一段氣勢磅礴、邏輯連貫的系統摘要。
你必須在摘要中，精準地展現出上述「核心精神評語」所描述的價值。
【格式警告】：直接輸出段落文字。嚴禁使用 Markdown 分隔線、嚴禁產生條列式清單。
論述完畢請強制輸出『[顯化完畢]』。
"""
        summary_user_prompt = f"請消化以下全文，並在狀態鎖定下進行純粹的思想顯化：\n\n{raw_text[:3000]}"
        
        generated_summary = ask_llm(summary_sys_prompt, summary_user_prompt, max_tokens=900, temp=0.35)
        
        if "[顯化完畢]" in generated_summary:
            generated_summary = generated_summary.split("[顯化完畢]")[0].strip()
        elif "顯化完畢" in generated_summary:
            generated_summary = generated_summary.split("顯化完畢")[0].strip()
            
        # 清理可能殘留的 markdown 雜訊
        generated_summary = generated_summary.replace("**---", "").replace("###", "").strip()
        
        print("✅ [顯化完成] AI 已成功在自鎖狀態下完成全脈絡融合！")

        return {
            "hex_code": source_hex,
            "state_name": state_info['name'],
            "state_desc": state_info['desc'],
            "summary": generated_summary,
            "full_text": raw_text
        }
    except Exception as e:
        print(f"工具調用失敗，原因為 邏輯顯化過程錯誤 ({e})")
        sys.exit(1)

def generate_trajectory_log(target_file, data):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")
    
    log_output = (
        f"## 📡 演化顯化軌跡：`{target_file}`\n"
        f"* **物理時間戳**：`{timestamp}`\n\n"
        f"### 1. 👁️ AI 全脈絡識讀 (Full-Context Neural Measurement)\n"
        f"*系統由 AI 大腦直接閱讀全文，不依賴數學平均值降維，精準捕捉文本的高維度突破意圖。*\n"
        f"* **觀測指紋**：`[{data['hex_code']}]` - **{data['state_name']}**\n"
        f"* **學術指紋評語**：\n"
        f"    > {data['state_desc']}\n\n"
        f"---\n"
        f"### 2. 🧠 狀態自鎖顯化摘要 (Auto-Locked Synthesis)\n"
        f"*(系統將上述測得之指紋與評語化為「絕對物理枷鎖」，強制約束 AI 進行全脈絡的摘要生成，確保論述與價值完美接地。)*\n\n"
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
        "        <p><em>V14.0 順向自鎖協議 | 本體論底層保護 | AJ Consulting</em></p>\n"
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
        
    print(f"\n🚀 啟動 AVH 造物引擎 (V14.0 絕對自鎖模式)，共偵測到 {len(source_files)} 個波包等待觀測...")
    
    with open("AVH_OBSERVATION_LOG.md", "w", encoding="utf-8") as log_file:
        log_file.write("# 📡 AVH 學術價值全像儀：狀態自鎖顯化軌跡\n")
        log_file.write("*本文件詳實紀錄系統如何利用 AI 進行高維度脈絡識讀。當系統取得原文的「學術指紋」後，會立即將該指紋與天命評語轉化為「絕對約束枷鎖」，強制 AI 進行精準、華麗且邏輯接地的論述顯化。*\n\n---\n")
        
        last_hex_code = ""
        for target_source in source_files:
            result_data = process_auto_locked_synthesis(target_source, manifest)
            if result_data:
                last_hex_code = result_data['hex_code']
                report = generate_trajectory_log(target_source, result_data)
                log_file.write(report)
                
                # 恢復華麗的生態系產出
                basename = os.path.splitext(target_source)[0]
                export_wordpress_html(basename, result_data)
                export_latex(basename, result_data)

    if last_hex_code:
        with open(os.environ.get("GITHUB_ENV", "env.tmp"), "a") as env_file:
            env_file.write(f"HEX_CODE={last_hex_code}\n")
