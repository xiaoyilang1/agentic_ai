import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)


# 5 个 iteration、每一轮 prompt 都比前一轮更具体
PROMPTS = {
    "v1 模糊": "写一段介绍 ReAct 的文字。",
    "v2 加目标读者": "写一段介绍 ReAct 的文字、给写过 Python 的软件工程师看。",
    "v3 加格式": "写一段介绍 ReAct 的文字、给写过 Python 的软件工程师看。100 字以内、用一个段落。",
    "v4 加 example 要求": "写一段介绍 ReAct 的文字、给写过 Python 的软件工程师看。100 字以内、用一个段落、结尾举一个具体例子（譬如查天气）。",
    "v5 加禁忌": "写一段介绍 ReAct 的文字、给写过 Python 的软件工程师看。100 字以内、用一个段落、结尾举一个具体例子（譬如查天气）。不要用「赋能」「驱动」「智能」这类空泛词汇。",
    }

outputs = {}
for label, prompt in PROMPTS.items():
    r = client.chat.completions.create(
        model="deepseek-v4-flash",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
        stream=False,
        extra_body={"thinking":{"type": "disabled"}}
    )
    text = r.choices[0].message.content
    outputs[label] = text
    print(f"\n--- [{label}] ({len(text)} chars) ---")
    print(text)

# === 自我验证 ===
v1_len, v5_len = len(outputs["v1 模糊"]), len(outputs["v5 加禁忌"])
banned_words = ("赋能", "驱动", "智能")
v5_has_banned = any(w in outputs["v5 加禁忌"] for w in banned_words)
assert v5_len > 0, "v5 必须有输出"
assert not v5_has_banned, f"v5 应该避免禁忌词、实际含: {[w for w in banned_words if w in outputs['v5 加禁忌']]}"
print(f"\n✅ 练习 4 通过 — v5 长度 {v5_len}、无禁忌词（本机 $0）")
print(f"💡 观察：v1 ({v1_len} chars) 通常比 v5 ({v5_len} chars) 「松」、加约束会逼 prompt 收敛")
print("💡 用 gemma4:e4b 跑这题特别有感——小 model 对 prompt 质量极敏感、5 轮 refine 的差距会比 Claude 更明显")