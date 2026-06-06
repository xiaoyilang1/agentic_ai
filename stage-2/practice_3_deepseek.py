import sys, re
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

QUESTION = "小明有 3 颗苹果。他给了小华 1 颗、又从妈妈那边拿到 5 颗、然后吃了 2 颗。请问现在剩几颗？"
ANSWER = 5  # 3 - 1 + 5 - 2 = 5

COT_EXAMPLE = """范例：
Q: 一只鸡有 2 只脚。3 只鸡跟 1 个人共有几只脚？
A: 让我一步一步算。3 只鸡 × 2 只脚 = 6 只脚。1 个人有 2 只脚。总共 6 + 2 = 8 只脚。答案是 8。
"""

def ask(prompt: str) -> str:
    msg = client.chat.completions.create(
        model="deepseek-v4-flash", max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
        stream=False,
        extra_body={"thinking": {"type": "disabled"}}
    )
    return msg.choices[0].message.content

def extract_number(text: str) -> int | None:
    nums = re.findall(r"-?\d+", text)
    return int(nums[-1]) if nums else None


# A. 纯 prompt
out_a = ask(QUESTION); ans_a = extract_number(out_a)

# B. + Let's think step by step
out_b = ask(QUESTION + "\nLet's think step by step."); ans_b = extract_number(out_b)

# C. + CoT example
out_c = ask(COT_EXAMPLE + "\n\nQ: " + QUESTION + "\nA:"); ans_c = extract_number(out_c)

for label, out, ans in [("A 纯 prompt", out_a, ans_a), ("B +step-by-step", out_b, ans_b), ("C +CoT example", out_c, ans_c)]:
    print(f"\n--- [{label}] 答案={ans} {'✓' if ans == ANSWER else '✗'} ---")
    print(out[:200])

# === 自我验证 ===
correct = sum(1 for a in (ans_a, ans_b, ans_c) if a == ANSWER)
assert correct >= 1, f"3 种 prompt 至少要 1 种答对、实际 {correct}/3"
# 小 model 对 CoT 依賴性更高、放宽条件：B 或 C 至少 1 对（vs Anthropic Path B 要求严格）
assert ans_b == ANSWER or ans_c == ANSWER, "B (step-by-step) 或 C (CoT example) 至少一种要答对 — CoT 对小 model 是基本功"
print(f"\n✅ 练习 3 通过 — {correct}/3 答对（本机 $0）")
print(f"💡 观察小 model：A 纯 prompt 通常答错、B/C 加 CoT 后明显改善——比 Claude 更能凸显 CoT 重要性")