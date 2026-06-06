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

# 中文情绪分类（正面 / 负面 / 中立）
TEST_SET = [
    ("这部电影超赞、看完想再看一次！", "正面"),
    ("剧情无聊、演员演技尴尬。", "负面"),
    ("这是一部 2019 年的电影。", "中立"),
    ("我不确定喜不喜欢、可能再想想。", "中立"),
    ("第一集很不错但第二集就崩了。", "负面"),
    ("看完心情很好、推荐！", "正面"),
]

FEW_SHOT_EXAMPLES = """范例：
input: 这家餐厅的牛排好吃到让我哭出来。
output: 正面

input: 服务生态度很差、我再也不会来了。
output: 负面

input: 这家店位于新北市三重区。
output: 中立
"""

def classify(text: str, *, use_few_shot: bool) -> str:
    prefix = FEW_SHOT_EXAMPLES + "\n" if use_few_shot else ""
    msg = client.chat.completions.create(
        model="deepseek-v4-flash",
        max_tokens=10,
        messages=[{"role": "user", "content": f"{prefix}input: {text}\noutput:"}],
        stream=False,
        extra_body={"thinking": {"type": "disabled"}}
    )
    return msg.choices[0].message.content.strip().splitlines()[0]


def evaluate(use_few_shot: bool) -> tuple[int, int]:
    correct = 0
    for text, label in TEST_SET:
        pred = classify(text, use_few_shot=use_few_shot)
        ok = label in pred
        print(f"  {'✓' if ok else '✗'} [{label}] {text[:30]}... → '{pred}'")
        if ok:
            correct += 1
    return correct, len(TEST_SET)


print("=== 0-shot ===")
c0, n = evaluate(use_few_shot=False)
print(f"正确 {c0}/{n} = {c0/n:.0%}")

print("\n=== 3-shot ===")
c3, _ = evaluate(use_few_shot=True)
print(f"正确 {c3}/{n} = {c3/n:.0%}")

# === 自我验证 ===
assert c3 >= c0, f"预期 3-shot 不比 0-shot 差、实际 {c3} < {c0}（小 model 样本小、跑几次比较）"
print(f"\n✅ 练习 2 通过 — 0-shot {c0}/{n}、3-shot {c3}/{n}（本机 $0）")
print("💡 观察：'中立' 在 0-shot 容易被误判成正面或负面、3-shot 后改善明显")
print("💡 小 model（gemma4:e4b）通常 0-shot 表现比 Claude 差更多、所以 few-shot 改善幅度更大")