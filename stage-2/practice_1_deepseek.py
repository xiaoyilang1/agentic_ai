import sys, json
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

SYSTEM_PROMPTS = {
    "严肃律师": "你是严谨的合约律师。回答要精准、引用法条编号、避免任何主观形容词。",
    "幼儿园老师": "你是温柔的幼儿园老师、要对 5 岁小孩说话。用比喻、口语、少于 80 字。",
    "JSON 机器": "你只回 JSON。schema: {\"answer\": string, \"confidence\": float}",
}
USER_MSG = "请帮我解释什么是租赁合约。"

outputs = {}
for label, system in SYSTEM_PROMPTS.items():
    # Anthropic 用 system= 参数（不放 messages 內）
    msg = client.chat.completions.create(
        model="deepseek-v4-flash", 
        max_tokens=200,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": USER_MSG}
        ],
        stream=False,
        extra_body={"thinking": {"type": "disabled"}}
    )
    outputs[label] = msg.choices[0].message.content
    print(f"\n--- [{label}] ---")
    print(outputs[label])

# 同样的 JSON assert（schema 跨 backend 通用）
json_output = outputs["JSON 机器"]
assert "{" in json_output and "}" in json_output
print(f"\n✅ 练习 1 通过（Anthropic）")