# 需要：pip install anthropic
# 环境变量：export ANTHROPIC_API_KEY=sk-ant-...
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.environ.get('DEEPSEEK_API_KEY'),
                base_url="https://api.deepseek.com")
                
# msg = client.messages.create(
#     model="claude-haiku-4-5",  # haiku 最便宜；换 sonnet 改这行
#     max_tokens=100,
#     messages=[{"role": "user", "content": "用一句话自我介绍。"}],
# )
msg = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[
        {"role": "user", "content": "用一句话自我介绍"},
    ],
    stream=False,
    extra_body={"thinking": {"type": "disabled"}}
)

# === 自我验证 ===
text = msg.choices[0].message.content
print("回应：", text)
print("usage:", msg.usage)

assert msg.choices[0].finish_reason in ("stop", "length"), f"非预期 finish_reason: {msg.choices[0].finish_reason}"
assert len(text) > 0, "回应不应为空"
assert msg.usage.prompt_tokens > 0 and msg.usage.completion_tokens > 0, "token 数应 > 0"
print("✅ 练习 1 通过 — 你已成功打通 Deepseek API")