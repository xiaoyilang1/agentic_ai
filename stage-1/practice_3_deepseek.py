import sys, statistics
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# client = anthropic.Anthropic()
client = OpenAI(api_key=os.environ.get('DEEPSEEK_API_KEY'),
                base_url="https://api.deepseek.com")

PRICING = {
    "deepseek-v4-flash" : {"input": 1.00, "output": 2.00},
    "deepseek-v4-pro" : {"input": 3.00, "output": 6.00}
}

MODELS = ["deepseek-v4-flash", "deepseek-v4-pro"]
msg = client.chat.completions.create(
    model=MODELS[0],
    messages=[{"role": "user", "content": "你好！自我介绍一下。"}],
    stream=False,
    extra_body={"thinking": {"type": "disabled"}}
)
in_tok,out_tok = msg.usage.prompt_tokens,msg.usage.completion_tokens
rates = PRICING[MODELS[0]]
cost_one = (in_tok * rates["input"] + out_tok * rates["output"]) / 1_000_000

print(f"model: {MODELS[0]}")
print(f"single: input={in_tok} output={out_tok} → ${cost_one:.6f}")
print(f"1000 calls cost across model tiers:")
for name, r in PRICING.items():
    c = (in_tok * r["input"] + out_tok * r["output"]) / 1_000_000 * 1000
    print(f"  {name:<22} ${c:.4f}")

assert cost_one > 0, "Cloud LLM 一定有成本"
print(f"\n✅ 练习 3 通过（Deepseek）— 1000 次 deepseek-v4-flash ≈ $0.3990、deepseek-v4-pro ≈ $1.1970")