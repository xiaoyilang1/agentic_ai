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

PROMPTS = {"中文": "用一句话描述一只猫在做什么。", "English": "Describe in one sentence what a cat is doing."}

for label, prompt in PROMPTS.items():
    output_tokens = []
    for _ in range(20):
        msg = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {"role": "user", "content": prompt},
        ],
        stream=False,
        extra_body={"thinking": {"type": "disabled"}}
    )
        # msg = client.messages.create(model="claude-haiku-4-5", max_tokens=80, temperature=1.0,
        #                              messages=[{"role": "user", "content": prompt}])
        output_tokens.append(msg.usage.completion_tokens)
    print(f"[{label}] input={msg.usage.prompt_tokens} output min/max/mean={min(output_tokens)}/{max(output_tokens)}/{sum(output_tokens)/len(output_tokens):.1f}")