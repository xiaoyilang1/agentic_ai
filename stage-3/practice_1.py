import sys, json,os
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(base_url="https://api.deepseek.com", 
                api_key=os.environ.get("DEEPSEEK_API_KEY"))


# Step 1: 定义 tool schema — OpenAI-compatible 格式包一层 {"type":"function", "function":{...}}
weather_tool = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "查询城市目前天气（晴/雨/阴），回传一个短字符串。",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名称（如「台北」）"},
            },
            "required": ["city"],
        },
    },
}

# Step 2: 问问题、让 LLM 自己决定要不要调用 tool
resp = client.chat.completions.create(
    model="deepseek-v4-flash",
    max_tokens=512,
    tools=[weather_tool],
    messages=[{"role": "user", "content": "台北现在有下雨吗？"}],
    stream=False,
    extra_body={"thinking":{"type": "disabled"}}
)

# === 自我验证 ===
msg = resp.choices[0].message
print("finish_reason:", resp.choices[0].finish_reason)
print("tool_calls:", msg.tool_calls)
print("content:", msg.content)

assert msg.tool_calls, "预期 LLM 会选择调用 tool（而非直接回答）"
tc = msg.tool_calls[0]
assert tc.function.name == "get_weather", f"预期调用 get_weather、实际 {tc.function.name}"
args = json.loads(tc.function.arguments)
assert args.get("city"), "预期 city 参数有值"
print(f"✅ 练习 1 通过 — deepseek-v4-flash 正确选了 get_weather、带 city='{args['city']}' 参数")