"""
练习 6：Schema 设计 — 错误 Schema

故意保留反面模式：描述太模糊、参数都用 string、没有必填项、没有枚举。
小模型（qwen2.5:3b）对 Schema 质量比大模型更敏感——坏的 Schema 在 Claude haiku
上可能还能猜对，在 qwen2.5:3b 上几乎必错。对照 `starter_good.py`。
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

MODEL = os.environ.get("MODEL", "deepseek-v4-flash")  # tool-use 稳定的 model 


def process_data(data: str = "") -> str:
    return f"processed generic data: {data}"


def convert_temperature(value: str = "") -> str:
    return f"converted something from: {value}"


# 反面模式：描述太短、参数全是 string、无必填、无枚举
TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "process_data",
            "description": "Process data.",
            "parameters": {"type": "object", "properties": {"data": {"type": "string"}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "convert_temperature",
            "description": "Convert a value.",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "string"}, "unit": {"type": "string"}},
            },
        },
    },
]

TOOL_IMPL = {
    "process_data": lambda i: process_data(i.get("data", "")),
    "convert_temperature": lambda i: convert_temperature(i.get("value", "")),
}


def select_and_run(question: str, client: Any = None) -> dict:
    client = OpenAI(base_url="https://api.deepseek.com", 
                    api_key=os.environ.get("DEEPSEEK_API_KEY"))
    resp = client.chat.completions.create(
        model=MODEL,
        tools=TOOLS_SPEC,
        messages=[{"role": "user", "content": question}],
    )
    msg = resp.choices[0].message
    tool_calls = msg.tool_calls or []
    if not tool_calls:
        return {"tool": None, "tool_input": {}, "observation": None}
    call = tool_calls[0]
    args = json.loads(call.function.arguments)
    return {"tool": call.function.name, "tool_input": args, "observation": TOOL_IMPL[call.function.name](args)}


if __name__ == "__main__":
    question = "Convert 32 Celsius to Fahrenheit."
    print(f"❓ 问题：{question}（使用 {MODEL}、错误 Schema）")
    result = select_and_run(question)
    print(f"   工具: {result['tool']}")
    print(f"   结果: {result['observation']}")

    # 宽松验证：坏 Schema 不保证选对工具，但至少要产生一个工具调用
    assert result["tool"] is not None, "even bad schema should produce an observable selection"
    if result["tool"] != "convert_temperature":
        print("⚠ 小模型 + 坏 Schema → 预期会选错。对照 starter_good.py 看修正后的差异")
    print("✅ 错误 Schema 示例运行完成 — 对照 starter_good.py")