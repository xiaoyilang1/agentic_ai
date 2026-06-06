"""
练习 2：多工具选择 

让本机  在 3 个 tool（web_search / calculator / calendar_lookup）里选一个。
重点不是工具强不强，是观察 schema 的 description / 参数 / required 如何引导模型选对。
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

MODEL = os.environ.get("MODEL", "deepseek-v4-flash")  # tool-use 稳定的 Ollama model 

# === 1. Tools 定义（含实现）===

def web_search(query: str) -> str:
    return f"search result: {query} -> Anthropic tool use docs and examples"


def calculator(expression: str) -> str:
    allowed = set("0123456789.+-*/() ")
    if any(ch not in allowed for ch in expression):
        return "error: calculator only accepts basic arithmetic"
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))  # noqa: S307
    except Exception as exc:  # noqa: BLE001
        return f"error: {exc}"


def calendar_lookup(date: str) -> str:
    events = {
        "2026-05-13": "10:00 Stage 3 review, 15:00 agent study group",
        "tomorrow": "10:00 Stage 3 review, 15:00 agent study group",
    }
    return events.get(date.strip(), f"no events found for {date}")


# OpenAI-compat 的 tools schema 要包一层 {"type": "function", "function": {...}}
def _wrap(name: str, description: str, field: str, field_description: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {field: {"type": "string", "description": field_description}},
                "required": [field],
            },
        },
    }


TOOLS_SPEC = [
    _wrap("web_search", "Search current or external information not in the prompt.", "query", "Search query"),
    _wrap("calculator", "Evaluate basic arithmetic with +, -, *, /, and parentheses.", "expression", "Math expression"),
    _wrap("calendar_lookup", "Look up events for a specific date or relative day.", "date", "Date to inspect"),
]

TOOL_IMPL = {
    "web_search": lambda args: web_search(args["query"]),
    "calculator": lambda args: calculator(args["expression"]),
    "calendar_lookup": lambda args: calendar_lookup(args["date"]),
}


# === 2. 单轮 tool selection ===

def run_tool_selection(question: str, client: Any = None) -> dict:
    """单轮 call：LLM 看完 question + tools 后选一个 tool 呼叫，本地执行 observation 接回去。"""
    client = OpenAI(base_url="https://api.deepseek.com", 
                    api_key=os.environ.get("DEEPSEEK_API_KEY"))
    resp = client.chat.completions.create(
        model=MODEL,
        tools=TOOLS_SPEC,
        messages=[{"role": "user", "content": question}],
    )
    msg = resp.choices[0].message
    text = msg.content or ""
    tool_calls = msg.tool_calls or []
    if not tool_calls:
        return {"tool": None, "thought": text, "observation": None}
    call = tool_calls[0]
    args = json.loads(call.function.arguments)
    fn = TOOL_IMPL.get(call.function.name, lambda _: f"error: unknown tool {call.function.name}")
    return {"tool": call.function.name, "tool_input": args, "thought": text, "observation": fn(args)}


# === 3. 自我验证 ===

if __name__ == "__main__":
    question = "What is (19 * 42) - 8? Use the best available tool."
    print(f"❓ 问题：{question}（using {MODEL}）")
    result = run_tool_selection(question)
    print(f"   tool: {result['tool']}")
    print(f"   tool_input: {result.get('tool_input')}")
    print(f"   observation: {result['observation']}")

    assert result["tool"] == "calculator", f"预期 calculator、得到 {result['tool']}"
    assert result["observation"] and not result["observation"].startswith("error:")
    print("✅ 练习 2 通过 — 你已用{MODEL}跑通 multi-tool selection")
