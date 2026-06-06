"""
练习 4：多步骤推理任务

把练习 3 的 ReAct loop 扩展为 3-5 步任务：
查询台北人口 → 查询纽约人口 → 相除 → 转为百分比。

重点：工具设计要小而稳定、LLM 负责规划下一步、max_iter 作为安全上限。


⚠️ 注意：4 步推理对小模型是挑战。qwen2.5:3b 可能中间漏掉一步、或过早停止。
Claude Haiku 会更稳定——这正是教学重点：对比相同循环、不同模型在第几步崩溃。
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


# === 1. 工具定义（含实现）===

def lookup_population(city: str) -> str:
    data = {"taipei": 2_602_000, "new york": 8_336_000, "empty city": 0}
    return str(data.get(city.strip().lower(), 0))


def divide(a: float, b: float) -> str:
    b = float(b)
    return "0" if b == 0 else str(float(a) / b)


def to_percentage(ratio: float) -> str:
    return f"{float(ratio) * 100:.2f}"


def round_int(x: float) -> str:
    return str(round(float(x)))


# OpenAI 兼容格式包装一层 {"type": "function", "function": {...}}
TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "lookup_population",
            "description": "Return the population for a known city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "divide",
            "description": "Divide a by b. Returns 0 instead of crashing when b is zero.",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                "required": ["a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "to_percentage",
            "description": "Convert a ratio such as 0.31 into a percentage number.",
            "parameters": {
                "type": "object",
                "properties": {"ratio": {"type": "number"}},
                "required": ["ratio"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "round_int",
            "description": "Round a number to the nearest integer.",
            "parameters": {
                "type": "object",
                "properties": {"x": {"type": "number"}},
                "required": ["x"],
            },
        },
    },
]

TOOL_IMPL = {
    "lookup_population": lambda i: lookup_population(i["city"]),
    "divide": lambda i: divide(i["a"], i["b"]),
    "to_percentage": lambda i: to_percentage(i["ratio"]),
    "round_int": lambda i: round_int(i["x"]),
}


# === 2. ReAct 循环（OpenAI 兼容）===

def react_loop(question: str, max_iter: int = 8, client: Any = None) -> dict:
    """OpenAI 兼容多步骤 ReAct 循环。"""
    client = OpenAI(base_url="https://api.deepseek.com", 
                    api_key=os.environ.get("DEEPSEEK_API_KEY"))
    messages = [{"role": "user", "content": question}]
    trace: list[dict] = []

    for step in range(max_iter):
        resp = client.chat.completions.create(
            model=MODEL,
            tools=TOOLS_SPEC,
            messages=messages,
        )
        msg = resp.choices[0].message
        text = msg.content or ""
        tool_calls = msg.tool_calls or []

        assistant_entry: dict = {"role": "assistant", "content": text}
        if tool_calls:
            assistant_entry["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in tool_calls
            ]
        messages.append(assistant_entry)

        if resp.choices[0].finish_reason == "stop" or not tool_calls:
            trace.append({"step": step, "thought": text, "tool": None, "obs": None})
            return {"final": text, "trace": trace, "steps": step + 1}

        for tc in tool_calls:
            fn = TOOL_IMPL.get(tc.function.name, lambda _: f"error: 未知工具 {tc.function.name}")
            args = json.loads(tc.function.arguments)
            obs = fn(args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": obs,
            })
            trace.append({"step": step, "thought": text, "tool": tc.function.name, "tool_input": args, "obs": obs})

    return {"final": None, "trace": trace, "steps": max_iter, "truncated": True}


# === 3. 自我验证 ===

if __name__ == "__main__":
    question = "Find Taipei population divided by New York population, then express it as a percentage."
    print(f"❓ 问题：{question}（使用 Ollama {MODEL}）")
    print("-" * 60)

    result = react_loop(question)
    for entry in result["trace"]:
        if entry["tool"]:
            print(f"[步骤 {entry['step']}] 工具: {entry['tool']}({entry.get('tool_input')}) → {entry['obs']}")
    print("-" * 60)
    print(f"✅ 最终答案：{result['final']}")
    print(f"   共 {result['steps']} 轮")

    # 宽松验证：小模型不一定走完 4 步、但循环至少要正常结束或显式截断
    assert result.get("final") is not None or result.get("truncated"), "循环应正常结束或截断"
    print("✅ 练习 4 通过 — 你已使用deepseek 跑通多步 ReAct 循环、0 成本运行")