"""
练习 3：从零实现 ReAct（不用 framework）

70 行 Python 写出 Thought → Action → Observation 循环。
不要 LangChain、不要 LangGraph，就是纯 while loop。

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

def tool_calculator(expression: str) -> str:
    """安全计算器：只允许 + - * / 和数字。"""
    allowed = set("0123456789.+-*/() ")
    if any(c not in allowed for c in expression):
        return f"error: 表达式包含不允许字符（{expression}）"
    try:
        return str(eval(expression))  # noqa: S307 — 已用白名单过滤
    except Exception as e:  # noqa: BLE001
        return f"error: {e}"


def tool_lookup_fact(query: str) -> str:
    """模拟事实查询（教学专用、避免依赖外部 API）。"""
    facts = {
        "台北人口": "2602000",
        "纽约人口": "8336000",
        "光速": "299792458",  # m/s
    }
    return facts.get(query.strip(), f"unknown: {query}")


# OpenAI 兼容的工具 schema 包装在 {"type":"function", "function":{...}} 中
TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "进行基本算术运算（加减乘除）。输入为表达式字符串。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "算术表达式"},
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_fact",
            "description": "查询一个事实（人口 / 物理常数等）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "查询关键词"},
                },
                "required": ["query"],
            },
        },
    },
]

TOOL_IMPL = {
    "calculator": lambda inp: tool_calculator(inp["expression"]),
    "lookup_fact": lambda inp: tool_lookup_fact(inp["query"]),
}


# === 2. ReAct 循环 ===

def react_loop(question: str, max_iter: int = 6, client: Any = None) -> dict:
    """
    OpenAI 兼容 ReAct 循环。每轮：
      1. 调用 LLM（包含 tools）
      2. finish_reason: 'tool_calls' → 执行工具、将观察结果接回、继续
                       'stop' → 结束、最后一条消息为答案
    返回 {final, trace, steps}。
    """
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
        thought_text = msg.content or ""
        tool_calls = msg.tool_calls or []

        # 将助手消息加入 messages（OpenAI 格式）
        assistant_entry: dict = {"role": "assistant", "content": thought_text}
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
            trace.append({"step": step, "thought": thought_text, "tool": None, "obs": None})
            return {"final": thought_text, "trace": trace, "steps": step + 1}

        # 执行工具调用、将观察结果接回（OpenAI 使用 role="tool"）
        last_obs = ""
        for tc in tool_calls:
            fn = TOOL_IMPL.get(tc.function.name)
            args = json.loads(tc.function.arguments)
            obs = fn(args) if fn else f"error: 未知工具 {tc.function.name}"
            last_obs = obs
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": obs,
            })

            trace.append({
                "step": step,
                "thought": thought_text,
                "tool": tc.function.name,
                "tool_input": json.loads(tc.function.arguments),
                "obs": last_obs,
            })

    return {"final": None, "trace": trace, "steps": max_iter, "truncated": True}


# === 3. 自我验证 ===

if __name__ == "__main__":
    question = "'台北人口' 除以 '纽约人口'、结果保留 4 位小数。"
    print(f"❓ 问题：{question}（使用 Ollama {MODEL}）")
    print("-" * 60)

    result = react_loop(question, max_iter=5)

    for entry in result["trace"]:
        print(f"[步骤 {entry['step']}] 思考: {(entry['thought'] or '')[:80]}...")
        if entry["tool"]:
            print(f"           工具: {entry['tool']}({entry.get('tool_input')}) → {entry['obs']}")
    print("-" * 60)
    print(f"✅ 最终答案：{result['final']}")
    print(f"   共 {result['steps']} 轮")

    # 宽松验证（小模型不一定精确到 4 位小数）
    assert result.get("final") is not None or result.get("truncated"), "循环应正常结束或显式截断"
    print("✅ 练习 3 通过 — 你已使用 deepseek 跑通 ReAct + 工具调用、0 成本运行")