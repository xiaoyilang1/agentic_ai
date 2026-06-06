"""
练习 5：工具错误处理 

故意让 fetch_weather 第一次返回结构化 error、第二次才成功。观察 ReAct 循环如何
把 error 观测结果交回 LLM，让模型自己决定重试 / 修改查询 / 放弃。
重点：error 是结构化 dict（`{"error", "retry_hint"}`），不是 Python 异常。

⚠️ 注意：小模型对 retry_hint 的后续处理能力较弱，可能直接放弃或忽略提示。
这正是教学点——同样结构化错误模式，不同模型的“理解能力”差异明显。
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

_failure_plan: list[bool] = [True, False]


def set_weather_failures(plan: list[bool]) -> None:
    global _failure_plan
    _failure_plan = list(plan)


def fetch_weather(city: str) -> dict:
    """模拟天气 API，用 _failure_plan 决定本次失败还是成功。"""
    should_fail = _failure_plan.pop(0) if _failure_plan else False
    if should_fail:
        return {"error": "network timeout", "retry_hint": "try again in 1s"}#错误处理
    return {"city": city, "forecast": "rain", "temperature_c": 24}


TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "fetch_weather",
            "description": "Fetch current weather. If an error is returned, inspect retry_hint before retrying.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name"}},
                "required": ["city"],
            },
        },
    }
]


def react_loop(question: str, max_iter: int = 5, client: Any = None) -> dict:
    """OpenAI 兼容 ReAct 循环。工具错误是结构化 JSON，放入 tool_result 交给 LLM。"""
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
            args = json.loads(tc.function.arguments)
            obs = fetch_weather(args["city"]) if tc.function.name == "fetch_weather" else {"error": "unknown tool"}
            # OpenAI 兼容的 tool message content 支持字符串，将 dict 序列化
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(obs, ensure_ascii=False),
            })
            trace.append({
                "step": step,
                "thought": text,
                "tool": tc.function.name,
                "tool_input": args,
                "obs": obs
            })

    return {"final": None, "trace": trace, "steps": max_iter, "truncated": True}


if __name__ == "__main__":
    set_weather_failures([True, False])  # 第一次失败、第二次成功
    print(f"❓ 问题：Will it rain in Taipei today?（使用 {MODEL}）")
    print("-" * 60)

    result = react_loop("Will it rain in Taipei today?")
    for entry in result["trace"]:
        if entry["tool"]:
            print(f"[步骤 {entry['step']}] 工具: {entry['tool']}({entry.get('tool_input')}) → {entry['obs']}")
    print("-" * 60)
    print(f"✅ 最终答案：{result['final']}")

    # 宽松验证：循环中至少出现一次结构化错误（小模型不一定会重试）
    saw_error = any(isinstance(e["obs"], dict) and "error" in e["obs"] for e in result["trace"])
    assert saw_error, "预期至少一轮工具返回结构化错误"
    print("✅ 练习 5 通过 — 你已使用 deepseek 观察到工具错误在 ReAct 循环中是数据而非异常、0 成本运行")