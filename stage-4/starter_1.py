"""Stage 4 练习 1：同一个 agent、两个框架 — LangGraph + DeepSeek（路径 A）。

任务：搜索 + 摘要的小 agent。给一个 query、agent 用 search tool 拿数据、回一段摘要。
这份用 LangGraph。对照的 CrewAI 版本见 starter_crewai_deepseek.py。

运行方式：
    pip install -r requirements.txt   # 需要 langgraph, langchain-openai
    export DEEPSEEK_API_KEY=sk-...
    python starter_deepseek.py

预算：约 ¥0.002/run（deepseek-chat）。
"""

from __future__ import annotations

import os
import sys
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
from dotenv import load_dotenv
load_dotenv()
# DeepSeek 配置
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise ValueError("请设置环境变量 DEEPSEEK_API_KEY")

MODEL = os.environ.get("MODEL", "deepseek-v4-flash")
BASE_URL = "https://api.deepseek.com"


# === Tool ===

@tool
def search(query: str) -> str:
    """Search a (fake, offline) knowledge base for a topic."""
    db = {
        "taipei": "Taipei is the capital of Taiwan, population ~2.6M, known for night markets.",
        "react agent": "ReAct (Reasoning + Acting) is an agent pattern: think → act → observe loop.",
    }
    return db.get(query.strip().lower(), f"no entry for {query}")


# === LangGraph 状态定义 ===

class State(TypedDict):
    messages: Annotated[list, add_messages]


def build_graph(llm: Any) -> Any:
    """组装一个 search → summarize 的图：LLM 看 query 决定要不要调用 search、看到 tool result 后给摘要。"""
    llm_with_tools = llm.bind_tools([search])

    def agent_node(state: State):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    def tool_node(state: State):
        msg = state["messages"][-1]
        results = []
        for call in msg.tool_calls:
            obs = search.invoke(call["args"])
            results.append(ToolMessage(content=obs, tool_call_id=call["id"]))
        return {"messages": results}

    def should_continue(state: State) -> str:
        return "tools" if state["messages"][-1].tool_calls else END

    g = StateGraph(State)
    g.add_node("agent", agent_node)
    g.add_node("tools", tool_node)
    g.set_entry_point("agent")
    g.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile()


def run(query: str, llm: Any = None) -> dict:
    """公共入口：运行一轮 agent、返回 final message + steps。"""
    if llm is None:
        llm = ChatOpenAI(
            base_url=BASE_URL,
            api_key=DEEPSEEK_API_KEY,
            model=MODEL,
            temperature=0,
        )
    graph = build_graph(llm)
    final_state = graph.invoke({"messages": [HumanMessage(content=query)]})
    return {
        "final": final_state["messages"][-1].content,
        "steps": len(final_state["messages"]),
    }


if __name__ == "__main__":
    query = "summarize what you know about Taipei"
    print(f"❓ Query: {query}（using DeepSeek {MODEL}）")
    print("-" * 60)
    result = run(query)
    print(f"✅ Final: {result['final']}")
    print(f"   Steps: {result['steps']}")
    assert result["final"], "expected non-empty summary"
    print("✅ 练习 1 通过 — LangGraph + DeepSeek，约 ¥0.002/run")