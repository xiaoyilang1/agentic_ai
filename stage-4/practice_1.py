"""Stage 4 练习 1：同一个 agent、CrewAI 版本 — 使用 DeepSeek API。

跟 starter.py (LangGraph) 同一个任务：search + summarize。
看代码风格差异——CrewAI 用 Agent + Task 抽象、LangGraph 用 StateGraph + node。

运行方式：
    pip install -r requirements.txt   # 包含 crewai, langchain-openai
    export DEEPSEEK_API_KEY=sk-...
    python starter_crewai_deepseek.py

预算：约 ¥0.002/run（deepseek-chat 百万 tokens ¥1 输入，¥2 输出）。
"""

from __future__ import annotations

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from crewai import Agent, Crew, Task
from crewai.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

# DeepSeek 配置
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise ValueError("请设置环境变量 DEEPSEEK_API_KEY")

MODEL = os.environ.get("MODEL", "deepseek-v4-flash")
BASE_URL = "https://api.deepseek.com"


@tool("search")
def search(query: str) -> str:
    """搜索（模拟的离线）知识库中的主题。"""
    db = {
        "taipei": "台北是台湾的首都，人口约260万，以夜市闻名。",
        "react agent": "ReAct（推理+行动）是一种智能体模式：思考 → 行动 → 观察循环。",
    }
    return db.get(query.strip().lower(), f"没有关于 {query} 的条目")


def build_crew(query: str) -> Crew:
    """CrewAI 风格：一个 agent + 一个 task，使用 DeepSeek LLM。"""
    llm = ChatOpenAI(
        model=MODEL,
        openai_api_key=DEEPSEEK_API_KEY,
        base_url=BASE_URL,
        temperature=0,
    )

    researcher = Agent(
        role="研究员",
        goal="查找并总结所请求的主题。",
        backstory="你搜索知识库并给出简洁的总结。",
        tools=[search],
        llm=llm,
        verbose=False,
    )
    task = Task(
        description=query,
        expected_output="基于搜索结果的1-2句总结。",
        agent=researcher,
    )
    return Crew(agents=[researcher], tasks=[task], verbose=False)


def run(query: str) -> dict:
    crew = build_crew(query)
    result = crew.kickoff()
    return {"final": str(result), "steps": None}


if __name__ == "__main__":
    query = "总结一下你对台北的了解"
    print(f"❓ 查询：{query}（使用 CrewAI + DeepSeek {MODEL}）")
    print("-" * 60)
    result = run(query)
    print(f"✅ 最终结果：{result['final']}")
    assert result["final"], "期望非空总结"
    print("✅ CrewAI 版本通过 —— 相同任务、不同框架，使用 DeepSeek API")