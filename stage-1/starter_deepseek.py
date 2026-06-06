"""
Stage 1 练习 5：Error Handling + Retry wrapper — Path B（DeepSeek）。

3 种错误情境 + 1 个 retry wrapper：
1. API key 错（401 AuthenticationError）→ 不要 retry、直接 raise
2. Rate limit（429 RateLimitError）→ exponential backoff retry
3. 网络错误（APIConnectionError）→ exponential backoff retry

运行方法：
pip install -r requirements.txt
export DEEPSEEK_API_KEY=sk-...
python starter_deepseek.py

预算：每次 ≈ $0.0001（只有“情境 2 正常 call”会真的打 API、deepseek-chat）。
"""

from __future__ import annotations

import os
import random
import sys
import time
from typing import Any, Callable
from dotenv import load_dotenv
load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)


# === Retry wrapper ===

RETRIABLE = (APIConnectionError, RateLimitError)
MAX_ATTEMPTS = 4
BASE_DELAY = 1.0  # 秒


def with_retry(fn: Callable[[], Any], *, max_attempts: int = MAX_ATTEMPTS, base_delay: float = BASE_DELAY, sleep_fn=time.sleep) -> Any:
    """
    Exponential backoff retry。
    - RETRIABLE 异常 → 等 base * 2^attempt 秒再试（含 jitter）
    - 不 retriable 异常（譬如 AuthenticationError）→ 直接 raise、不浪费时间
    """
    last_exc = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except RETRIABLE as e:  # noqa: PERF203
            last_exc = e
            if attempt == max_attempts - 1:
                break  # 最后一次失败、break 出来 raise
            delay = base_delay * (2 ** attempt) + random.uniform(0, 0.3)
            print(f"  ⚠ attempt {attempt+1}/{max_attempts} fail ({type(e).__name__}); retry in {delay:.1f}s")
            sleep_fn(delay)
    raise last_exc  # type: ignore[misc]


# === 3 个错误情境 demo ===

def demo_bad_key() -> None:
    """情境 1: 故意用坏 key、看 AuthenticationError 怎么 raise。"""
    print("\n[情境 1] 故意用坏 API key")
    client = OpenAI(api_key="sk-ant-FAKE-KEY-DO-NOT-USE", 
                    base_url="https://api.deepseek.com")
    try:
        client.chat.completions.create(
            model="deepseek-v4-flash",
            max_tokens=10,
            messages=[{"role": "user", "content": "hi"}],
            stream=False,
            extra_body={"thinking": {"type": "disabled"}}
        )
    except AuthenticationError as e:
        print(f"  ✅ 抓到 AuthenticationError: {e.status_code}")
        print(f"  💡 production 处理: 立刻 alert、stop retry（key 不会自己变对）")


def demo_with_retry() -> None:
    """情境 2: 包 with_retry 跑一次正常 call、应该第 1 次就成功"""
    print("\n[情境 2] 正常 call、with_retry 包装")
    api_key = os.environ.get('DEEPSEEK_API_KEY')
    if not api_key:
        raise ValueError("未找到 DEEPSEEK_API_KEY 环境变量，请先设置您的 API 密钥。")
    client = OpenAI(api_key=api_key, 
                    base_url="https://api.deepseek.com")

    def call():
        return client.chat.completions.create(
            model="deepseek-v4-flash",
            max_tokens=10,
            messages=[{"role": "user", "content": "用一个 emoji 回答。"}],
            stream=False,
            extra_body={"thinking": {"type": "disabled"}}
        )

    msg = with_retry(call)
    print(f"  ✅ 成功、第一次就过: [{msg.choices[0].message.content}]")


def demo_too_long_prompt() -> None:
    """情境 3: 故意丢超大 prompt、看 context window 满了怎样"""
    print("\n[情境 3] Prompt 超过 context window")
    api_key = os.environ.get('DEEPSEEK_API_KEY')
    if not api_key:
        raise ValueError("未找到 DEEPSEEK_API_KEY 环境变量，请先设置您的 API 密钥。")
    client = OpenAI(api_key=api_key, 
                    base_url="https://api.deepseek.com")
    huge_prompt = "重复很多次的 token。" * 300_000  # ~1.5M tokens、绝对超过任何 model

    try:
        client.chat.completions.create(
            model="deepseek-chat",
            max_tokens=10,
            messages=[{"role": "user", "content": huge_prompt}],
            stream=False,
            extra_body={"thinking": {"type": "disabled"}}
     )
    except APIStatusError as e:
        print(f"  ✅ 抓到 APIStatusError: {e.status_code}")
        print(f"  💡 生产环境处理: 在 client 端先 count token、超过就拒、别浪费 API call")


if __name__ == "__main__":
    demo_bad_key()
    demo_with_retry()
    demo_too_long_prompt()

    # === 自我验证 ===
    print("\n✅ 练习 5 通过 — 你已了解 3 种错误如何 raise、知道何时该 retry 何时应 stop")