import requests
import os
from dotenv import load_dotenv

load_dotenv()
# 从环境变量读取 token（推荐做法，不要硬编码在代码里）
TOKEN = os.environ.get("GITHUB_TOKEN")

url = "https://api.github.com/user"

# 请求1：不带 token（会收到 401）
print("=== 无认证请求 ===")
resp_no_auth = requests.get(url)
print(f"状态码: {resp_no_auth.status_code}")
if resp_no_auth.status_code == 401:
    print("响应体:", resp_no_auth.json().get("message"))
print()

# 请求2：带 token
if TOKEN:
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Accept": "application/vnd.github+json"  # 推荐 GitHub API 版本头
    }
    print("=== 带认证请求 ===")
    resp_auth = requests.get(url, headers=headers)
    print(f"状态码: {resp_auth.status_code}")
    if resp_auth.status_code == 200:
        user_data = resp_auth.json()
        print(f"用户名: {user_data['login']}")
        print(f"用户ID: {user_data['id']}")
        print(f"公开仓库数: {user_data['public_repos']}")
        print(f"头像URL: {user_data['avatar_url']}")
    else:
        print("请求失败:", resp_auth.json())
else:
    print("请先设置环境变量 GITHUB_TOKEN")