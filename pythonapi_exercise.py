import requests

def get_followers_count(username):
    # GitHub API 端点
    url = f"https://api.github.com/users/{username}"
    
    # 设置 User-Agent 头（GitHub API 建议添加，否则可能会被拒绝请求）
    headers = {
        "User-Agent": "Python-Script"
    }
    
    try:
        # 发送 GET 请求
        response = requests.get(url, headers=headers)
        
        # 检查请求是否成功 (状态码 200 表示成功)
        response.raise_for_status()
        
        # 将响应内容解析为 JSON 字典
        data = response.json()
        
        # 提取 followers 数量
        followers_count = data.get("followers", 0)
        
        print(f"用户 {username} 的 follower 数量为: {followers_count}")
        
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP 错误发生: {http_err}")
    except Exception as err:
        print(f"其他错误发生: {err}")

def api_demo():
    # 发送 GET 请求
    response = requests.get("https://httpbin.org/get")

    # 输出状态码（200 表示成功）
    if response.status_code == 200:
        data = response.json()
        # 处理数据
    else:
        print("请求失败，状态码:", response.status_code)

    # 输出响应体（字符串）
    print("Response text:")
    print(response.text)


    # 现在可以像操作字典一样访问数据
    print("我的IP:", data["origin"])
    print("请求URL:", data["url"])

    # 使用 params 参数发送 GET 请求
    params = {
    "search": "python",
    "page": 1
    }
    response = requests.get("https://httpbin.org/get", params=params)
    print(response.url)  # 会显示 https://httpbin.org/get?search=python&page=1
    if response.status_code == 200:
        print(response.json()["args"])  # 服务器解析出的参数
        # 处理数据
    else:
        print("请求失败，状态码:", response.status_code)
    

    # 发送 POST 请求
    payload = {
    "username": "deepseek",
    "task": "learn REST API"
    }

    response = requests.post("https://httpbin.org/post", json=payload)
    data = response.json()

    print("我们发送的JSON:", data["json"])        # httpbin 原样返回发送的 JSON
    print("服务器收到的 headers:", data["headers"]["Content-Type"])  # application/json
    #如果 API 要求表单格式（application/x-www-form-urlencoded），使用 data 参数：
    response = requests.post("https://httpbin.org/post", data={"key": "value"})
    print("我们发送的JSON:", data["json"])        # httpbin 原样返回发送的 JSON
    print("服务器收到的 headers:", data["headers"]["Content-Type"])  # application/json


    #调用一个真实 API
    username = "xiaoyilang1"  # 换成你的 GitHub 用户名
    url = f"https://api.github.com/users/{username}"
    response = requests.get(url)

    if response.status_code == 200:
        user = response.json()
        print(f"用户 {user['login']}，仓库数：{user['public_repos']}")
    else:
        print("请求失败", response.status_code)

if __name__ == "__main__":
    get_followers_count("torvalds")
    api_demo()
