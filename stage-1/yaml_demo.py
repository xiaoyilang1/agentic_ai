import yaml

# 读取 YAML 文件
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)   # safe_load 防止执行恶意代码

print("原始配置：")
print(yaml.dump(config, allow_unicode=True, default_flow_style=False))

# 修改一个值：将 debug 改为 true
config["app"]["debug"] = True

# 修改另一个值：将超时时间改为 60
config["api"]["timeout"] = 60

# 修改用户名
config["user"]["name"] = "Bob"

print("\n修改后的配置：")
print(yaml.dump(config, allow_unicode=True, default_flow_style=False))

# 写回文件（保留原有结构）
with open("config.yaml", "w", encoding="utf-8") as f:
    yaml.dump(
        config,
        f,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False        # 不按键排序，保持插入顺序
    )

print("\n配置已写回 config.yaml")