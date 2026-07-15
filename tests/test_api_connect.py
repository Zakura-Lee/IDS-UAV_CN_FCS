import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"  # 使用标准 v1 端点
)

response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[{"role": "user", "content": "Say 'Hello, API works!'"}],
    max_tokens=50,
    extra_body={"thinking": {"type": "enabled"}}  # 核心：显式启用思考模式
)

reasoning = getattr(response.choices[0].message, "reasoning_content", None)  # 获取推理过程
if reasoning:
    print("Reasoning:", reasoning)  # 打印推理过程
print("Answer:", response.choices[0].message.content)  # 打印最终答案