from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

response = client.completions.create(
    model="Qwen3-0.6B-Base",
    prompt="讲一个关于NBA的励志故事",
    max_tokens=500
)

print(response.choices[0].text)