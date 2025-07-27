from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

chat_response = client.chat.completions.create(
    model="Qwen3-0.6B-Base",
    messages=[
 {"role": "user", "content": "why is AI so important in today's world?"},
 ]
)

print(chat_response.choices[0].message.content)