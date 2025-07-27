import os
from dotenv import load_dotenv,find_dotenv
from openai import OpenAI


if __name__ == '__main__':
    load_dotenv(find_dotenv())
    
    client = OpenAI(
        api_key=os.environ["api_key"],
        base_url=os.environ["base_url"]
        )
    
    #chat 方式调用

    response=client.chat.completions.create(
        model="glm-z1-flash",#模型名称
        messages=[ #聊天历史消息   
            {"role":"system","content": "你是一个精通pytihon编程的AI助手"},       
            {"role": "user", "content": "常用的数据类型都有哪些？"},
            {"role": "assistant", "content": "常用的数据类型有：字符串（str）、整数（int）、浮点数（float）、布尔值（bool）、列表（list）、元组（tuple）、字典（dict）、集合（set）"},   
            {"role": "user", "content": "请将这些数据类型都列出来"}        
        ],
        #temperature=0,#采样温度，控制输出的随机性，值越小越 输出越稳定，默认值是0.75,值越大输出越随机，创造性，越小越稳定
        #最大token数，限制输出的token数量，默认值是1024
        max_tokens=500,
        #stream=False,#是否流式输出，默认值是False
        #top_p=1 核取样取值范围：[0.0,1.0]，默认值为0.9，控制取值概率，取概率最大的token
    )

    #token，通常可以理解成文本的基本单位，字词，标点，片段等
    #共分两种，上下文及最大的输出token
    print(response.choices[0].message.content)

 