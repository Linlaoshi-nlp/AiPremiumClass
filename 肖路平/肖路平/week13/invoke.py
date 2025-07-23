import requests
import json

if __name__ == '__main__':
    url = 'http://localhost:11434/api/generate' #generate 一次性文本生成
    data = {
        "model": "deepseek-r1:7b",
        "prompt": "大海是什么颜色的？",#空think或nothink,不思考
        "stream": False,

    }
   
    response = requests.post(url=url, json=data)

    if response.status_code == 200:
        print(response.text) # 打印运行结果
    