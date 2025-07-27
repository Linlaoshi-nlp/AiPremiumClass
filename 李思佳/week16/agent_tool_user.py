from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain import hub
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_tool_calling_agent, AgentExecutor

load_dotenv()

# 构建LLM
model = ChatOpenAI(model="gpt-3.5-turbo") #支持function call

# 提示词模版
prompt = hub.pull("hwchase17/openai-functions-agent")

# 构建工具
search= TavilySearchResults(max_results=2)


# print(search.invoke("上海今天的天气？"))

# 工具列表
tools = [search]


agent = create_tool_calling_agent(llm=model, prompt=prompt, tools=tools)

# agent executor
executor = AgentExecutor(agent=agent, tools=tools, verbose=True) #verbose可视化

# 运行agent
msgs = executor.invoke({"input":"北京今天的天气如何，中文回答"})
print(msgs['output'])
