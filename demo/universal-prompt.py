# 通用提示词模板：zero-shot （没有提示，完全基于模型训练的数据完成回答，完全信赖模型 ）
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# 加载模板
prompt_template = PromptTemplate.from_template(
    "我的邻居喜欢：{poet}，你帮我写一首：{story}，言简意赅"
)

load_dotenv()

model = ChatOpenAI(
    model="qwen3.6-plus",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("BASE_URL"),
)
# 普通调用：调用format方法注入信息即可
# prompt_test = prompt_template.format(poet="王维", story="唐诗")
# res = model.invoke(input=prompt_test)
# print(res.content)

# 链式调用：调用format方法注入信息，再调用invoke方法
chain = prompt_template | model
res = chain.invoke(input={"poet": "王维", "story": "唐诗"})
print(res.content)
