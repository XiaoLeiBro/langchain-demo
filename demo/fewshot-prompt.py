# 样本提示词模板：few-shot

from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# 创建样本提示词模板
example_prompt = PromptTemplate = PromptTemplate.from_template("单词:{word},反义词:{antonym}")
# 样本的懂爱注入数据：list内部套字典
example_data = [
    {"word": "大", "antonym": "小"},
    {"word": "上", "antonym": "下"},
    {"word": "多", "antonym": "少"},
]

few_shot_prompt_template = FewShotPromptTemplate(
    example_prompt=example_prompt,  # 样本提示词模板
    examples=example_data,  # 样本数据（用来注入动态数据的），list内套字典
    prefix="告诉我单词的反义词，我提供如下的示例：",  # 提示词前缀：样本之前的提示词（提示词模板内可以有动态参数，通过prefix和suffix拼接成最终的提示词）
    suffix="基于前面的示例告诉我：{input_word}的反义词是？",  # 提示词后缀：样本之后的提示词 （提示词模板内可以有动态参数，通过prefix和suffix拼接成最终的提示词）
    input_variables=["input_word"],  # 提示词动态参数：声明在前缀或后缀所需要注入的变量名
)

prompt_test = few_shot_prompt_template.invoke(input={"input_word": "左"}).to_string()

# print(prompt_test)

load_dotenv()

model = ChatOpenAI(
    model="qwen3.6-plus",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("BASE_URL"),
)
res = model.invoke(input=prompt_test)
print(res.content)
