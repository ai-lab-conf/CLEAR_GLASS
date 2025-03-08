#pip install langchain-core langchain-openai langchain-chroma chromadb
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
import os
os.environ["OPENAI_API_KEY"] = "sk-xxxx"

example_prompt = PromptTemplate.from_template("Question: {question}\n{answer}")

examples = [
    {
        "question": " ",
        "answer": """ """
    }
]


few_shot_prompt = FewShotPromptTemplate(
    examples=examples,  # List of examples
    example_prompt=example_prompt,
    suffix="Question: {input}",
    input_variables=["input"]
)

llm = ChatOpenAI(
    model_name="xxxx",
    temperature=0.7
)


def get_response(question):
    formatted_prompt = few_shot_prompt.invoke({"input": question})

    response = llm.invoke(formatted_prompt.to_string())
    
    return response.content