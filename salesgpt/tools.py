'''
Author: sunyilong yilong.sun@miniso.com
Date: 2023-08-22 09:59:23
LastEditors: sunyilong yilong.sun@miniso.com
LastEditTime: 2023-08-22 09:59:23
FilePath: /data02/SalesGPT/salesgpt/tools.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from langchain.agents import Tool
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
import os
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_BASE"] = "https://minisoopenai.openai.azure.com/"
os.environ["OPENAI_API_KEY"] = "2719d64195484f14a3694f4259eae035"
llm = ChatOpenAI(temperature=0, model_kwargs={'engine':"minisoGPT3-5"})

def setup_knowledge_base(product_catalog: str = None):
    """
    We assume that the product catalog is simply a text string.
    """
    # load product catalog
    with open(product_catalog, "r",encoding="UTF-8") as f:
        product_catalog = f.read()
    os.environ["OPENAI_API_TYPE"] = "azure"
    os.environ["OPENAI_API_BASE"] = "https://minisoopenai.openai.azure.com/"
    os.environ["OPENAI_API_KEY"] = "2719d64195484f14a3694f4259eae035"

    text_splitter = CharacterTextSplitter(chunk_size=10, chunk_overlap=0)
    texts = text_splitter.split_text(product_catalog)

    llm = OpenAI(temperature=0)
    # embeddings = OpenAIEmbeddings()
    embeddings = OpenAIEmbeddings(deployment="embedding-2")
    docsearch = Chroma.from_texts(
        texts, embeddings, collection_name="product-knowledge-base"
    )

    knowledge_base = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=docsearch.as_retriever()
    )
    return knowledge_base


def get_tools(knowledge_base):
    # we only use one tool for now, but this is highly extensible!
    tools = [
        Tool(
            name="ProductSearch",
            func=knowledge_base.run,
            description="useful for when you need to answer questions about product information",
        )
    ]

    return tools
