
from lida import Manager, TextGenerationConfig , llm  
from dotenv import load_dotenv
import os
import openai
from PIL import Image
from io import BytesIO
import base64
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from llama_index.core import ServiceContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import PromptHelper
from llama_index.llms.openai import OpenAI
from llama_index.core.text_splitter import TokenTextSplitter
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from flashrank.Ranker import Ranker, RerankRequest
import streamlit as st
import json
import tiktoken 


os.environ["OPENAI_API_KEY"] = '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'



DATA_PATH = 'data/'
DB_FAISS_PATH = 'vectorstore/db_faiss'


def base64_to_image(base64_string):
    # Decode the base64 string
    byte_data = base64.b64decode(base64_string)
    
    # Use BytesIO to convert the byte data to image
    return Image.open(BytesIO(byte_data))

# New: Generate insights from texts using LIDA
def generate_insights_with_lida(texts):
    insights = []
    # Initialize your LIDA Manager here with the appropriate configuration
    lida = Manager(text_gen = llm("openai"))  # Placeholder: Fill in with actual initialization parameters
    textgen_config = TextGenerationConfig(n=1, temperature=0.5, model="gpt-3.5-turbo-0301", use_cache=True)
    
    for text in texts:
        # Example of generating a summary and insights for each text
        summary = lida.summarize(text, summary_method="default", textgen_config=textgen_config)
        goals = lida.goals(summary, n=2, textgen_config=textgen_config)
        # Assuming you want to visualize the first goal
        library = "seaborn"
        charts = lida.visualize(summary=summary, goal=goals[0], textgen_config=textgen_config, library=library)
        img_base64_string = charts[0].raster
        img = base64_to_image(img_base64_string)
        # Append your insights or relevant information here
        insights.append({"summary": summary, "goals": goals, "image": img})
    return insights

def vectorstoreindex():

    llm = OpenAI(model='gpt-3.5-turbo', temperature=0, max_tokens=256)
    embed_model = OpenAIEmbedding()
    text_splitter = TokenTextSplitter(
    separator=" ",
    chunk_size=1024,
    chunk_overlap=20,
    backup_separators=["\n"],
    tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode
    )

    # node_parser = SimpleNodeParser.from_defaults(
    # text_splitter=text_splitter
    # )
    
    prompt_helper = PromptHelper(
    context_window=4096, 
    num_output=256, 
    chunk_overlap_ratio=0.1, 
    chunk_size_limit=None
    )

    service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embed_model,
    text_splitter=text_splitter,
    prompt_helper=prompt_helper
    )

    documents = SimpleDirectoryReader(input_dir=DATA_PATH).load_data()
    index = VectorStoreIndex.from_documents(
        documents, 
        service_context = service_context
        )
    index.storage_context.persist()
    query_engine = index.as_query_engine(service_context=service_context)

    return query_engine

# # Create vector database
# def create_vector_db():
#     loader = DirectoryLoader(DATA_PATH,
#                              glob='*.pdf',
#                              loader_cls=PyPDFLoader)

#     documents = loader.load()
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
#                                                    chunk_overlap=50)
#     texts = text_splitter.split_documents(documents)

#     embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
#                                        model_kwargs={'device': 'cpu'})

#     db = FAISS.from_documents(texts, embeddings)
#     db.save_local(DB_FAISS_PATH)

#     return texts



def get_result(query,choice):
    if choice == "Nano":
        ranker = Ranker()
    elif choice == "Small":
        ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/opt")
    elif choice == "Medium":
        ranker = Ranker(model_name="rank-T5-flan", cache_dir="/opt")
    elif choice == "Large":
        ranker = Ranker(model_name="ms-marco-MultiBERT-L-12", cache_dir="/opt")

    rerankrequest = RerankRequest(query=query)
    results = ranker.rerank(rerankrequest)
    print(results)

    
if __name__ == "__main__":
    # Uncomment if needed or load/define texts appropriately
     query_engine = vectorstoreindex()
     response = query_engine.query("What was payment from bandhan Bank?")
     print(response)
     get_result(response,"Medium")
    # Define some example texts
    # texts = ["This is the first text.", "This is the second text."]
     #insights = generate_insights_with_lida(texts)
    # Here you can use the insights, for example, display them with Streamlit or process them further