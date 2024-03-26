from llama_index.core import ServiceContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import PromptHelper
from llama_index.llms.openai import OpenAI
from llama_index.core.text_splitter import TokenTextSplitter
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
import tiktoken 
import os
import openai

os.environ["OPENAI_API_KEY"] = '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'


DATA_PATH = 'data/'
DB_FAISS_PATH = 'vectorstore/db_faiss'

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

#query_engine = index.as_query_engine(service_context=service_context)
query_engine = vectorstoreindex()
response = query_engine.query("What is HNSW?")
print(response)