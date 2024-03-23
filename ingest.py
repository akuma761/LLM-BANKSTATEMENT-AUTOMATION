
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

# Create vector database
def create_vector_db():
    loader = DirectoryLoader(DATA_PATH,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)

    
if __name__ == "__main__":
    # Uncomment if needed or load/define texts appropriately
    # create_vector_db()
    # Define some example texts
    texts = ["This is the first text.", "This is the second text."]
    insights = generate_insights_with_lida(texts)
    # Here you can use the insights, for example, display them with Streamlit or process them further