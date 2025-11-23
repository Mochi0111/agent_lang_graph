import faiss
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS, InMemoryVectorStore

load_dotenv("/lg_agent/.env", override=True)

loader = CSVLoader(file_path="/lg_agent/app/dummy.csv")
docs = loader.load()

embedding = OpenAIEmbeddings(model="text-embedding-ada-002")

vector_store = FAISS.from_documents(
    documents=docs,
    embedding=embedding,   
)

vector_store.save_local("/lg_agent/app/faiss_store")
