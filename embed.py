import os
import datetime
from dotenv import load_dotenv
from supabase.client import Client, create_client
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import SupabaseVectorStore
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders.csv_loader import CSVLoader

load_dotenv()

supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

loader = CSVLoader(file_path='./csv-loader/testloader.csv') # put .csv in the csv-loader folder and change the file name here
youtube_url_documents = loader.load()
youtube_url_list = [document.page_content for document in youtube_url_documents]

all_documents = []

for url in youtube_url_list:
    loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
    documents = loader.load()
    all_documents.extend(documents)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
docs = text_splitter.split_documents(all_documents)

for doc in docs:
    if isinstance(doc.metadata.get('publish_date'), datetime.datetime):
        doc.metadata['publish_date'] = doc.metadata['publish_date'].isoformat()

embeddings = OpenAIEmbeddings()

vector_store = SupabaseVectorStore.from_documents(
    docs,
    embeddings,
    client=supabase,
    table_name=os.environ.get("TABLE_NAME"),
)

for doc in docs:
    print(doc.metadata)