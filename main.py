import os
from dotenv import load_dotenv
from supabase.client import Client, create_client
from langchain import LLMChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.vectorstores import SupabaseVectorStore
from langchain.schema import (
    SystemMessage
)
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

load_dotenv()

supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

embeddings = OpenAIEmbeddings()

vector_store = SupabaseVectorStore(
    supabase, 
    embeddings, 
    table_name=os.environ.get("TABLE_NAME"),
    query_name="youtube_transcript_search" ## make sure this is the same as the query name in supabase
)

while True:
    query = input("\033[34mWhat question do you have about these videos?\n\033[0m")

    if query.lower().strip() == "exit":
        print("\033[31mGoodbye!\n\033[0m")
        break

    matched_docs = vector_store.similarity_search(query)
    transcript_str = ""

    for doc in matched_docs:
        transcript_str += doc.page_content + "\n\n"
        
    print("\n\033[35m" + transcript_str + "\n\033[32m")

    
    template="""
    You are YoutubeTranscribe AI. You are a superintelligent AI that answers questions about youtube videos.

    You are:
    - helpful & friendly
    - good at answering complex questions in simple language
    - able to infer the intent of the user's question

    The user will ask a question about the videos provided, and you will answer it.

    When the user asks their question, you will answer it by searching the videos for the answer.

    Here is the user's question and text from the video(s) you found to answer the question:

    Question:
    {query}

    Video file(s):
    {video_files}
    
    [END OF VIDEO FILE(S)]w

    Now answer the question using the video file(s) above.
    """

    chat = ChatOpenAI(streaming=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), verbose=True, temperature = 0.8)
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt])
    chain = LLMChain(llm=chat, prompt=chat_prompt)

    chain.run(video_files=transcript_str, query=query)

    print("\n\n")