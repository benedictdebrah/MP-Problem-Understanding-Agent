import os
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.knowledge.langchain import LangChainKnowledgeBase
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from agno.storage.agent.sqlite import SqliteAgentStorage

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AGNO_API_KEY = os.getenv("agno_API_KEY")
chroma_db_dir = "./db"

# Initialize your vector database and retriever
db = Chroma(
    collection_name="langchain",
    persist_directory=chroma_db_dir,
    embedding_function=OpenAIEmbeddings(),
)
retriever = db.as_retriever()
knowledge_base = LangChainKnowledgeBase(retriever=retriever)

# Setup agent storage
storage = SqliteAgentStorage(
    table_name="agent_sessions",
    db_file="tmp/data.db",
)

# Create the agent
agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    knowledge=knowledge_base,
    description=(
        "You are an expert assistant for question-answering on the Materials Project platform. "
        "Your responses should draw from provided retrieved context and any other resources as needed to address the user's question accurately. "
        "If sufficient context is unavailable, respond based on your extensive knowledge as a materials science expert, offering guidance in the field. "
        "Keep answers concise and informative, with a maximum of four sentences."
    ),
    instructions=[
        "Analyze each question to determine if retrieved context provides a complete answer.",
        "If the context is sufficient, use it directly to answer the question concisely.",
        "If the retrieved context is insufficient, draw from your expert knowledge in materials science to provide accurate guidance.",
        "Limit responses to four sentences, focusing on clarity and relevance.",
        "Do not provide wrong citations or references.",
        "If referencing external sources or links, explicitly state that the provided links or references may not work or may not be accurate and encourage users to verify the information.",
        "If you don’t know the answer, be honest and say you don't know."
    ],
    search_knowledge=True,
    add_history_to_messages=True,
    num_history_responses=3,
    read_chat_history=True,
    show_tool_calls=True,
    markdown=True,
    debug_mode=True,
    storage=storage,
)
