from agno.agent import Agent
from agno.models.openai import OpenAIChat

from agno.knowledge.langchain import LangChainKnowledgeBase
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from agno.playground import Playground, serve_playground_app
from agno.storage.agent.sqlite import SqliteAgentStorage



import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access the keys directly, as they will already be set in the environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AGNO_API_KEY = os.getenv("agno_API_KEY")

chroma_db_dir = "./db"


# -*- Get the vectordb
db = Chroma(
    collection_name="langchain",  
    persist_directory=chroma_db_dir,   
    embedding_function=OpenAIEmbeddings(), 

)
# -*- Create a retriever from the vector store
retriever = db.as_retriever()

# -*- Create a knowledge base from the vector store
knowledge_base = LangChainKnowledgeBase(retriever=retriever)

storage = SqliteAgentStorage(
    # store sessions in the ai.sessions table
    table_name="agent_sessions",
    # db_file: Sqlite database file
    db_file="tmp/data.db",
)

agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    knowledge=knowledge_base,
    description="You are an expert assistant for question-answering on the Materials Project platform. "
    "Your responses should draw from provided retrieved context and any other resources as needed to address the user's question accurately. "
    "If sufficient context is unavailable, respond based on your extensive knowledge as a materials science expert, offering guidance in the field. "
    "Keep answers concise and informative, with a maximum of four sentences.",

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


# # -*- Create a run
# agent.print_response("Can you explain the different formats in which the data is provided, and how I can convert them for use in machine learning models?", stream=True)
# # -*- Print the messages in the memory
# pprint([m.model_dump(include={"role", "content"}) for m in agent.memory.messages])

# # -*- Ask a follow up question that continues the conversation
# agent.print_response("What types of materials data are available on the Materials Project, and how can I access them?", stream=True)
# # -*- Print the messages in the memory
# pprint([m.model_dump(include={"role", "content"}) for m in agent.memory.messages])

app = Playground(agents=[agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("playground:app", reload=True)