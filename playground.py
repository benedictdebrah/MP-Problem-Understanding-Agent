import typer
from rich.prompt import Prompt
from rich.pretty import pprint
from typing import Optional
from phi.agent import Agent
from phi.model.openai import OpenAIChat

from phi.knowledge.langchain import LangChainKnowledgeBase
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from phi.playground import Playground, serve_playground_app


import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access the keys directly, as they will already be set in the environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PHI_API_KEY = os.getenv("PHI_API_KEY")

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
    "If you don’t know the answer, be honest and say you don't know."],

    search_knowledge=True,
    add_history_to_messages=True,
    num_history_responses=3,
    read_chat_history=True,
    show_tool_calls=True,
    markdown=True,
    debug_mode=True,
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