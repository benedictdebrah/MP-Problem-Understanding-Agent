import getpass
import os
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder


# Load environment variables from .env file
load_dotenv()

# Access API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true")  # Default to "true" if not set in .env

# Initialize the language model
llm = ChatOpenAI(model="gpt-4o-mini")

# Load your pre-existing vector database with embeddings
vectorstore = Chroma(
    collection_name="langchain",  
    persist_directory="./db",   
    embedding_function=OpenAIEmbeddings(), 
)
retriever = vectorstore.as_retriever()

# Define system prompt for question-answering
system_prompt = (
    "You are an assistant for question-answering tasks for the materials project platform. "
    "Use the following pieces of retrieved context and other external resources to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use four sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

# Set up the prompt template for QA
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Initialize history-aware retriever
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        ("chat_history", "{chat_history}"),  # Placeholder for chat history
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm=llm, retriever=retriever, prompt=contextualize_q_prompt
)

# Set up QA chain with history-aware retriever
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Initialize chat history
chat_history = []

print("Chatbot ready! Type 'exit' to end the conversation.")

# Start continuous conversation
while True:
    # Get user input
    user_input = input("You: ")
    
    # End conversation if user types 'exit'
    if user_input.lower() == "exit":
        print("Ending conversation.")
        break

    # Invoke the RAG chain with the current user input and chat history
    ai_response = rag_chain.invoke({"input": user_input, "chat_history": chat_history})
    
    # Print AI response
    print("AI:", ai_response["answer"])
    
    # Update chat history with the new interaction
    chat_history.extend([
        HumanMessage(content=user_input),
        AIMessage(content=ai_response["answer"]),
    ])
