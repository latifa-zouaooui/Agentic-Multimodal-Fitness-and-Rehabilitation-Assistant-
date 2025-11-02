# Import necessary modules
import torch
from typing import List, Dict
#import datasets
from transformers import AutoTokenizer,AutoModelForCausalLM
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from tqdm import tqdm
from huggingface_hub import InferenceClient
from langchain.memory import VectorStoreRetrieverMemory
from langchain_community.docstore import InMemoryDocstore

import json
with open("chunks.json", "r", encoding="utf-8") as f:
    loaded_chunks = json.load(f)
print(loaded_chunks[:2])

docs_processed_yassmine = [chunk[0] for chunk in loaded_chunks]  # Extract text from nested lists

docs_yassmine = [Document(page_content=text) for text in docs_processed_yassmine]

from huggingface_hub import login
login(token="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")

embedding_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

import faiss

vectordb_gymnastic = FAISS.from_documents(
    documents=docs_yassmine,
    embedding=embedding_model,
    distance_strategy=DistanceStrategy.COSINE,
)

"""tokenizer_mistral = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
mistral_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    device_map="auto",
    torch_dtype=torch.float16
)
"""
SYSTEM_PROMPT = """You are a helpful AI assistant. When responding, use this exact format:
Action: tool_name (or "Final Answer")
Action Input: input_text
Observation: result (if using a tool)
Final Answer: your_response (when done).
Also if the question contains seperate parts try to respond on each one seperatly and then combine them in a final answer
"""

from langchain_community.chat_models import ChatHuggingFace
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

"""text_generator = pipeline(
    "text-generation",
    model=mistral_model,
    tokenizer=tokenizer_mistral,
    max_new_tokens=512
    # do_sample=True,
    # temperature=0.7,
    # return_full_text=False
)"""

from langchain_groq import ChatGroq
import os
GROQ_API_KEY=os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    # api_key= "",
    model="llama-3.3-70b-versatile",
    temperature=0.3,
    max_tokens=1024,
    request_timeout=30

  )

# Use the ChatGroq object directly as the language model
# llm_pipeline = HuggingFacePipeline(llm=llm ,model_kwargs={"system_prompt": SYSTEM_PROMPT})
llm_pipeline = llm

from langchain_community.utilities import SerpAPIWrapper

from langchain_community.tools import DuckDuckGoSearchRun

def search(query: str) -> str:
    search = DuckDuckGoSearchRun()
    return search.run(query)

from langchain.tools import Tool

# You can create the tool to pass to an agent
search_tool = Tool(
    name="search",
    description="search on the web to get relevent informations",
    func=search,
)

from langchain.tools import Tool

# Define the retrieval functions
def retrieve_documents_gymnastic(query: str) -> str:
    docs = vectordb_gymnastic.similarity_search(query, k=3)
    if docs :
        return "\n".join([doc.page_content for doc in docs])
    else:
        return "Could not find relevant information."


# Create the Retriever Tools
retriever_tool_gymnastic = Tool(
    name="Gymnastic retriever",
    func=retrieve_documents_gymnastic,
    description="Retrieve relevant documents from the vector store related to GYMNASTIC based on semantic similarity."
)

from langchain.memory import ConversationBufferWindowMemory, ConversationEntityMemory

# buffer_memory = ConversationBufferWindowMemory( k=3,memory_key="chat_history",return_messages=True)
# entity_memory = ConversationEntityMemory(llm=llm_pipeline)

memory = ConversationBufferWindowMemory( k=3,memory_key="chat_history",return_messages=True)

from langchain.agents import initialize_agent, AgentType

agent_executor_informations = initialize_agent(
    tools=[retriever_tool_gymnastic,search_tool],
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
    handle_parsing_errors=True,
)

agent_executor_informations.invoke("what are the jobs in demand in tunisia")

agent_executor_informations.invoke("how to exercice in gymnastic")

agent_executor_informations.invoke("give me gymnastic exercices")
agent_executor_informations.invoke("what is Handstand and how to do it ?")
