import chainlit as cl
import json
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import Tool
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import initialize_agent, AgentType

# Charger les données
with open("chunks.json", "r", encoding="utf-8") as f:
    loaded_chunks = json.load(f)

print(f"Chargement de {len(loaded_chunks)} chunks...")

# Préparer les documents
docs_processed = [chunk[0] for chunk in loaded_chunks]
docs = [Document(page_content=text) for text in docs_processed]

# Initialiser le modèle d'embedding
embedding_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

# Créer la base vectorielle
vectordb_gymnastic = FAISS.from_documents(
    documents=docs,
    embedding=embedding_model,
    distance_strategy=DistanceStrategy.COSINE,
)

# Configuration du modèle LLM
llm = ChatGroq(
    # api_key="",
    model="llama-3.3-70b-versatile",
    temperature=0.3,
    max_tokens=1024,
    request_timeout=30
)

# Outil de recherche web
def search(query: str) -> str:
    search = DuckDuckGoSearchRun()
    return search.run(query)

search_tool = Tool(
    name="web_search",
    description="Useful for searching current information on the web about sports, news, or general topics",
    func=search,
)

# Outil de récupération de documents - AMÉLIORÉ
def retrieve_documents_gymnastic(query: str) -> str:
    print(f"Recherche dans les documents pour: {query}")  # Debug
    docs = vectordb_gymnastic.similarity_search(query, k=3)
    if docs:
        result = "\n".join([f"- {doc.page_content}" for doc in docs])
        print(f"Documents trouvés: {len(docs)}")  # Debug
        return result
    else:
        print("Aucun document trouvé")  # Debug
        return "No relevant information found in the knowledge base."

retriever_tool_gymnastic = Tool(
    name="gymnastic_knowledge_base",
    description="Useful for searching information about gymnastics exercises, techniques, training methods, and fitness. ALWAYS use this tool when the question is about gymnastics, exercises, training, or physical fitness.",
    func=retrieve_documents_gymnastic,
)

@cl.on_chat_start
async def on_chat_start():
    # Initialiser la mémoire
    memory = ConversationBufferWindowMemory(
        k=3, 
        memory_key="chat_history", 
        return_messages=True,
        input_key="input"
    )
    
    # System prompt AMÉLIORÉ pour forcer l'utilisation des tools
    system_message = """You are a helpful assistant specialized in gymnastics and fitness. 
    
    IMPORTANT RULES:
    1. For ANY question about gymnastics, exercises, training, fitness, or sports techniques, you MUST use the 'gymnastic_knowledge_base' tool FIRST
    2. Only use 'web_search' for non-gymnastics topics or very recent news
    3. After using 'gymnastic_knowledge_base', you MUST incorporate the information found into your final answer
    4. Never skip using the knowledge base for gymnastics-related questions
    
    Think step by step and always follow these rules."""
    
    # Initialiser l'agent avec un prompt système
    agent = initialize_agent(
        tools=[retriever_tool_gymnastic, search_tool],
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,
        handle_parsing_errors=True,
        agent_kwargs={
            'system_message': system_message
        }
    )
    
    # Stocker l'agent dans la session utilisateur
    cl.user_session.set("agent", agent)
    
    await cl.Message(content="Bonjour ! Je suis votre assistant spécialisé en gymnastique. Je peux vous aider avec des exercices, techniques et conseils d'entraînement. Posez-moi vos questions !").send()

@cl.on_message
async def on_message(message: cl.Message):
    # Récupérer l'agent de la session utilisateur
    agent = cl.user_session.get("agent")
    
    # Afficher un indicateur de traitement
    msg = cl.Message(content="")
    await msg.send()
    
    try:
        # Exécuter la requête avec l'agent
        response = await cl.make_async(agent.run)({
            "input": message.content,
            "chat_history": agent.memory.chat_memory.messages
        })
        
        # Mettre à jour le message avec la réponse
        msg.content = response
        await msg.update()
        
    except Exception as e:
        # Gérer les erreurs
        error_msg = f"Une erreur s'est produite : {str(e)}"
        msg.content = error_msg
        await msg.update()
        print(f"Error: {e}")  # Debug