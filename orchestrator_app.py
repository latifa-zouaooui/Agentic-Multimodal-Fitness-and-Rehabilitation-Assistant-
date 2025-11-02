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
from langchain.schema import HumanMessage

# Configuration
# GROQ_API_KEY = ""
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

# Initialize embedding model (shared)
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# Initialize LLMs
llm_agent = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.3-70b-versatile",
    temperature=0.3,
    max_tokens=1024,
    request_timeout=30
)

llm_orchestrator = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.3-70b-versatile",
    temperature=0.1,
    max_tokens=256,
    request_timeout=30
)


def load_vectordb(json_file: str) -> FAISS:
    """Load and create FAISS vectorstore from JSON chunks"""
    with open(json_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    print(f"Loading {len(chunks)} chunks from {json_file}...")
    
    docs_processed = [chunk[0] for chunk in chunks]
    docs = [Document(page_content=text) for text in docs_processed]
    
    return FAISS.from_documents(
        documents=docs,
        embedding=embedding_model,
        distance_strategy=DistanceStrategy.COSINE,
    )


# Load vector databases
vectordb_gymnastic = load_vectordb("chunks.json")
vectordb_fitness = load_vectordb("fusion_chunks.json")


def create_retriever_tool(vectordb: FAISS, name: str, description: str) -> Tool:
    """Create a retriever tool for a specific knowledge base"""
    def retrieve_documents(query: str) -> str:
        print(f"[{name}] Searching for: {query}")
        docs = vectordb.similarity_search(query, k=3)
        
        if docs:
            result = "\n".join([f"- {doc.page_content}" for doc in docs])
            print(f"[{name}] Found {len(docs)} documents")
            return result
        else:
            print(f"[{name}] No documents found")
            return "No relevant information found in the knowledge base."
    
    return Tool(name=name, description=description, func=retrieve_documents)


def create_search_tool() -> Tool:
    """Create web search tool"""
    def search(query: str) -> str:
        search_engine = DuckDuckGoSearchRun()
        return search_engine.run(query)
    
    return Tool(
        name="web_search",
        description="Useful for searching current information on the web about sports, news, or general topics",
        func=search,
    )


def create_agent(tools: list, specialty: str) -> any:
    """Create a specialized agent"""
    memory = ConversationBufferWindowMemory(
        k=3,
        memory_key="chat_history",
        return_messages=True,
        input_key="input"
    )
    
    system_message = f"""You are a helpful assistant specialized in {specialty}.
    
    IMPORTANT RULES:
    1. For ANY question about {specialty}, you MUST use the '{specialty}_knowledge_base' tool FIRST
    2. Only use 'web_search' for non-{specialty} topics or very recent news
    3. After using the knowledge base, you MUST incorporate the information found into your final answer
    4. Never skip using the knowledge base for {specialty}-related questions
    5. Provide clear, structured, and detailed answers based on the knowledge base
    
    Think step by step and always follow these rules."""
    
    return initialize_agent(
        tools=tools,
        llm=llm_agent,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,
        handle_parsing_errors=True,
        agent_kwargs={'system_message': system_message}
    )


def route_query(user_query: str) -> str:
    """Orchestrator: Routes query to appropriate agent using LLM"""
    
    routing_prompt = f"""You are a query routing expert. Analyze the user's question and determine which specialized agent should handle it.

Available agents:
- GYMNASTIC: Handles questions about gymnastics, gymnastic exercises, gymnastic techniques, gymnastic training, apparatus work, floor exercises, rings, bars, vault, beam, tumbling, and competitive gymnastics
- FITNESS: Handles questions about general fitness, workout routines, strength training, cardio, muscle building, weight loss, nutrition, conditioning, and general physical training (excluding gymnastics)

User question: "{user_query}"

Respond with ONLY ONE WORD - either "GYMNASTIC" or "FITNESS" based on which agent is most appropriate.
If the question is about gymnastics or gymnastic-specific techniques, respond with "GYMNASTIC".
If the question is about general fitness, training, or exercises (not gymnastics), respond with "FITNESS".

Your response (one word only):"""

    try:
        response = llm_orchestrator.invoke([HumanMessage(content=routing_prompt)])
        route = response.content.strip().upper()
        
        # Validate response
        if "GYMNASTIC" in route:
            return "GYMNASTIC"
        elif "FITNESS" in route:
            return "FITNESS"
        else:
            # Default to FITNESS for general fitness questions
            return "FITNESS"
            
    except Exception as e:
        print(f"Routing error: {e}")
        return "FITNESS"  # Default fallback


@cl.on_chat_start
async def on_chat_start():
    """Initialize agents on chat start"""
    
    # Create tools
    search_tool = create_search_tool()
    
    gymnastic_retriever = create_retriever_tool(
        vectordb_gymnastic,
        "gymnastic_knowledge_base",
        "Useful for searching information about gymnastics exercises, techniques, training methods, and gymnastic-specific skills. ALWAYS use this tool when the question is about gymnastics."
    )
    
    fitness_retriever = create_retriever_tool(
        vectordb_fitness,
        "fitness_knowledge_base",
        "Useful for searching information about fitness exercises, workout routines, strength training, and general physical fitness. ALWAYS use this tool when the question is about fitness."
    )
    
    # Create specialized agents
    gymnastic_agent = create_agent(
        [gymnastic_retriever, search_tool],
        "gymnastics"
    )
    
    fitness_agent = create_agent(
        [fitness_retriever, search_tool],
        "fitness"
    )
    
    # Store agents in session
    cl.user_session.set("gymnastic_agent", gymnastic_agent)
    cl.user_session.set("fitness_agent", fitness_agent)
    
    welcome_msg = """👋 Bonjour ! Je suis votre assistant intelligent spécialisé en sport.

Je peux vous aider avec :
🤸 **Gymnastique** : exercices, techniques, entraînement gymnique
💪 **Fitness** : musculation, cardio, routines d'entraînement

Posez-moi vos questions, je vous dirigerai vers le bon expert !"""
    
    await cl.Message(content=welcome_msg).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming messages with orchestration"""
    
    # Show processing indicator
    msg = cl.Message(content="🤔 Analyse de votre question...")
    await msg.send()
    
    try:
        # Route the query
        route = route_query(message.content)
        print(f"\n{'='*50}")
        print(f"ORCHESTRATOR ROUTING: {route}")
        print(f"{'='*50}\n")
        
        # Get appropriate agent
        if route == "GYMNASTIC":
            agent = cl.user_session.get("gymnastic_agent")
            agent_name = "🤸 Agent Gymnastique"
        else:  # FITNESS
            agent = cl.user_session.get("fitness_agent")
            agent_name = "💪 Agent Fitness"
        
        # Update message to show which agent is handling the query
        msg.content = f"{agent_name} traite votre question..."
        await msg.update()
        
        # Execute query with selected agent
        response = await cl.make_async(agent.run)({
            "input": message.content,
            "chat_history": agent.memory.chat_memory.messages
        })
        
        # Update with final response
        msg.content = f"**{agent_name}**\n\n{response}"
        await msg.update()
        
    except Exception as e:
        error_msg = f"❌ Une erreur s'est produite : {str(e)}"
        msg.content = error_msg
        await msg.update()
        print(f"Error: {e}")