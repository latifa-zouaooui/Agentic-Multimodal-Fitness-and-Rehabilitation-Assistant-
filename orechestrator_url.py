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
import requests

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
    """Create web search tool for general information"""
    def search(query: str) -> str:
        search_engine = DuckDuckGoSearchRun()
        return search_engine.run(query)
    
    return Tool(
        name="web_search",
        description="Useful for searching current information on the web about sports, news, or general topics",
        func=search,
    )


def create_media_search_tool() -> Tool:
    """Create specialized tool for finding exercise images and videos with actual URLs"""
    def search_exercise_media(query: str) -> str:
        """
        Search for images and videos of exercises using DDGS
        Returns actual URLs and descriptions
        """
        try:
            from duckduckgo_search import DDGS
            
            print(f"[MEDIA_SEARCH] Searching for visual content: {query}")
            
            ddgs = DDGS()
            
            # Search for YouTube videos
            video_query = f"{query} exercise tutorial"
            print(f"[MEDIA_SEARCH] Video query: {video_query}")
            
            video_results = []
            try:
                results = ddgs.text(video_query, max_results=5)
                for r in results:
                    if 'youtube.com' in r['href'] or 'youtu.be' in r['href']:
                        video_results.append(f"🎥 **{r['title']}**\n   {r['href']}\n   {r['body'][:150]}...\n")
                    if len(video_results) >= 3:
                        break
            except Exception as e:
                print(f"[MEDIA_SEARCH] Video search error: {e}")
            
            # If no YouTube videos found, get general video results
            if not video_results:
                try:
                    results = ddgs.text(video_query + " site:youtube.com", max_results=3)
                    for r in results:
                        video_results.append(f"🎥 **{r['title']}**\n   {r['href']}\n   {r['body'][:150]}...\n")
                except:
                    pass
            
            # Search for images
            image_query = f"{query} exercise form"
            print(f"[MEDIA_SEARCH] Image query: {image_query}")
            
            image_results = []
            try:
                results = ddgs.images(image_query, max_results=5)
                for i, r in enumerate(results):
                    if i >= 3:
                        break
                    image_results.append(f"📸 **Image {i+1}**: {r['title']}\n   {r['image']}\n   Source: {r['url']}\n")
            except Exception as e:
                print(f"[MEDIA_SEARCH] Image search error: {e}")
            
            # Format results
            result = f"## 🎯 Ressources visuelles pour: **{query}**\n\n"
            
            if video_results:
                result += "### 🎥 Vidéos:\n\n"
                result += "\n".join(video_results)
                result += "\n"
            else:
                result += "### 🎥 Vidéos:\n\n"
                result += f"Recherchez sur YouTube: https://www.youtube.com/results?search_query={query.replace(' ', '+')}+exercise+tutorial\n\n"
            
            if image_results:
                result += "### 📸 Images:\n\n"
                result += "\n".join(image_results)
                result += "\n"
            else:
                result += "### 📸 Images:\n\n"
                result += f"Recherchez sur Google Images: https://www.google.com/search?q={query.replace(' ', '+')}+exercise&tbm=isch\n\n"
            
            result += "\n💡 **Astuce:** Cliquez sur les liens pour voir les démonstrations visuelles détaillées!"
            
            print(f"[MEDIA_SEARCH] Found {len(video_results)} videos and {len(image_results)} images")
            return result
            
        except ImportError:
            print("[MEDIA_SEARCH] duckduckgo_search not installed, using fallback")
            # Fallback to basic search
            return f"""## 🎯 Ressources visuelles pour: **{query}**

### 🎥 Vidéos:
YouTube: https://www.youtube.com/results?search_query={query.replace(' ', '+')}+exercise+tutorial

### 📸 Images:
Google Images: https://www.google.com/search?q={query.replace(' ', '+')}+exercise&tbm=isch

### 🌐 Sites recommandés:
- ExRx.net: https://exrx.net/
- Bodybuilding.com: https://www.bodybuilding.com/exercises/
- Fitness Blender: https://www.fitnessblender.com/

💡 **Astuce:** Cliquez sur les liens pour accéder aux démonstrations visuelles!"""
            
        except Exception as e:
            print(f"[MEDIA_SEARCH] Error: {e}")
            return f"""## 🎯 Ressources visuelles pour: **{query}**

### 🎥 Vidéos:
YouTube: https://www.youtube.com/results?search_query={query.replace(' ', '+')}+exercise+tutorial

### 📸 Images:
Google Images: https://www.google.com/search?q={query.replace(' ', '+')}+exercise&tbm=isch

💡 Cliquez sur les liens ci-dessus pour voir les démonstrations!"""
    
    return Tool(
        name="exercise_media_search",
        description="CRITICAL: ALWAYS use this tool when user asks for 'video', 'image', 'montre', 'voir', 'regarder', 'démonstration', 'show', 'watch', or wants visual content of an exercise. This searches YouTube and web for exercise videos and images WITH DIRECT URLs. Input should be the exercise name in English (e.g., 'push-ups' for 'pompes', 'squat' for 'squat').",
        func=search_exercise_media,
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
    1. For ANY question about {specialty}, you MUST use the '{specialty}_knowledge_base' tool FIRST to get exercise information
    2. **CRITICAL**: When users ask for "video", "image", "montre", "voir", "regarder", "démonstration", "show me", "watch", or ANY visual content request, you MUST IMMEDIATELY use the 'exercise_media_search' tool with the exercise name
    3. When using 'exercise_media_search', translate French exercise names to English (e.g., "pompes" → "push-ups", "squat" → "squat", "traction" → "pull-ups")
    4. Only use 'web_search' for non-{specialty} topics or very recent news, NOT for exercise videos/images
    5. After using the knowledge base, you MUST incorporate the information found into your final answer
    6. Never skip using the media search tool when users want visual content
    7. Provide clear, structured, and detailed answers based on the knowledge base
    8. When providing media links, format them nicely and explain what the user will find
    
    EXAMPLES:
    - User: "montre moi une video de pompe" → Use exercise_media_search with "push-ups"
    - User: "show me squat video" → Use exercise_media_search with "squat"
    - User: "je veux voir des images de deadlift" → Use exercise_media_search with "deadlift"
    
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
    media_search_tool = create_media_search_tool()
    
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
    
    # Create specialized agents with media search tool
    gymnastic_agent = create_agent(
        [gymnastic_retriever, media_search_tool, search_tool],
        "gymnastics"
    )
    
    fitness_agent = create_agent(
        [fitness_retriever, media_search_tool, search_tool],
        "fitness"
    )
    
    # Store agents in session
    cl.user_session.set("gymnastic_agent", gymnastic_agent)
    cl.user_session.set("fitness_agent", fitness_agent)
    
    welcome_msg = """👋 Bonjour ! Je suis votre assistant intelligent spécialisé en sport.

Je peux vous aider avec :
🤸 **Gymnastique** : exercices, techniques, entraînement gymnique
💪 **Fitness** : musculation, cardio, routines d'entraînement
🎥 **Ressources visuelles** : vidéos et images d'exercices

**Exemples de questions :**
- "Comment faire un squat correctement ?"
- "Montre-moi une vidéo de pompes"
- "Je veux voir des images de l'exercice du pont"
- "Donne-moi un programme de musculation"

Posez-moi vos questions !"""
    
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