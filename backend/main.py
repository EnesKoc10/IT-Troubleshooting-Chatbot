import argparse
import logging
import os
import re
import uuid
from typing import Annotated, Optional, List, Dict, Any
from typing_extensions import TypedDict
import json
import asyncio
import sqlite3

from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


class SimpleSqliteSaver:
    def __init__(self, db_path):
        self.db_path = db_path
        import sqlite3
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                thread_id TEXT,
                checkpoint_id TEXT,
                data TEXT,
                PRIMARY KEY (thread_id, checkpoint_id)
            )
        """)
        self.conn.commit()
    
    def get(self, config):
        # Basit implementasyon
        return None
    
    def put(self, config, checkpoint, metadata=None):
        # Basit implementasyon
        pass
    
    def get_next_version(self, config, checkpoint):
        # Basit implementasyon
        return "1"


# Import RAG functionality - simplified version
def query_qdrant(qdrant_url: str, qdrant_port: int, qdrant_api_key: str, collection: str, query: str, k: int, model_name: str):
    """Simplified Qdrant query function."""
    try:
        import requests
        # For now, return empty results - will be implemented when Qdrant is ready
        return []
    except Exception as e:
        print(f"Error querying Qdrant: {e}")
        return []

class EmbeddingModel:
    """Simplified embedding model."""
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
    
    def encode(self, texts):
        """Simplified encoding - returns dummy embeddings."""
        return [[0.1] * 384 for _ in texts]

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="BT Destek Asistanı", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class State(TypedDict):
    """
    Kalıcı hafızalı BT destek konuşma durumu.
    """
    messages: Annotated[list, add_messages]
    user_profile: dict
    ticket_id: str
    issue_category: str
    priority: str
    status: str

class ChatRequest(BaseModel):
    message: str
    thread_id: str
    user_info: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    response: str
    thread_id: str
    ticket_id: str
    issue_category: str
    priority: str
    status: str
    relevant_docs: List[Dict[str, Any]]

# Persistent memory (SQLite) ----------------------------------------------------

MEMORY_DB_PATH = os.getenv("MEMORY_DB_PATH", "/app/data/memory.db")


class ConversationMemory:
    """
    Very small SQLite-backed memory to persist chat messages by thread_id.
    Schema:
      messages(thread_id TEXT, role TEXT, content TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._ensure_db()

    def _ensure_db(self) -> None:
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    thread_id TEXT,
                    role TEXT CHECK(role IN ('user','assistant')),
                    content TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS profiles (
                    thread_id TEXT PRIMARY KEY,
                    data TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def load_messages(self, thread_id: str) -> List[Dict[str, str]]:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        try:
            cur = conn.execute(
                "SELECT role, content FROM messages WHERE thread_id = ? ORDER BY created_at ASC",
                (thread_id,),
            )
            rows = cur.fetchall()
            return [{"role": r, "content": c} for (r, c) in rows]
        finally:
            conn.close()

    def append_message(self, thread_id: str, role: str, content: str) -> None:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        try:
            conn.execute(
                "INSERT INTO messages(thread_id, role, content) VALUES (?,?,?)",
                (thread_id, role, content),
            )
            conn.commit()
        finally:
            conn.close()

    def load_profile(self, thread_id: str) -> Dict[str, Any]:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        try:
            cur = conn.execute(
                "SELECT data FROM profiles WHERE thread_id = ?",
                (thread_id,),
            )
            row = cur.fetchone()
            if not row:
                return {}
            try:
                return json.loads(row[0]) or {}
            except Exception:
                return {}
        finally:
            conn.close()

    def save_profile(self, thread_id: str, profile: Dict[str, Any]) -> None:
        data = json.dumps(profile or {}, ensure_ascii=False)
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        try:
            conn.execute(
                 (thread_id, data),
            )
            conn.commit()
        finally:
            conn.close()


memory_store = ConversationMemory(MEMORY_DB_PATH)


def to_langchain_messages(history: List[Dict[str, str]]):
    converted: List[Any] = []
    for item in history:
        role = (item.get("role") or "").lower()
        content = item.get("content") or ""
        if role == "user":
            converted.append(HumanMessage(content=content))
        elif role == "assistant":
            converted.append(AIMessage(content=content))
    return converted

# Structured output schema for LLM classification
class IssueClassification(BaseModel):
    category: str = Field(description=(
        "One of: network, hardware, software, email, security, database, server, general"
    ))
    priority: str = Field(description=(
        "One of: high, medium, low"
    ))

# Initialize LLMs: OpenAI for classification, Gemini for response generation
openai_api_key = os.getenv("OPENAI_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY ortam değişkenlerinde bulunamadı")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY ortam değişkenlerinde bulunamadı")

llm_openai = init_chat_model(
    "gpt-4o-mini",
    model_provider="openai",
    api_key=openai_api_key,
    temperature=0.5,
)

llm_gemini = init_chat_model(
    "gemini-2.5-flash",
    model_provider="google_genai",
    api_key=google_api_key,
    temperature=0.2,
)

# RAG Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "it_support_kb")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def classify_issue_with_llm(text: str) -> Dict[str, str]:
    """
    Use the LLM with structured output to classify category and priority.
    Falls back to empty dict if the provider doesn't support structured output.
    """
    try:
        structured = llm_openai.with_structured_output(IssueClassification)
        prompt = (
         "Sen bir BT destek sınıflandırıcısısın. Kullanıcı mesajına göre kategori ve öncelik belirle.\n"
        "Kategoriler: network, hardware, software, email, security, database, server, general.\n"
        "Öncelik: high, medium, low.\n"
        "Sadece geçerli kategori ve öncelik isimlerini kullanarak JSON/Yapılandırılmış çıktıyı doldur.\n\n"
        f"Kullanıcı mesajı: {text}"
         )
        result = structured.invoke(prompt)
        if isinstance(result, IssueClassification):
            raw = {"category": result.category, "priority": result.priority}
        elif isinstance(result, dict):
            raw = {"category": result.get("category", ""), "priority": result.get("priority", "")}
        else:
            raw = {}

        category = (raw.get("category") or "").strip().lower()
        priority = (raw.get("priority") or "").strip().lower()

        valid_categories = {"network", "hardware", "software", "email", "security", "database", "server", "general"}
        valid_priorities = {"high", "medium", "low"}

        result_norm: Dict[str, str] = {}
        if category in valid_categories:
            result_norm["category"] = category
        if priority in valid_priorities:
            result_norm["priority"] = priority
        return result_norm

    except Exception as e:
        logger.warning(f"LLM classification failed: {e}")
        return {}

def get_relevant_documents(query: str, k: int = 3) -> List[Dict[str, Any]]:
    """
    Bilgi bankasından ilgili dokümanları RAG ile getir.
    """
    try:
        results = query_qdrant(
            qdrant_url=QDRANT_URL,
            qdrant_port=QDRANT_PORT,
            qdrant_api_key=QDRANT_API_KEY,
            collection=QDRANT_COLLECTION,
            query=query,
            k=k,
            model_name=MODEL_NAME
        )
        
        docs = []
        for meta, score in results:
            docs.append({
                "text": meta["text"],
                "source": meta.get("source", "Bilinmiyor"),
                "page": meta.get("page", 0),
                "score": score
            })
        return docs
    except Exception as e:
        logger.error(f"Dokümanlar alınırken hata: {e}")
        return []

def create_system_prompt(issue_category: str, priority: str, relevant_docs: List[Dict[str, Any]], user_profile: Optional[Dict[str, Any]] = None) -> str:
    """
    BT destek asistanı için sistem yönergesini oluştur.
    """
    base_prompt = """Sen, deneyimli bir BT destek asistanısın. Görevin, kullanıcıların teknik sorunlarını hızlı, net ve profesyonel şekilde çözmelerine yardımcı olmak.

    İlkeler:
    1. Yardımcı, sabırlı ve profesyonel ol
    2. Gerekirse netleştirici sorular sor ama birdan birden fazla soru sorma
    3. Adım adım, uygulanabilir çözümler ver
    4. Sadece Türkçe cevap ver
    5. Kısa cevaplar ver

    Kullanıcı Profili:
    - Category: {category}
    - Priority: {priority}
    {user_profile_text}

    İlgili Bilgi Bankası Dokümanları:
    {knowledge_base}

    Unutma:
    - Basit kontrolden başlayıp karmaşığa doğru ilerle
    - Varsa spesifik hata mesajlarını iste
    - Mümkünse alternatif çözüm yolları sun
    - Sonunda bir sonraki adımı veya gerekirse yönlendirmeyi belirt
    """

    knowledge_text = ""
    if relevant_docs:
        knowledge_text = "\n".join([
            f"Source: {doc['source']} (Page {doc['page']})\n{doc['text']}\n"
            for doc in relevant_docs
        ])
    else:
        knowledge_text = "Bu sorun için özel bir doküman bulunamadı."

    user_profile_text = ""
    if user_profile:
        name = (user_profile.get("ad") or user_profile.get("isim") or user_profile.get("name") or "").strip()
        if name:
            user_profile_text += f"- İsim: {name}\n"
        for key in ("departman", "email", "lokasyon"):
            if user_profile.get(key):
                user_profile_text += f"- {key.title()}: {user_profile.get(key)}\n"
    if not user_profile_text:
        user_profile_text = "(profil bilgisi yok)"

    return base_prompt.format(
        category=issue_category or "general",
        priority=priority or "medium",
        knowledge_base=knowledge_text,
        user_profile_text=user_profile_text
    )

def it_support_agent(state: State):
    """
    BT destek ajanının ana fonksiyonu.
    """
    last_user_msg = ""
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            last_user_msg = m.content
            break
    
    # İlgili dokümanları getir
    relevant_docs = get_relevant_documents(last_user_msg, k=3)
    
    # Bağlamla sistem promptunu oluştur
    system_prompt = create_system_prompt(
        state.get("issue_category"),
        state.get("priority"),
        relevant_docs
    )
    
    # Mesajları sistem promptuyla hazırla
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    
    # AI yanıtını al (Gemini ile üretim)
    ai_response = llm_gemini.invoke(messages)
    ai_content = ai_response.content if hasattr(ai_response, "content") else str(ai_response)
    
    # Yanıta göre durumu güncelle
    if "escalate" in ai_content.lower() or "manager" in ai_content.lower():
        state["status"] = "escalated"
    elif "resolved" in ai_content.lower() or "fixed" in ai_content.lower():
        state["status"] = "resolved"
    else:
        state["status"] = "in_progress"
    
    return {
        "messages": [AIMessage(content=ai_content)],
        "user_profile": state.get("user_profile", {}),
        "ticket_id": state.get("ticket_id", ""),
        "issue_category": state.get("issue_category", "general"),
        "priority": state.get("priority", "medium"),
        "status": state["status"]
    }

def build_graph():
    """
    Build the LangGraph for IT support.
    """
    graph_builder = StateGraph(State)
    graph_builder.add_node("it_support_agent", it_support_agent)
    graph_builder.add_edge(START, "it_support_agent")
    graph_builder.add_edge("it_support_agent", END)

    # No checkpoint for now - simplified version
    return graph_builder.compile()

# Global graph instance
graph = build_graph()

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint for real-time responses.
    """
    async def generate_response():
        try:
            # Generate ticket ID (per request)
            ticket_id = str(uuid.uuid4())[:8]

            # Load prior conversation and persist this user message
            history = memory_store.load_messages(request.thread_id)
            history_messages = to_langchain_messages(history)
            try:
                memory_store.append_message(request.thread_id, "user", request.message)
            except Exception as e:
                logger.warning(f"Memory append (user) failed: {e}")

            # Classify the issue using LLM
            classification = classify_issue_with_llm(request.message)
            
            # Log dosyasına da yaz (güvenli şekilde)
            try:
                with open("/opt/rag_bot/classification.log", "a") as f:
                    f.write(f"{datetime.now()}: {classification}\n")
            except Exception as e:
                logger.warning(f"Log dosyası yazma hatası: {e}")
                
            issue_category = classification.get("category", "general")
            priority = classification.get("priority", "medium")

            # Prepare inputs
            inputs = {
                "messages": history_messages + [HumanMessage(content=request.message)],
                "ticket_id": ticket_id,
                "user_profile": request.user_info or {},
                "issue_category": issue_category,
                "priority": priority,
                "status": "open"
            }

            config = {"configurable": {"thread_id": request.thread_id}}

            # Get response from graph
            result = graph.invoke(inputs, config)
            
            # Extract AI message
            ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
            response_text = ai_messages[-1].content if ai_messages else "I could not generate a response."
            
            # Stream the response
            for word in response_text.split():
                yield f"data: {json.dumps({'content': word + ' '})}\n\n"
                await asyncio.sleep(0.05)  # Small delay for streaming effect
            
            # Send final metadata
            yield f"data: {json.dumps({'done': True, 'ticket_id': result.get('ticket_id', ticket_id), 'issue_category': result.get('issue_category', 'general'), 'priority': result.get('priority', 'medium'), 'status': result.get('status', 'open')})}\n\n"

            # Persist assistant reply after streaming completes
            try:
                memory_store.append_message(request.thread_id, "assistant", response_text)
            except Exception as e:
                logger.warning(f"Memory append (assistant) failed: {e}")
            
        except Exception as e:
            logger.error(f"Error in streaming chat: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_response(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

@app.get("/tickets/{thread_id}")
async def get_ticket_info(thread_id: str):
    """
    Get ticket information for a thread.
    """
    try:
        # This would typically query the database for ticket info
        # For now, return basic info
        return {
            "thread_id": thread_id,
            "status": "active",
            "created_at": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail="Ticket not found")

@app.get("/health")
async def health():
    return {"status": "ok"}

def main():
    print("Starting IT Support Agent server on 0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
