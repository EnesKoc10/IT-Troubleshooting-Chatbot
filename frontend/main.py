import streamlit as st
import requests
import json
import uuid
import time
from datetime import datetime
from typing import Dict, Any

# Page configuration
st.set_page_config(
    page_title="BT Destek Asistanı",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .chat-container {
        background: #f8f9fa;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .user-message {
        background: #007bff;
        color: white;
        padding: 1rem;
        border-radius: 15px 15px 5px 15px;
        margin: 0.5rem 0;
        margin-left: 20%;
        text-align: right;
    }
    
    .assistant-message {
        background: #e9ecef;
        color: #333;
        padding: 1rem;
        border-radius: 15px 15px 15px 5px;
        margin: 0.5rem 0;
        margin-right: 20%;
    }
    
    .ticket-info {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 500;
    }
    
    .status-open { background: #fff3cd; color: #856404; }
    .status-in_progress { background: #d1ecf1; color: #0c5460; }
    .status-resolved { background: #d4edda; color: #155724; }
    .status-escalated { background: #f8d7da; color: #721c24; }
    
    .priority-high { background: #f8d7da; color: #721c24; }
    .priority-medium { background: #fff3cd; color: #856404; }
    .priority-low { background: #d4edda; color: #155724; }
    
    .sidebar-section {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Configuration
BACKEND_URL = "http://backend:8000"

class ITSupportChat:
    def __init__(self):
        self.session_id = str(uuid.uuid4())[:8]
        self.thread_id = str(uuid.uuid4())[:8]
        self.user_info = {}
        self.ticket_info = {}
        
    def send_message(self, message: str, use_streaming: bool = True) -> Dict[str, Any]:
        """Send message to backend and get response."""
        try:
            if use_streaming:
                return self._stream_response(message)
            else:
                return self._get_response(message)
        except Exception as e:
            st.error(f"Error communicating with backend: {e}")
            return {"error": str(e)}
    
    
    def _stream_response(self, message: str) -> Dict[str, Any]:
        """Get streaming response from backend."""
        try:
            response = requests.post(
                f"{BACKEND_URL}/chat/stream",
                json={
                    "message": message,
                    "thread_id": self.thread_id,
                    "user_info": self.user_info
                },
                stream=True,
                timeout=30
            )
            
            if response.status_code == 200:
                full_response = ""
                final_meta: Dict[str, Any] = {}
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            data = json.loads(line[6:])
                            if 'content' in data:
                                full_response += data['content']
                            elif data.get('done'):
                                final_meta = data
                # Build result with meta if available
                if final_meta:
                    return {
                        "response": full_response,
                        "ticket_id": final_meta.get('ticket_id', ''),
                        "thread_id": self.thread_id,
                        "issue_category": final_meta.get('issue_category'),
                        "priority": final_meta.get('priority'),
                        "status": final_meta.get('status'),
                    }
                return {"response": full_response}
            else:
                return {"error": f"Backend error: {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}

def initialize_session():
    """Initialize session state."""
    if 'chat' not in st.session_state:
        st.session_state.chat = ITSupportChat()
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'ticket_info' not in st.session_state:
        st.session_state.ticket_info = {}
    if 'ticket_info_locked' not in st.session_state:
        st.session_state.ticket_info_locked = False  # Konuşma bilgileri kilitli mi?

def display_header():
    """Display the main header."""
    st.markdown("""
    <div class="main-header">
        <h1>🛠️ BT Destek Asistanı</h1>
        <p>Teknik sorunlarınız için anında yardım alın</p>
    </div>
    """, unsafe_allow_html=True)

def display_sidebar():
    """Display sidebar with ticket information and controls."""
    with st.sidebar:
        st.markdown("### 📋 Konuşma Bilgileri")
        
        if st.session_state.ticket_info:
            # Expect only English keys in ticket_info
            ticket_info = {
                "ticket_id": st.session_state.ticket_info.get("ticket_id", "N/A"),
                "issue_category": st.session_state.ticket_info.get("issue_category", "N/A"),
                "priority": st.session_state.ticket_info.get("priority", "medium"),
                "status": st.session_state.ticket_info.get("status", "open"),
            }
            priority_label = ticket_info.get('priority', 'medium')
            status_label = ticket_info.get('status', 'open')
            st.markdown(f"""
            <div class="ticket-info">
                <strong>Konuşma ID:</strong> {ticket_info.get('ticket_id', 'N/A')}<br>
                <strong>Kategori:</strong> {ticket_info.get('issue_category', 'N/A')}<br>
                <strong>Öncelik:</strong> <span class="status-badge priority-{priority_label}">{str(priority_label).title()}</span><br>
                <strong>Durum:</strong> <span class="status-badge status-{status_label}">{str(status_label).replace('_', ' ').title()}</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### 🎯 Hızlı İşlemler")
        if st.button("🔄 Yeni Konuşma", use_container_width=True):
            st.session_state.chat = ITSupportChat()
            st.session_state.messages = []
            st.session_state.ticket_info = {}
            st.session_state.ticket_info_locked = False  # Kilitli bilgileri sıfırla
            st.rerun()
        
        st.markdown("### 💡 İpuçları")
        st.markdown("""
        - Sorununuzu mümkün olduğunca net anlatın
        - Varsa hata mesajlarını ekleyin
        - Sistem/yazılım sürümünüzü belirtin
        - Sorun oluşurken ne yaptığınızı kısaca açıklayın
        """)

def display_messages():
    """Display chat messages."""
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="user-message">
                <strong>Sen:</strong> {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="assistant-message">
                <strong>BT Destek:</strong> {message["content"]}
            </div>
            """, unsafe_allow_html=True)


def display_chat_input():
    """Display chat input and handle message sending."""
    with st.container():
        st.markdown("### 💬 BT Destek ile Sohbet")
        
        # Chat input
        user_input = st.text_area(
            "Teknik sorununuzu anlatın:",
            placeholder="Örn: Bilgisayar açılmıyor, mavi ekran hatası alıyorum...",
            height=100,
            key=f"chat_input_{len(st.session_state.messages)}"
        )
        
        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            send_button = st.button("🚀 Gönder", use_container_width=True, type="primary")

        if send_button and user_input:
            # Add user message to chat
            st.session_state.messages.append({
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now()
            })
            
            # Get response from backend
            with st.spinner("BT Destek düşünüyor..."):
                response = st.session_state.chat.send_message(user_input, use_streaming=True)
            
            if "error" in response:
                st.error(f"Hata: {response['error']}")
            else:
                # Add assistant response to chat
                assistant_response = response.get("response", "Yanıt alınamadı")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": assistant_response,
                    "timestamp": datetime.now()
                })
                
                # Show success message
                st.success("✅ BT Destek yanıt verdi!")
                
                # Update ticket information - sadece kilitli değilse güncelle
                if "ticket_id" in response and not st.session_state.ticket_info_locked:
                    st.session_state.ticket_info = {
                        "ticket_id": response.get("ticket_id", ""),
                        "issue_category": response.get("issue_category"),
                        "priority": response.get("priority"),
                        "status": response.get("status"),
                    }
                    # Bilgileri kilitle - bir daha değişmesin
                    st.session_state.ticket_info_locked = True
                
                # Force rerun to show the new message
                st.experimental_rerun()

def main():
    """Main application function."""
    initialize_session()
    
    # Display header
    display_header()
    
    # Create main layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Chat container
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Display messages
        if st.session_state.messages:
            display_messages()
        else:
            st.markdown("""
            <div class="assistant-message">
                <strong>BT Destek:</strong> Merhaba! Teknik sorunlarınızda size yardımcı olmak için buradayım.
                Lütfen probleminizi kısaca açıklayın, en iyi şekilde destek olayım.
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Chat input
        display_chat_input()
        
        # Knowledge base section removed per request
    
    with col2:
        # Sidebar
        display_sidebar()

if __name__ == "__main__":
    main()
