<<<<<<< HEAD
# 🛠️ IT Support Agent

A comprehensive IT support system with RAG-powered knowledge base, modern streaming chatbot interface, and intelligent issue categorization.

## 🏗️ Architecture

The system consists of three main components:

- **Backend** (`/backend`): FastAPI-based IT support agent with LangGraph for conversation management
- **Frontend** (`/frontend`): Modern Streamlit chatbot interface with real-time streaming
- **Vector Database** (`/vectordatabase`): RAG-powered knowledge base using Qdrant and sentence transformers

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- OpenAI API key

### Installation

1. **Clone and setup:**
   ```bash
   cd /opt/rag_bot
   cp env.example .env
   ```

2. **Configure environment:**
   Edit `.env` file and add your OpenAI API key:
   ```bash
   OPENAI_API_KEY=your_actual_openai_api_key_here
   ```

3. **Start the system:**
   ```bash
   ./start.sh
   ```

4. **Access the services:**
   - Frontend: http://localhost:8501
   - Backend API: http://localhost:8000
   - Qdrant Vector DB: http://localhost:6333

## 📋 Features

### Backend Agent
- **Intelligent Issue Detection**: Automatically categorizes IT issues (network, hardware, software, etc.)
- **Priority Assessment**: Determines issue priority (high, medium, low)
- **RAG Integration**: Retrieves relevant documentation from knowledge base
- **Conversation Memory**: Maintains context across interactions
- **Ticket Management**: Generates and tracks support tickets
- **Streaming Responses**: Real-time response streaming

### Frontend Chatbot
- **Modern UI**: Clean, responsive design with real-time updates
- **Streaming Chat**: Real-time message streaming for better UX
- **Ticket Tracking**: Visual ticket information and status
- **Quick Actions**: Easy conversation management
- **Mobile Responsive**: Works on all devices

### Vector Database
- **Knowledge Base**: Comprehensive IT support documentation
- **Semantic Search**: Advanced document retrieval using embeddings
- **Multiple Formats**: Supports PDF, Markdown, and text files
- **Scalable**: Built on Qdrant vector database

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `QDRANT_URL` | Qdrant server URL | http://localhost:6333 |
| `QDRANT_PORT` | Qdrant server port | 6333 |
| `QDRANT_COLLECTION` | Vector collection name | it_support_kb |
| `BACKEND_PORT` | Backend API port | 8000 |
| `FRONTEND_PORT` | Frontend port | 8501 |

### API Endpoints

#### Backend API (`http://localhost:8000`)

- `POST /chat` - Send message to IT support agent
- `POST /chat/stream` - Streaming chat endpoint
- `GET /health` - Health check
- `GET /tickets/{thread_id}` - Get ticket information

#### Example API Usage

```bash
# Send a message
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "My computer won't start",
    "thread_id": "user123"
  }'

# Get streaming response
curl -X POST "http://localhost:8000/chat/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "I have a network issue",
    "thread_id": "user123"
  }'
```

## 🧠 Knowledge Base

The system includes a comprehensive IT support knowledge base covering:

- **Network Issues**: Internet connectivity, WiFi problems, DNS issues
- **Hardware Issues**: Computer startup, BSOD, printer problems
- **Software Issues**: Application crashes, performance, updates
- **Email Issues**: Outlook, Gmail troubleshooting
- **Security Issues**: Passwords, antivirus, firewall
- **Database Issues**: SQL Server, MySQL problems
- **Server Issues**: Web servers, Apache, Nginx
- **Mobile Support**: iPhone, Android troubleshooting
- **Remote Support**: VPN, remote desktop

## 🔄 Development

### Running Individual Services

#### Backend Only
```bash
cd backend
pip install -r requirements.txt
python main.py --host 0.0.0.0 --port 8000
```

#### Frontend Only
```bash
cd frontend
pip install -r requirements.txt
streamlit run main.py --server.port 8501
```

#### Vector Database
```bash
cd vectordatabase
pip install -r requirements.txt
python buildRAG.py build --pdf-dir . --pattern "*.md"
```

### Building Knowledge Base

```bash
# Build from markdown files
python buildRAG.py build --pdf-dir /path/to/docs --pattern "*.md"

# Query the knowledge base
python buildRAG.py query "network connectivity issues" --k 5
```

## 📊 Monitoring

### Health Checks

- Backend: `GET http://localhost:8000/health`
- Qdrant: `GET http://localhost:6333/health`
- Frontend: Check Streamlit logs

### Logs

```bash
# View all service logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f backend
docker-compose logs -f frontend
docker-compose logs -f vectordatabase
```

## 🛠️ Troubleshooting

### Common Issues

1. **Backend not starting**: Check OpenAI API key in `.env`
2. **Frontend can't connect**: Ensure backend is running on port 8000
3. **Vector DB empty**: Check Qdrant is running and vectordatabase service completed
4. **Slow responses**: Check system resources and network connectivity

### Debug Mode

```bash
# Run with debug logging
docker-compose up --build --force-recreate
```

## 📈 Scaling

### Production Deployment

1. **Use production Qdrant**: Deploy Qdrant on dedicated server
2. **Load balancing**: Use nginx for frontend/backend
3. **Database**: Use PostgreSQL for ticket storage
4. **Monitoring**: Add Prometheus/Grafana for metrics

### Environment-Specific Configs

```bash
# Production
docker-compose -f docker-compose.prod.yml up

# Development
docker-compose -f docker-compose.dev.yml up
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new features
4. Submit pull request

## 📄 License

MIT License - see LICENSE file for details

## 🆘 Support

For issues and questions:
- Check the troubleshooting section
- Review Docker logs
- Open an issue on GitHub

---

**Built with ❤️ for IT Support Teams**
=======
# IT-Troubleshooting-Chatbot
>>>>>>> f8f0c55fc45e02d5cd90064a92106a526d1bd1c8
