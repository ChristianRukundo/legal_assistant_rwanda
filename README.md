# Inyandiko Legal AI Assistant - Backend

A sophisticated voice-enabled legal information system for Rwandan law in Kinyarwanda, built with FastAPI and advanced AI technologies.

## 🚀 Features

- **Advanced RAG Pipeline**: Multi-modal retrieval with hybrid search strategies
- **Voice Processing**: Kinyarwanda ASR and TTS with emotion detection
- **Document Intelligence**: Comprehensive PDF processing with OCR support
- **Enterprise Caching**: Multi-layer caching with Redis and memory optimization
- **Query Analysis**: Intent detection, entity extraction, and complexity scoring
- **Real-time Processing**: Async architecture for high-performance operations

## 📋 Prerequisites

- Python 3.11+
- CUDA-compatible GPU (recommended for optimal performance)
- 16GB+ RAM
- Redis server (for caching)
- Legal PDF documents in Kinyarwanda

## 🛠️ Installation

### 1. Environment Setup

\`\`\`bash
# Clone repository and navigate to backend
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
\`\`\`

### 2. Configuration

\`\`\`bash
# Copy environment template
cp .env.example .env

# Edit .env with your configurations
nano .env
\`\`\`

### 3. Document Setup

\`\`\`bash
# Create legal documents directory
mkdir legal_docs

# Add your PDF legal documents to this directory
# Example: cp /path/to/your/legal/docs/*.pdf legal_docs/
\`\`\`

### 4. Database Initialization

\`\`\`bash
# Process documents and create vector database
python process_docs.py
\`\`\`

## 🚀 Running the Application

### Development Mode
\`\`\`bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
\`\`\`

### Production Mode
\`\`\`bash
python main.py
\`\`\`

### Docker Deployment
\`\`\`bash
# Build image
docker build -t inyandiko-backend .

# Run container
docker run -p 8000:8000 -v $(pwd)/legal_docs:/app/legal_docs inyandiko-backend
\`\`\`

## 📡 API Endpoints

### Health Check
- **GET** `/health`
- Returns system health status and component information

### Text Query
- **POST** `/query`
- **Body**: 
\`\`\`json
{
  "query": "Ni iki gihano cy'ubujurire?",
  "language": "rw",
  "session_id": "optional-session-id"
}
\`\`\`
- **Response**:
\`\`\`json
{
  "answer": "Ubujurire ni icyaha...",
  "citations": [
    {
      "source_pdf": "penal_code.pdf",
      "page_number": 45,
      "excerpt": "Article 123..."
    }
  ],
  "confidence_score": 0.92,
  "processing_time": 1.23
}
\`\`\`

### Voice Query
- **POST** `/voice_query`
- **Body**: `multipart/form-data` with `audio_file`
- **Response**: Audio stream (MP3)

### PDF Page Viewer
- **GET** `/get_pdf_page/{pdf_name}/{page_num}`
- **Response**: PNG image of the PDF page

## 🏗️ Architecture

### Core Components

1. **Advanced RAG Pipeline** (`advanced_rag_pipeline.py`)
   - Multi-modal document retrieval
   - Hybrid search combining vector similarity, BM25, and TF-IDF
   - Cross-encoder reranking for precision
   - Document clustering and knowledge graphs

2. **Sophisticated Voice Processor** (`sophisticated_voice_processor.py`)
   - Multi-language ASR with Whisper models
   - Kinyarwanda TTS with emotion detection
   - Audio quality enhancement and noise reduction
   - Real-time streaming capabilities

3. **Document Intelligence Processor** (`advanced_document_processor.py`)
   - Multi-format document support (PDF, DOCX, HTML, etc.)
   - OCR for scanned documents
   - Advanced chunking strategies
   - Metadata extraction and classification

4. **Enterprise Caching System** (`enterprise_caching_system.py`)
   - Multi-layer caching (memory, Redis, disk)
   - Intelligent cache warming and invalidation
   - Compression and serialization optimization
   - Performance monitoring and analytics

5. **Intelligent Query Processor** (`intelligent_query_processor.py`)
   - Intent classification and entity extraction
   - Query expansion and reformulation
   - Legal domain detection
   - Context-aware processing

## 🤖 AI Models Used

- **Embeddings**: `intfloat/multilingual-e5-large`
- **LLM**: `mistralai/Mistral-7B-Instruct-v0.2`
- **ASR**: `openai/whisper-base`
- **TTS**: `facebook/mms-tts-kin`
- **Cross-Encoder**: `cross-encoder/ms-marco-MiniLM-L-6-v2`

## ⚙️ Configuration Options

### Environment Variables

\`\`\`bash
# Model Configuration
EMBEDDING_MODEL=intfloat/multilingual-e5-large
LLM_MODEL=mistralai/Mistral-7B-Instruct-v0.2
ASR_MODEL=openai/whisper-base
TTS_MODEL=facebook/mms-tts-kin

# Paths
LEGAL_DOCS_DIR=legal_docs
VECTOR_DB_DIR=vector_db

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000

# Caching
REDIS_URL=redis://localhost:6379
CACHE_TTL=3600
ENABLE_CACHE_WARMING=true

# Performance
MAX_WORKERS=4
BATCH_SIZE=32
GPU_MEMORY_FRACTION=0.8

# Security
ENABLE_RATE_LIMITING=true
MAX_REQUESTS_PER_MINUTE=30
ENABLE_QUERY_VALIDATION=true

# Logging
LOG_LEVEL=INFO
ENABLE_STRUCTURED_LOGGING=true
LOG_FILE=logs/inyandiko.log
\`\`\`

## 🔧 Performance Optimization

### GPU Acceleration
- Automatic GPU detection and utilization
- Model quantization for memory efficiency
- Batch processing for improved throughput

### Caching Strategy
- Query result caching with intelligent invalidation
- Model output caching for repeated requests
- Document embedding caching for faster retrieval

### Memory Management
- Lazy model loading to reduce startup time
- Garbage collection optimization
- Memory pool management for large documents

## 🔒 Security Features

- Path traversal protection for PDF access
- Query validation and sanitization
- Rate limiting with Redis backend
- CORS configuration for web security
- Input validation with Pydantic models

## 📊 Monitoring & Analytics

- Prometheus metrics integration
- Structured logging with correlation IDs
- Performance tracking and alerting
- Usage analytics and reporting
- Health check endpoints

## 🧪 Testing

\`\`\`bash
# Run unit tests
pytest tests/

# Run integration tests
pytest tests/integration/

# Run performance tests
pytest tests/performance/
\`\`\`

## 🐛 Troubleshooting

### Common Issues

1. **Models not loading**
   - Check GPU memory availability
   - Verify model paths in environment variables
   - Ensure sufficient disk space for model downloads

2. **Vector database errors**
   - Run `python process_docs.py` to rebuild database
   - Check legal_docs directory contains PDF files
   - Verify write permissions for vector_db directory

3. **Voice processing issues**
   - Install system audio libraries: `apt-get install ffmpeg libsndfile1`
   - Check audio file format compatibility
   - Verify microphone permissions for recording

4. **Performance issues**
   - Enable GPU acceleration if available
   - Increase cache size and TTL
   - Optimize batch sizes for your hardware

### Logs and Debugging

\`\`\`bash
# View application logs
tail -f logs/inyandiko.log

# Check component health
curl http://localhost:8000/health

# Monitor metrics
curl http://localhost:8000/metrics
\`\`\`

## 📈 Scaling Considerations

### Horizontal Scaling
- Load balancer configuration
- Shared Redis cache across instances
- Database connection pooling

### Vertical Scaling
- GPU memory optimization
- CPU core utilization
- Memory allocation tuning

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For technical support and questions:
- Email: support@inyandiko.rw
- Documentation: https://docs.inyandiko.rw
- Issues: https://github.com/inyandiko/backend/issues

## 🔄 Version History

- **v2.0.0** - Advanced RAG pipeline with enterprise features
- **v1.5.0** - Voice processing improvements and caching
- **v1.0.0** - Initial release with basic RAG functionality
