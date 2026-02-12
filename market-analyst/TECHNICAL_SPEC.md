# Market Analyst - Technical Implementation Specification

## Overview

A real-time multi-agent stock analysis platform that combines technical analysis, fundamental research (RAG-based), and news aggregation to generate comprehensive investment reports.

---

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Frontend (Streamlit)                            │
├──────────────┬──────────────────────────────┬───────────────────────────────┤
│  Left Panel  │        Center Panel          │         Right Panel           │
│  Agent Logs  │        Final Report          │         Artifacts             │
└──────┬───────┴──────────────┬───────────────┴───────────────┬───────────────┘
       │                      │                               │
       └──────────────────────┼───────────────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │   LangGraph State │
                    │    (Streaming)    │
                    └─────────┬─────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────▼───────┐   ┌─────────▼─────────┐   ┌──────▼──────┐
│   Technical   │   │       RAG         │   │    News     │
│     Agent     │   │      Agent        │   │    Agent    │
└───────────────┘   └───────────────────┘   └─────────────┘
```

---

## UI Layout Specification

### Header Section
| Element | Description | Data Source |
|---------|-------------|-------------|
| Ticker Input | Text field with placeholder: "Enter Ticker (e.g., AAPL)" | User Input |
| PDF Upload | File uploader for company reports/filings | User Input |
| Analyze Button | Triggers the multi-agent workflow | User Action |

### Left Sidebar - Agent Thoughts Panel
| Feature | Description | Implementation |
|---------|-------------|----------------|
| Purpose | Real-time scrolling log of agent activities | LangGraph State Streaming |
| Format | Timestamped, color-coded status updates | WebSocket/SSE |

**Status Indicators:**
- `🟢` Supervisor Agent actions (delegation, orchestration)
- `🔵` Technical Agent actions (chart generation, indicator calculation)
- `🟣` RAG Agent actions (vector DB queries, document retrieval)
- `🟠` News Agent actions (API calls, headline extraction)

**Example Log Output:**
```
[10:23:01] 🟢 Supervisor received request for AAPL
[10:23:02] 🟢 Supervisor delegated to Technical Agent
[10:23:03] 🔵 Technical Agent fetching price data...
[10:23:05] 🔵 Technical Agent generating candlestick chart...
[10:23:07] 🟢 Supervisor delegated to RAG Agent
[10:23:08] 🟣 RAG Agent querying Vector DB...
[10:23:10] 🟣 RAG Agent retrieved 5 relevant chunks
[10:23:12] 🟠 News Agent fetching latest headlines...
[10:23:15] 🟢 Supervisor compiling final report...
```

### Center Panel - Final Report
| Tab | Content | Data Source |
|-----|---------|-------------|
| **Summary** | Executive overview with key metrics and recommendation | Final LLM Response |
| **Fundamental Deep Dive** | Detailed analysis with citations from uploaded PDFs | RAG Agent Output |
| **Technicals** | Technical analysis narrative with indicator interpretations | Technical Agent Output |

**Rendering:** Markdown with support for tables, lists, and inline citations

### Right Panel - Artifacts
| Artifact | Description | Source |
|----------|-------------|--------|
| Chart Image | Generated technical chart (candlestick + indicators) | Technical Agent Tool Output |
| News Headlines | Top 3 relevant news items with links | News Agent Tool Output |
| Source Documents | Cited document chunks from RAG | RAG Agent Tool Output |

---

## Agent Specifications

### 1. Supervisor Agent
**Role:** Orchestrator and final report compiler

**Responsibilities:**
- Parse user request and determine required analyses
- Delegate tasks to specialized agents
- Aggregate outputs into coherent final report
- Stream status updates to frontend

**Tools:** None (orchestration only)

### 2. Technical Agent
**Role:** Price data analysis and chart generation

**Responsibilities:**
- Fetch historical price data (via yfinance or similar)
- Calculate technical indicators (RSI, MACD, Moving Averages, etc.)
- Generate visualization (candlestick chart with overlays)
- Produce technical analysis narrative

**Tools:**
- `fetch_price_data(ticker, period)` - Retrieve OHLCV data
- `calculate_indicators(data, indicators[])` - Compute technical indicators
- `generate_chart(data, indicators[])` - Create matplotlib/plotly chart
- `analyze_technicals(data, indicators)` - LLM-based interpretation

### 3. RAG Agent
**Role:** Document-based fundamental analysis

**Responsibilities:**
- Process and embed uploaded PDF documents
- Query vector database for relevant information
- Generate fundamental analysis with citations

**Tools:**
- `ingest_document(pdf_file)` - Parse and embed PDF into vector store
- `query_vector_db(query, top_k)` - Retrieve relevant chunks
- `generate_analysis(context, query)` - LLM analysis with citations

**Vector Store:** ChromaDB / Pinecone / FAISS

### 4. News Agent
**Role:** Real-time news aggregation

**Responsibilities:**
- Fetch latest news for given ticker
- Filter and rank by relevance
- Extract key headlines and summaries

**Tools:**
- `fetch_news(ticker, limit)` - API call to news provider
- `summarize_headlines(articles)` - Extract key information

**Data Sources:** NewsAPI, Alpha Vantage News, or similar

---

## Technology Stack

### Backend
| Component | Technology |
|-----------|------------|
| Agent Framework | LangGraph |
| LLM Provider | OpenAI / Anthropic |
| Vector Database | ChromaDB / FAISS |
| PDF Processing | PyMuPDF / pdfplumber |
| Financial Data | yfinance |
| News API | NewsAPI / Alpha Vantage |
| Charting | Plotly / Matplotlib |

### Frontend
| Component | Technology |
|-----------|------------|
| Framework | Streamlit |
| Real-time Updates | Streamlit callbacks / st.empty() |
| Markdown Rendering | Native Streamlit |
| Image Display | st.image() |

---

## Data Flow

```
1. User Input
   ├── Ticker Symbol ────────────────────────────────────┐
   └── PDF Upload (optional) ────────────────────────────┤
                                                         │
2. Supervisor Agent                                      │
   ├── Receives request ◄────────────────────────────────┘
   ├── Streams: "🟢 Supervisor received request"
   └── Delegates to agents (parallel where possible)
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
3. Parallel Agent Execution
   │
   ├── Technical Agent
   │   ├── Streams: "🔵 Fetching price data..."
   │   ├── Streams: "🔵 Generating chart..."
   │   └── Returns: {chart_image, analysis_text, indicators}
   │
   ├── RAG Agent (if PDF provided)
   │   ├── Streams: "🟣 Querying Vector DB..."
   │   ├── Streams: "🟣 Retrieved N chunks"
   │   └── Returns: {analysis_with_citations, source_chunks}
   │
   └── News Agent
       ├── Streams: "🟠 Fetching headlines..."
       └── Returns: {headlines[], summaries[]}
                    │
                    ▼
4. Supervisor Aggregation
   ├── Streams: "🟢 Compiling final report..."
   ├── Merges all agent outputs
   └── Generates final markdown report
                    │
                    ▼
5. Frontend Render
   ├── Left Panel: Complete agent log
   ├── Center: Tabbed report (Summary | Fundamentals | Technicals)
   └── Right Panel: Chart + Headlines artifacts
```

---

## File Structure

```
market-analyst/
├── backend/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── supervisor.py
│   │   ├── technical_agent.py
│   │   ├── rag_agent.py
│   │   └── news_agent.py
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── price_data.py
│   │   ├── chart_generator.py
│   │   ├── vector_store.py
│   │   ├── pdf_processor.py
│   │   └── news_fetcher.py
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── state.py
│   │   └── workflow.py
│   ├── config.py
│   └── main.py
├── frontend/
│   ├── app.py
│   ├── components/
│   │   ├── __init__.py
│   │   ├── agent_logs.py
│   │   ├── report_tabs.py
│   │   └── artifacts_panel.py
│   └── styles/
│       └── custom.css
├── data/
│   └── vector_store/
├── .env.example
├── requirements.txt
├── README.md
└── TECHNICAL_SPEC.md
```

---

## Environment Variables

```env
# LLM Configuration
OPENAI_API_KEY=your_openai_key
# or
ANTHROPIC_API_KEY=your_anthropic_key

# News API
NEWS_API_KEY=your_news_api_key

# Vector Store (if using cloud)
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENVIRONMENT=your_environment

# Optional
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
```

---

## Implementation Phases

### Phase 1: Core Infrastructure
- [ ] Set up project structure
- [ ] Configure LangGraph workflow
- [ ] Implement state management
- [ ] Create basic Streamlit UI shell

### Phase 2: Technical Analysis Agent
- [ ] Implement price data fetching
- [ ] Build technical indicator calculations
- [ ] Create chart generation tool
- [ ] Integrate with LangGraph

### Phase 3: RAG Agent
- [ ] Set up vector store
- [ ] Implement PDF ingestion pipeline
- [ ] Build retrieval and citation system
- [ ] Integrate with LangGraph

### Phase 4: News Agent
- [ ] Integrate news API
- [ ] Implement headline extraction
- [ ] Add relevance filtering
- [ ] Integrate with LangGraph

### Phase 5: Frontend Polish
- [ ] Implement real-time streaming logs
- [ ] Build tabbed report view
- [ ] Create artifacts panel
- [ ] Add error handling and loading states

### Phase 6: Testing & Optimization
- [ ] End-to-end testing
- [ ] Performance optimization
- [ ] Error handling refinement
- [ ] Documentation

---

## API Contracts

### LangGraph State Schema
```python
from typing import TypedDict, List, Optional, Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    ticker: str
    pdf_content: Optional[str]
    messages: Annotated[list, add_messages]
    agent_logs: List[dict]  # {"timestamp", "agent", "message", "status"}
    technical_output: Optional[dict]  # {"chart_path", "analysis", "indicators"}
    rag_output: Optional[dict]  # {"analysis", "citations", "chunks"}
    news_output: Optional[dict]  # {"headlines", "summaries"}
    final_report: Optional[str]
```

### Streaming Event Format
```json
{
  "timestamp": "2024-01-15T10:23:01Z",
  "agent": "technical",
  "status": "running",
  "message": "Generating candlestick chart...",
  "emoji": "🔵"
}
```

---

## Notes

- All agents should implement proper error handling with graceful degradation
- Streaming updates should be throttled to prevent UI performance issues
- Consider implementing caching for frequently requested tickers
- PDF processing should support async operations for large documents
