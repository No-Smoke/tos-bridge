# RAGFlow — Deep Technical Analysis
# Date: 2026-05-24 | Source: GitHub infiniflow/ragflow main branch (v0.25.5)
# Status: 🟡 Later — specific use case, non-trivial integration

---

## What RAGFlow Actually Is

RAGFlow is a self-hosted RAG engine whose primary differentiator is **DeepDoc** — an in-house document understanding pipeline that extracts structured content from complex documents (scanned PDFs, tables, figures, engineering drawings) before chunking. This is genuinely better than naive text splitting + embeddings.

It is NOT primarily a vector store or an LLM — it's a document ingestion and retrieval pipeline that wraps LLM + vector store internally.

---

## Docker Stack — What You're Actually Deploying

Two files: `docker-compose.yml` (main service) + `docker-compose-base.yml` (dependencies)

### Core services (always running):
| Service | Image | Purpose | RAM budget |
|---------|-------|---------|-----------|
| ragflow-cpu | infiniflow/ragflow | Main app (API + web UI + workers) | MEM_LIMIT (default 8GB!) |
| mysql | mysql:8.0.39 | Metadata store (datasets, documents, users) | ~1GB |
| minio | pgsty/minio | Object storage (original files + parsed chunks) | ~512MB |
| redis/valkey | valkey/valkey:8 | Task queue + caching | 128MB (capped) |

### Vector store (choose ONE profile):
| Option | Notes |
|--------|-------|
| `elasticsearch` (default) | Ships Elasticsearch 8.11.3. ~4-6GB RAM. The default. |
| `infinity` | InfiniFlow's own vector DB. Lighter. |
| `opensearch` | Alternative to ES. Similar resource profile. |
| `oceanbase` / `seekdb` | Specialty options, not relevant here. |

**No Qdrant option.** RAGFlow does not support external Qdrant as a vector store backend. The vector store is locked to the options in `DOC_ENGINE` env var.

### GPU acceleration (optional profile):
| Option | Notes |
|--------|-------|
| `cpu` (default) | DeepDoc inference runs CPU-only (slow but works) |
| `gpu` | NVIDIA CUDA only — NOT AMD ROCm. No RDNA3 support. |

### Optional services:
- `tei-cpu` / `tei-gpu` — Text Embeddings Inference (HuggingFace TEI) — for local embedding without calling Ollama
- `kibana` — Elasticsearch UI (rarely needed)

### Total resource footprint:
- **Minimum RAM: ~10GB** (ragflow 8GB + mysql 1GB + minio 0.5GB + redis 0.128GB)
- **Realistic comfortable: ~16GB** with Elasticsearch and active document processing
- **Disk: 50GB minimum** (official docs) — Elasticsearch index + MinIO object store grow with ingested docs
- **CPU: 4+ cores** — document parsing is CPU-intensive when GPU not available

---

## Vector Store Architecture — The Critical Constraint

RAGFlow uses its **own internal vector store** (Elasticsearch by default). It writes parsed document chunks + embeddings into ES, not into an external Qdrant.

**There is no config option to point RAGFlow at your existing Qdrant CT201.**

This is the fundamental architectural tension for your fleet:
- Your knowledge system: Qdrant CT201 (111+ collections) + Neo4j CT202 + neo4j_interaction_id invariant
- RAGFlow's system: Elasticsearch (or Infinity) inside RAGFlow's Docker network + MinIO + MySQL

RAGFlow and your tos-bridge/Qdrant system are parallel knowledge stores, not one feeding the other — unless you build a bridge.

---

## LLM + Embedding Integration

### Embedding models (confirmed in code):
RAGFlow has a full provider system (`rag/llm/embedding_model.py`). Supported:
- **Ollama** — `OllamaEmbed` class (`_FACTORY_NAME = "Ollama"`). Config: `base_url` pointing at your Ollama instance. Calls `ollama.Client(host=base_url).embeddings(prompt=text, model=model_name)`.
- OpenAI, Azure OpenAI, LocalAI (OpenAI-compat), HuggingFace TEI, Xinference, ZhipuAI, QWen, BaiChuan, and many more.

**Ollama embedding is natively supported.** You can configure RAGFlow to call your CT200 Ollama at `http://192.168.1.76:11434` for embeddings (mxbai-embed-large, nomic-embed-text, etc.).

### Chat/LLM models:
RAGFlow uses LiteLLM under the hood for chat. Confirmed Ollama support in `rag/llm/chat_model.py`. Same config pattern — point at CT200.

### Configuration path:
In `service_conf.yaml.template`:
```yaml
user_default_llm:
  default_models:
    embedding_model:
      api_key: 'x'
      base_url: 'http://192.168.1.76:11434'  # your Ollama CT200
    chat_model:
      name: 'gemma4:latest'
      factory: 'Ollama'
      api_key: 'x'
      base_url: 'http://192.168.1.76:11434'
```

Or configure via the RAGFlow web UI (Settings → Model Providers → Ollama). The UI config is stored in MySQL and overrides the yaml for per-user settings.

---

## Document Parsing Pipeline (DeepDoc)

Source: `deepdoc/` directory

### Supported parsers:
| Parser | File types | Notes |
|--------|-----------|-------|
| `pdf_parser.py` | PDF (native + scanned) | Layout detection + OCR for scanned |
| `docx_parser.py` | .docx | Tables, headers, formatting |
| `excel_parser.py` | .xlsx, .xls | Spreadsheets → structured rows |
| `ppt_parser.py` | .pptx | Slides |
| `html_parser.py` | HTML | Web pages |
| `markdown_parser.py` | .md | Code blocks, headers |
| `txt_parser.py` | .txt | Plain text |
| `epub_parser.py` | .epub | E-books |
| `json_parser.py` | .json | Structured data |
| `figure_parser.py` | Images | Layout + figure extraction |
| `docling_parser.py` | Multi-format | IBM Docling integration |
| `mineru_parser.py` | Multi-format | MinerU integration (academic papers) |
| `paddleocr_parser.py` | Scanned docs | PaddleOCR for Chinese + multilingual |

### OCR and vision:
- `deepdoc/vision/ocr.py` — OCR inference
- `deepdoc/vision/layout_recognizer.py` — Page layout detection (text blocks, tables, figures)
- **CPU mode:** All runs on CPU. Slow (~30-120s per PDF page for complex docs), but works.
- **GPU mode:** NVIDIA CUDA only — no AMD ROCm GPU acceleration for DeepDoc.

### What this means for your use case:
If you're ingesting battery spec sheets (PDFs with tables of specs, scanned engineering drawings, IEC/IEEE standard documents), RAGFlow's parsing extracts:
- Structured table cells (not just "a table was here")
- Page layout relationships (figure caption → figure)
- Section hierarchy
- Embedded images with OCR text

Your current Qdrant + mxbai-embed-large approach gets raw text chunks. RAGFlow gets structured semantic units. That's the genuine value.

---

## HTTP API

Base URL: `http://<host>:9380/api/v1`
Auth: `Authorization: Bearer <api_key>` header on all requests.

### Key endpoints:

**Datasets (Knowledge Bases):**
```
POST   /api/v1/datasets                          Create dataset
GET    /api/v1/datasets?name=...&page=1          List datasets
PUT    /api/v1/datasets/{dataset_id}             Update dataset config
DELETE /api/v1/datasets                          Delete datasets (body: {ids: [...]})
```

**Documents:**
```
POST   /api/v1/datasets/{dataset_id}/documents   Upload files (multipart/form-data)
GET    /api/v1/datasets/{dataset_id}/documents   List documents
DELETE /api/v1/datasets/{dataset_id}/documents   Delete documents
POST   /api/v1/datasets/{dataset_id}/chunks      Start parsing (body: {document_ids: [...]})
DELETE /api/v1/datasets/{dataset_id}/chunks      Cancel parsing
```

**Retrieval:**
```
POST   /api/v1/datasets/{dataset_id}/search      Search within one dataset
POST   /api/v1/datasets/search                   Search across multiple datasets
POST   /api/v1/retrieval                         Legacy retrieval endpoint (SDK compat)
```

Retrieval request body:
```json
{
  "dataset_ids": ["<id1>", "<id2>"],
  "document_ids": ["<optional filter>"],
  "question": "What is the capacity of this battery system?",
  "page": 1,
  "page_size": 10,
  "similarity_threshold": 0.2,
  "vector_similarity_weight": 0.3,
  "top_k": 1024,
  "keyword": false,
  "rerank_id": null
}
```

Retrieval response:
```json
{
  "code": 0,
  "data": {
    "chunks": [
      {
        "id": "<chunk_id>",
        "content": "...",
        "document_id": "<doc_id>",
        "document_name": "battery_spec_v2.pdf",
        "dataset_id": "<dataset_id>",
        "similarity": 0.87,
        "vector_similarity": 0.91,
        "term_similarity": 0.72,
        "positions": [[page, x0, y0, x1, y1], ...],
        "important_keywords": [...],
        "doc_type": "pdf"
      }
    ],
    "total": 42
  }
}
```

Note: `positions` gives you the bounding box in the source document — useful for citation highlighting.

**Document metadata (custom fields):**
```
PATCH  /api/v1/datasets/{dataset_id}/documents/metadatas
```
Body: `{selector: {document_ids: [...]}, updates: [{key: "author", value: "..."}], deletes: []}`
You can add arbitrary metadata key-value pairs to documents, then filter retrieval by them.

---

## Python SDK

Install: `pip install ragflow-sdk`

```python
from ragflow_sdk import RAGFlow

rag = RAGFlow(api_key="ragflow-xxx", base_url="http://192.168.1.X:9380")

# Create a dataset (knowledge base)
dataset = rag.create_dataset(
    name="battery-specs",
    chunk_method="paper",    # or "naive", "table", "picture", etc.
    description="Battery spec sheets and standards"
)

# Upload a document
with open("battery_spec.pdf", "rb") as f:
    docs = dataset.upload_documents([{"display_name": "battery_spec.pdf", "blob": f.read()}])

# Parse (trigger DeepDoc pipeline)
doc_ids = [d.id for d in docs]
results = dataset.parse_documents(doc_ids)  # blocking, polls until done

# Retrieval (via dataset object)
chunks = dataset.list_documents()  # or use HTTP retrieval endpoint directly
```

Retrieval is via the HTTP API directly (the SDK's dataset object doesn't have a `.search()` method — use `requests` or the MCP server).

---

## MCP Server (Built-In)

RAGFlow ships an **official MCP server** at `mcp/server/server.py`. This is a first-class feature, not a community hack.

### Startup (via docker-compose):
```yaml
command:
  - --enable-mcpserver
  - --mcp-host=0.0.0.0
  - --mcp-port=9382
  - --mcp-base-url=http://127.0.0.1:9380
  - --mcp-script-path=/ragflow/mcp/server/server.py
  - --mcp-mode=self-host
  - --mcp-host-api-key=ragflow-xxx
```

Ports exposed: `:9382` (MCP), `:9380` (HTTP API), `:80` (web UI via nginx)

### Transport:
Supports both SSE and Streamable HTTP (MCP spec). Configurable via flags.

### Tools exposed:
Currently **one tool**: `ragflow_retrieval`

```json
{
  "name": "ragflow_retrieval",
  "description": "Retrieve relevant chunks from the RAGFlow retrieve interface...",
  "inputSchema": {
    "dataset_ids": ["optional array of dataset IDs — omit to search all"],
    "document_ids": ["optional filter by document"],
    "question": "required",
    "page": 1,
    "page_size": 10,
    "similarity_threshold": 0.2,
    "vector_similarity_weight": 0.3,
    "keyword": false,
    "top_k": 1024,
    "rerank_id": null,
    "force_refresh": false
  }
}
```

Returns chunks with similarity scores, positions, document metadata. An LLM calling this tool gets structured retrieval with source attribution.

### Architecture: self-host mode vs host mode
- `self-host`: MCP server authenticates against its own RAGFlow instance (one API key per deployment)
- `host`: Multi-tenant — each MCP client passes its own RAGFlow API key per request

---

## GPU / AMD ROCm Status

**DeepDoc GPU acceleration: NVIDIA CUDA only.**

In `docker-compose-base.yml`, the gpu profile uses:
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

No ROCm/AMD device mapping. The DeepDoc vision models (layout recognizer, OCR) use PyTorch/ONNX with CUDA backend only.

**What this means:**
- CPU mode (default) works fine on your Ivan setup — just slower
- For a battery spec sheet (15-20 page PDF), CPU parsing takes ~2-5 minutes. Acceptable for async ingestion.
- Do NOT attempt to give RAGFlow GPU access on Ivan — it won't use AMD GPU correctly, and would conflict with CT200/CT207 anyway

---

## Resource Footprint on Ivan

| Resource | Requirement | Ivan capacity | Verdict |
|---------|-------------|---------------|---------|
| RAM | ~10-16GB minimum | 64GB total, ~14-24GB free | ⚠️ Tight but possible |
| CPU | 4+ cores, sustained during parsing | Xeon W-2133 6C/12T | ✅ OK if not parsing during IWO/ComfyUI |
| Disk (rpool NVMe) | 50GB minimum for ES + MinIO | 2×1TB mirror, ~60-70% used | ⚠️ Needs audit before install |
| Disk (bulk-pool SATA) | Could mount MinIO data here | 7.25TB available | ✅ Move MinIO to bulk-pool |
| GPU | Not used (CPU mode) | N/A | ✅ No contention |
| Network | Internal LAN only | N/A | ✅ No tunnel needed |

**RAM is the real constraint.** Ivan has 64GB but the fleet already uses ~40-50GB under load. Adding 10-16GB for RAGFlow is feasible only when IWO/DBOS isn't running a heavy job. This needs measurement before committing.

New container: CT222 or similar. Docker Compose stack inside the CT (like CT204 Nextcloud or CT209 Firecrawl).

---

## Integration Options with tos-bridge

Three patterns, ordered by integration depth:

### Pattern 1: Parallel Stores (Simplest, Most Isolated)
RAGFlow manages its own ES store for document parsing + retrieval. Your tos-bridge/Qdrant stays unchanged. tos-bridge handles structured knowledge (interactions, decisions, project memory). RAGFlow handles document-level retrieval (spec sheets, standards, manuals).

Two separate tools in Claude Desktop / Hermes:
- `ragflow_retrieval` → engineering docs
- `search_with_graph` (tos-bridge) → interaction memory + project knowledge

**Integration effort:** Low. Just register RAGFlow's MCP server as a second MCP server in Claude Desktop.
**neo4j_interaction_id invariant:** Preserved — RAGFlow doesn't touch Qdrant.
**Risk:** Low. Fully isolated.
**Downside:** No cross-referencing. A battery spec chunk in RAGFlow doesn't know it was retrieved during interaction #1234 in Neo4j.

---

### Pattern 2: RAGFlow → tos-bridge Bridge (Recommended starting point)
After RAGFlow parses and retrieves a chunk, a custom function writes a summary of that retrieval event into tos-bridge (Qdrant + Neo4j). The RAGFlow chunk content is NOT written to Qdrant verbatim — only the interaction record.

```python
# After RAGFlow retrieval returns chunks:
def on_ragflow_retrieval(question, chunks, session_id):
    summary = f"Retrieved {len(chunks)} chunks from RAGFlow for: {question[:200]}"
    tos_bridge.store_doc_with_graph(
        text=summary,              # <400 tokens — summary only, not full chunk
        collection="system_configuration",
        metadata={
            "source": "ragflow",
            "dataset": chunks[0].dataset_id,
            "question": question,
            "top_chunk_doc": chunks[0].document_name,
            "similarity": chunks[0].similarity,
        }
    )
    # neo4j_interaction_id is generated by tos-bridge internally
```

**Integration effort:** Medium. New code in tos-bridge or a wrapper layer.
**neo4j_interaction_id invariant:** Preserved — tos-bridge generates the ID as normal.
**Risk:** Low. tos-bridge write is additive; RAGFlow is independent.

---

### Pattern 3: tos-bridge as RAGFlow MCP Client (Advanced)
Add a `ragflow_ingest` tool and `ragflow_retrieve` tool to tos-bridge's MCP server. tos-bridge becomes the single MCP interface — internally it routes document queries to RAGFlow and stores the retrieval interaction in Neo4j.

New MCP tools in tos-bridge:
```python
@tool("ragflow_ingest")
async def ragflow_ingest(file_path: str, dataset_name: str, metadata: dict) -> dict:
    """Upload and parse a document in RAGFlow, then record the ingestion in Neo4j."""
    # 1. Upload to RAGFlow via HTTP API
    # 2. Trigger parsing (async)
    # 3. Store ingestion event in tos-bridge Qdrant/Neo4j
    # Returns: {dataset_id, document_id, neo4j_interaction_id}

@tool("ragflow_retrieve")
async def ragflow_retrieve(question: str, dataset_ids: list = None) -> dict:
    """Retrieve from RAGFlow, record retrieval event in Neo4j, return chunks."""
    # 1. POST /api/v1/datasets/search to RAGFlow
    # 2. Record retrieval interaction in tos-bridge Neo4j (preserves neo4j_interaction_id)
    # 3. Return chunks with neo4j_interaction_id appended to each chunk's metadata
    # Returns: {chunks: [...], neo4j_interaction_id: "..."}
```

**Integration effort:** High. Requires modifying tos-bridge MCP server code.
**neo4j_interaction_id invariant:** Fully preserved — every retrieval gets a Neo4j node.
**Risk:** Medium. Modifying tos-bridge is the highest-risk step.
**Benefit:** Single MCP interface for all knowledge ops. Cross-referencing between RAGFlow chunks and interaction history.

---

## Concrete Recommendation

**Pattern 1 is the right starting point.** Here's why:

1. RAGFlow's built-in MCP server (`ragflow_retrieval` tool) is already production-ready — no custom code needed to get retrieval working in Claude Desktop or Hermes.
2. Your tos-bridge invariant is untouched.
3. You learn RAGFlow's behaviour with real documents (battery specs, IEC standards) before committing to integration complexity.
4. Pattern 2 or 3 can be added incrementally after Pattern 1 proves value.

**Install trigger:** When you have a specific batch of engineering documents (battery spec sheets, substation manuals, IEC/IEEE standards) that you want to query via Claude Desktop. Not before — there's no point running the 10-16GB stack idle.

---

## What Still Needs Investigation Before Install

1. **Ivan RAM headroom** — measure peak fleet RAM under IWO load. Need confirmed 12-16GB free before adding RAGFlow. Don't guess.
2. **rpool disk space** — audit current usage. RAGFlow needs 50GB+; MinIO data should go on bulk-pool (mount as volume in CT config).
3. **Elasticsearch vs Infinity** — Infinity is InfiniFlow's own lighter vector DB and uses far less RAM than ES. Worth benchmarking before committing to the 8GB ES default.
4. **Parsing speed** — test a sample PDF on CPU. If a 20-page battery spec takes 10 minutes, async ingestion is fine. If it's 2 hours, that changes the workflow.
5. **Ollama embedding integration** — confirm mxbai-embed-large works with RAGFlow's OllamaEmbed. The 512-token hard limit may cause issues with RAGFlow's chunk sizes (which can be larger than 512 tokens for "paper" chunk method).

---

## Summary Verdict

**🟡 Later — specific trigger, non-trivial resource requirements, clear value when that trigger fires.**

- **Value:** Genuine. DeepDoc document parsing is measurably better than naive chunking for complex PDFs.
- **Use case:** Battery spec sheets, IEC/IEEE standards, scanned engineering drawings → queryable via Claude.
- **Blocker today:** RAM (10-16GB minimum), disk (50GB+), and no current batch of documents to justify it.
- **Integration path:** Pattern 1 (parallel MCP server) → no custom code, no tos-bridge risk. Register ragflow:9382 as second MCP server in Claude Desktop config.
- **Install when:** You have a set of engineering documents you want to query, AND you've confirmed RAM headroom, AND you've measured CPU parsing speed on a sample doc.
