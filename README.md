<p align="center">
  <img src="frontend/logo.png" alt="ProRef AI Logo" width="120" />
</p>

<h1 align="center">PROREF AI — Personalized Prompt Refiner</h1>

<p align="center">
  <strong>Transform vague prompts into precise, domain-specific, personalized prompts — with a learning loop that gets smarter with every rating.</strong>
</p>

<p align="center">
  A RAG-powered web app (v2.0) that retrieves the best prompt-engineering technique and domain-specific best practice for your input, then uses an LLM to rewrite your prompt personalized to your profile. The system <em>learns</em> from your feedback and self-expands its knowledge base over time.
</p>

<p align="center">
  Built with FastAPI · ChromaDB · Sentence Transformers · Groq / Ollama
</p>

---

## What It Does

1. **You create a profile** — your working domain (`dev` / `marketing` / `data_analysis`), interests, and an optional bio
2. **You submit a rough prompt** — e.g. *"write me something about marketing"*
3. **The system retrieves** (via RAG) the most relevant prompt-engineering technique + domain-specific best practice — **reranked by historical success rates from user feedback**
4. **An LLM rewrites** your prompt using the retrieved context, personalized to your profile and your **past similar prompts**
5. **You get back** the refined prompt + a "why it's better" explanation + tags showing which technique and domain tip were used
6. **You rate it** 👍 or 👎 — good ratings update technique weights in ChromaDB and may **promote the pair into the knowledge base** automatically

The transparent explanation and the learning loop are the key differentiators — this is **not** a black box, and it **improves with use**.

---

## What's New in v2.0

| Feature | Description |
|---|---|
| **Feedback Loop** | `POST /feedback` lets users rate refinements `"good"` or `"bad"` |
| **Feedback-Aware Reranking** | Technique success rates are updated in ChromaDB after every rating — better techniques get picked more often |
| **Self-Expanding RAG** | "Good" refinements that meet quality thresholds are automatically promoted into the `domain_knowledge` collection as validated examples |
| **Prompt History** | Every raw prompt is embedded in `user_prompt_history`; similar past prompts are injected into the LLM context for deeper personalization |
| **Arabic → English Pipeline** | `POST /translate-refine` accepts an Arabic prompt, translates it literally, then refines the translation using the full RAG pipeline |
| **Refinement History** | `GET /history/{user_id}` returns a user's past refinements with ratings |
| **Background Processing** | Success rate updates run in a background thread so they never block API responses |
| **Success Rate Persistence** | Learned technique weights survive server restarts (stored in ChromaDB metadata) |
| **Idempotent KB Promotion** | Each refinement can only be promoted to the knowledge base once (UUID-keyed, checked before upsert) |
| **New UI (v2.0)** | Redesigned dark workspace with sidebar navigation, tabbed views (Refine / Arabic / History), and inline feedback buttons |

---

## Architecture

![ProRef AI System Architecture](frontend/architecture_diagram.png)

| Endpoint | Module | Description |
|---|---|---|
| `POST /onboard` | `main.py` → `rag.py` | Stores user profile to disk + ChromaDB |
| `POST /refine` | `main.py` → `rag.py` → `llm.py` → `feedback.py` | Full RAG + LLM refinement pipeline |
| `POST /translate-refine` | `main.py` → `llm.py` → `rag.py` → `feedback.py` | Arabic translation + RAG refinement |
| `POST /feedback` | `main.py` → `feedback.py` (background) | Rates a refinement, updates success rates, may promote to KB |
| `GET /history/{user_id}` | `main.py` → `feedback.py` | Returns a user's past refinements with ratings |

### The Learning Loop

![ProRef AI Learning Loop](frontend/learning_loop_diagram.png)

---

## Tech Stack

| Layer | Technology | Cost |
|---|---|---|
| Backend | FastAPI + Uvicorn | Free |
| Vector DB | ChromaDB (persistent local) | Free |
| Embeddings | `all-MiniLM-L6-v2` via sentence-transformers | Free |
| LLM | Groq (default) or Ollama (local) | Free |
| Frontend | Vanilla HTML/Tailwind CSS/JS — no build step | Free |
| Storage | JSONL files on disk + ChromaDB | Free |

**Total cost to run: $0.** No paid APIs required.

---

## Setup & Run

### 1. Clone & Create Virtual Environment

```bash
git clone https://github.com/xvadel/proref.git
cd proref

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** On first run, `sentence-transformers` will download the `all-MiniLM-L6-v2` model (~80 MB). This is automatic and only happens once.

### 3. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and set your LLM provider:

**Option A — Groq (recommended, default):**
1. Get a free API key at [console.groq.com](https://console.groq.com)
2. Set `GROQ_API_KEY=your_key_here` in `.env`
3. Optionally change `GROQ_MODEL` (default: `llama-3.1-8b-instant`)

**Option B — Ollama (fully local, no API key):**
1. Install Ollama from [ollama.ai](https://ollama.ai)
2. Pull a model: `ollama pull llama3.1`
3. Set `LLM_PROVIDER=ollama` in `.env`

### 4. Run the App

```bash
.venv\Scripts\uvicorn app.main:app --reload   # Windows
# or
uvicorn app.main:app --reload                 # macOS/Linux (if venv is active)
```

Open **http://localhost:8000** in your browser. The knowledge base is automatically ingested on startup — no manual steps needed.

---

## API Reference

### `POST /onboard` — Create/Update User Profile

```bash
curl -X POST http://localhost:8000/onboard \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "alice",
    "domain": "dev",
    "interests": ["python", "rag", "fastapi"],
    "bio": "CS student building AI-powered tools"
  }'
```

**Response:**
```json
{ "user_id": "alice", "status": "onboarded" }
```

---

### `POST /refine` — Refine a Raw Prompt

```bash
curl -X POST http://localhost:8000/refine \
  -H "Content-Type: application/json" \
  -d '{ "user_id": "alice", "raw_prompt": "help me write a REST API" }'
```

**Response:**
```json
{
  "refinement_id": "550e8400-e29b-41d4-a716-446655440000",
  "refined_prompt": "You are a senior Python backend engineer...",
  "explanation": "Added a specific tech stack, role persona, and success criteria...",
  "technique_used": "Role / Persona Setting",
  "domain_tip_used": "Specify Tech Stack & Constraints"
}
```

---

### `POST /translate-refine` — Arabic Prompt → Refined English *(new)*

```bash
curl -X POST http://localhost:8000/translate-refine \
  -H "Content-Type: application/json" \
  -d '{ "user_id": "alice", "arabic_prompt": "ساعدني في كتابة واجهة برمجية" }'
```

**Response:**
```json
{
  "refinement_id": "...",
  "original_arabic": "ساعدني في كتابة واجهة برمجية",
  "translated_english": "Help me write an API",
  "refined_prompt": "You are a senior Python backend engineer...",
  "explanation": "...",
  "technique_used": "Role / Persona Setting",
  "domain_tip_used": "Specify Tech Stack & Constraints"
}
```

---

### `POST /feedback` — Rate a Refinement *(new)*

```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "refinement_id": "550e8400-e29b-41d4-a716-446655440000",
    "user_id": "alice",
    "rating": "good"
  }'
```

**Response:**
```json
{
  "refinement_id": "550e8400-...",
  "rating": "good",
  "promoted_to_kb": true
}
```

> `promoted_to_kb: true` means this refinement was added to the knowledge base as a validated example, making future refinements for similar prompts better.

---

### `GET /history/{user_id}` — Refinement History *(new)*

```bash
curl http://localhost:8000/history/alice?limit=10
```

**Response:**
```json
{
  "user_id": "alice",
  "total": 3,
  "entries": [
    {
      "refinement_id": "...",
      "raw_prompt": "help me write a REST API",
      "refined_prompt": "You are a senior Python backend engineer...",
      "technique_used": "Role / Persona Setting",
      "domain_tip_used": "Specify Tech Stack & Constraints",
      "rating": "good",
      "timestamp": "2026-07-07T01:09:42+00:00",
      "source": "refine"
    }
  ]
}
```

---

## Project Structure

```
proref/
├── app/
│   ├── __init__.py          # Package init
│   ├── main.py              # FastAPI app, all routes, lifespan startup
│   ├── models.py            # Pydantic request/response schemas (all endpoints)
│   ├── rag.py               # ChromaDB setup, ingestion, feedback-aware retrieval
│   ├── llm.py               # Pluggable LLM client (Groq / Ollama) + Arabic pipeline
│   └── feedback.py          # Feedback log, success-rate computation, KB promotion
├── data/
│   ├── prompt_techniques.json         # Prompt-engineering techniques
│   ├── promoted_examples.jsonl        # Audit log of KB-promoted refinements
│   └── domain_knowledge/
│       ├── dev.json                   # Software dev best practices
│       ├── marketing.json             # Marketing best practices
│       └── data_analysis.json         # Data analysis best practices
├── feedback/                          # Created at runtime (gitignored)
│   └── {user_id}.jsonl               # Per-user append-only feedback logs
├── frontend/
│   ├── index.html           # Dark-themed SPA (Onboarding + Workspace)
│   └── logo.png             # App logo (served at /static/logo.png)
├── user_profiles/            # Created at runtime (gitignored)
│   └── {user_id}.json       # User profile JSON files
├── chroma_db/                # Created at runtime (gitignored)
├── scripts/
│   └── ingest.py             # Manual knowledge base re-ingestion script
├── requirements.txt
├── pyproject.toml
├── .env.example
├── .gitignore
└── README.md
```

---

## How RAG Works Here

![RAG Refinement Pipeline](frontend/rag_pipeline_diagram.png)

### On Startup
- `prompt_techniques.json` and `data/domain_knowledge/*.json` are upserted into ChromaDB
- Existing **success rates** from feedback logs are re-applied to technique metadata so learned weights **survive server restarts**

### Key Design Decisions

| Decision | Why |
|---|---|
| Retrieve top-3, then rerank by `success_rate` | Pure similarity misses real-world effectiveness; success rates add a human signal |
| Retrieve similar past prompts **before** embedding | Avoids the current prompt appearing in its own history results (self-match bug) |
| Promote to `domain_knowledge` (not a separate collection) | Promoted examples are retrieved by the same `retrieve_domain_tip()` path — no extra code |
| Background thread for success rate updates | Keeps the `/feedback` response instant; ChromaDB writes are non-blocking |
| UUID-keyed idempotency for promotions | Re-running `/feedback` with `"good"` won't create duplicate KB entries |

---

## Arabic Support

`POST /translate-refine` enables a **two-LLM-call pipeline** for Arabic prompts:

1. **Translate**: LLM converts the Arabic prompt to a literal English translation (faithfully, no interpretation)
2. **Retrieve**: The English translation is used for RAG lookups (techniques + domain tips)
3. **Retrieve history**: Similar past English prompts from the user are fetched for personalization
4. **Refine**: A second LLM call rewrites the translation into a polished, professional English prompt using all context
5. **Log**: Stored as `source: "translate-refine"` for history and feedback tracking

The model used (Groq's `llama-3.1-8b-instant`) handles Arabic natively — no extra library or translation API needed.

---

## License

MIT — use it, learn from it, build on it.
