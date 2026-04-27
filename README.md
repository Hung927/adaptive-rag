# RAG System

基於 LangGraph 的 RAG 文件問答系統，支援 PDF / DOCX / TXT，使用 Qdrant 向量資料庫與 Azure OpenAI。

## 功能

- 文件前處理：PDF（pymupdf4llm）、DOCX（python-docx）、TXT / Markdown
- 向量儲存：Qdrant（Docker）
- LangGraph pipeline：檢索 → 品質審查 → 查詢改寫 → 生成 → 品質審查 → 評估
- 可觀測性：Langfuse v4 追蹤每個 node 的 input / output
- 介面：Streamlit UI + FastAPI REST

## 架構

```
src/rag/
├── core/           # 基礎層：config、types、LLM client、embedding、tracing
├── ingestion/      # 前處理：loader（PDF/DOCX/TXT）、chunker、indexer
├── retrieval/      # 檢索層：QdrantStore、retriever
├── generation/     # 生成層：build_context、generate_answer
├── pipeline/       # LangGraph 工作流
│   ├── state.py    # PipelineState
│   ├── builder.py  # build_pipeline / run_pipeline
│   └── nodes/      # retrieve → review_retrieval → rewrite → generate → review_generation → evaluate
└── api/            # FastAPI（/health /ingest /chat /documents）
```

### Pipeline 流程

全部功能啟用時（預設）：

```
[Query]
   │
   ▼
[retrieve]          ── Qdrant 向量搜尋
   │
   ▼
[review_retrieval]  ── 審查 chunks 是否足以回答問題
   │
   ├── passed ──────────────────────────────────────────▶ [generate]
   └── failed ──▶ [rewrite] ──▶ [retrieve] ──▶ ...（最多重試 N 次）
                                                          │
                                                          ▼
                                                       [generate]  ── Azure OpenAI 生成回答
                                                          │
                                                          ▼
                                                  [review_generation] ── 審查回答是否與參考資料一致
                                                          │
                                                   ├── passed ──▶ [evaluate]
                                                   └── failed ──▶ [generate]（最多重試 N 次）
                                                                       │
                                                                       ▼
                                                                  [evaluate]  ── LLM-as-judge 評分
                                                                       │
                                                                       ▼
                                                                      END
```

各 node 說明：

| Node | 說明 |
|------|------|
| `retrieve` | Qdrant 向量搜尋，retry 時排除已確認的 chunks |
| `review_retrieval` | 審查 chunks 相關性，輸出 `confirmed_chunks` |
| `rewrite` | 根據 retrieval review 的 feedback 改寫查詢 |
| `generate` | 根據 `confirmed_chunks` 生成回答 |
| `review_generation` | 審查回答是否忠實於參考資料 |
| `evaluate` | LLM-as-judge 評分（faithfulness / answer_relevance / context_precision，各 0–5） |

## 快速開始

### 1. 安裝依賴

```bash
uv sync
```

### 2. 設定環境變數

```bash
cp .env.example .env
```

編輯 `.env`（必填）：

```env
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
```

### 3. 啟動 Qdrant

```bash
docker compose up -d
```

### 4. 啟動服務

```bash
# 只啟動 API（http://localhost:8000）
bash start_api.sh

# API + Streamlit UI（http://localhost:8501）
bash start_ui.sh
```

## 環境變數參考

### Azure OpenAI

| 變數 | 必填 | 預設值 | 說明 |
|------|------|--------|------|
| `AZURE_OPENAI_API_KEY` | ✓ | — | Azure OpenAI API Key |
| `AZURE_OPENAI_ENDPOINT` | ✓ | — | Azure OpenAI Endpoint |
| `AZURE_OPENAI_API_VERSION` | | `2024-06-01` | API 版本 |
| `AZURE_OPENAI_CHAT_DEPLOYMENT` | | `gpt-4o` | Chat model deployment 名稱 |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | | `text-embedding-3-small` | Embedding model deployment 名稱 |

### Qdrant

| 變數 | 預設值 | 說明 |
|------|--------|------|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant 連線位址 |
| `QDRANT_COLLECTION` | `documents` | Collection 名稱 |

### 文件切割

| 變數 | 預設值 | 說明 |
|------|--------|------|
| `CHUNK_SIZE` | `500` | 每個 chunk 的 token 數 |
| `CHUNK_OVERLAP` | `50` | 相鄰 chunk 重疊 token 數 |

### Pipeline 功能開關

| 變數 | 預設值 | 說明 |
|------|--------|------|
| `MAX_REVIEW_RETRIES` | `2` | 審查失敗最大重試次數 |
| `ENABLE_REVIEW_RETRIEVAL` | `true` | 啟用 review_retrieval + rewrite loop |
| `ENABLE_REVIEW_GENERATION` | `true` | 啟用 review_generation loop |
| `ENABLE_EVALUATE` | `true` | 啟用最終 LLM-as-judge 評估 |

### Langfuse 可觀測性

| 變數 | 預設值 | 說明 |
|------|--------|------|
| `LANGFUSE_SECRET_KEY` | 空（停用） | Langfuse Secret Key，填入後自動啟用追蹤 |
| `LANGFUSE_PUBLIC_KEY` | 空 | Langfuse Public Key |
| `LANGFUSE_HOST` | `https://cloud.langfuse.com` | Langfuse 位址（自架時修改） |

Langfuse 啟用後，每次 `/chat` 請求會在 Langfuse 建立一個 trace，每個 pipeline node 各有一個 span，記錄 input / output。

## API 端點

| 方法 | 路徑 | 說明 |
|------|------|------|
| GET | `/health` | 健康檢查 |
| POST | `/ingest` | 上傳並處理文件 |
| POST | `/chat?query=...` | 文件問答 |
| GET | `/documents` | 列出已上傳文件 |
| DELETE | `/documents/{source_file}` | 刪除文件 |

API 文件（Swagger）：http://localhost:8000/docs

### `/chat` 回傳格式

```json
{
  "answer": "根據資料...",
  "query": "原始問題",
  "rewritten_query": "改寫後的查詢",
  "sources": [
    { "chunk_id": "...", "text": "...", "source_file": "...", "similarity": 0.92 }
  ],
  "review_passed": true,
  "eval_scores": {
    "faithfulness": 5,
    "answer_relevance": 4,
    "context_precision": 5,
    "reasoning": {
      "faithfulness": "回答完全基於參考資料",
      "answer_relevance": "回答切中問題但略有遺漏",
      "context_precision": "所有 chunks 均與問題高度相關"
    }
  }
}
```

## 測試

```bash
uv run pytest
uv run pytest --cov=rag
```

## 技術棧

| 元件 | 版本 |
|------|------|
| Python | ≥ 3.11 |
| LangGraph | ≥ 0.2 |
| Qdrant Client | ≥ 1.9 |
| FastAPI | ≥ 0.115 |
| Streamlit | ≥ 1.38 |
| Langfuse | ≥ 4.0 |
| pymupdf4llm | ≥ 0.0.17 |
