# @peleke.s/langchain-qortex

LangChain.js VectorStore backed by [qortex](https://github.com/Peleke/qortex) knowledge graph. Graph-enhanced retrieval via MCP.

Drop-in replacement for MemoryVectorStore, Chroma, Pinecone, or any LangChain VectorStore. Same API. Same chains. Same retriever. Plus graph structure, rules, and feedback-driven learning.

## Install

```bash
npm install @peleke.s/langchain-qortex @langchain/core
```

## Quick Start

```typescript
import { QortexVectorStore } from "@peleke.s/langchain-qortex";
import { OpenAIEmbeddings } from "@langchain/openai";

// Create store with any LangChain embeddings
const store = new QortexVectorStore(new OpenAIEmbeddings(), {
  indexName: "my-docs",
  domain: "engineering",
});
await store.connect();

// Add documents (standard LangChain)
await store.addDocuments([
  { pageContent: "OAuth2 authorization framework", metadata: { source: "rfc6749" } },
  { pageContent: "JWT token validation", metadata: { source: "rfc7519" } },
]);

// Search (graph-enhanced: embedding + PPR + rules)
const docs = await store.similaritySearch("authentication", 5);
// docs[0].metadata.node_id  -> graph node ID
// docs[0].metadata.rules    -> linked rules from the knowledge graph

// Use as retriever in any LangChain chain
const retriever = store.asRetriever({ k: 10 });
```

## Graph Extras

Beyond standard VectorStore operations, QortexVectorStore exposes qortex's graph capabilities:

```typescript
// Explore a concept's graph neighborhood
const neighborhood = await store.explore(docs[0].metadata.node_id);
// neighborhood.edges     -> typed relationships (REQUIRES, EXTENDS, etc.)
// neighborhood.neighbors -> connected concepts
// neighborhood.rules     -> linked rules

// Get projected rules
const rules = await store.getRules({ domains: ["engineering"] });

// Close the feedback loop (improves future retrieval)
await store.feedback({
  [docs[0].id]: "accepted",
  [docs[1].id]: "rejected",
});
```

## API

### `QortexVectorStore`

Extends `VectorStore` from `@langchain/core`.

| Method | Description |
|--------|-------------|
| `addDocuments(docs, options?)` | Embed and store documents |
| `addVectors(vectors, docs, options?)` | Store pre-computed vectors |
| `similaritySearch(query, k, filter?)` | Graph-enhanced text search (uses qortex_query) |
| `similaritySearchWithScore(query, k, filter?)` | Same, with scores |
| `similaritySearchVectorWithScore(vector, k, filter?)` | Raw vector search (uses qortex_vector_query) |
| `asRetriever(options?)` | Create a LangChain retriever |
| `explore(nodeId, depth?)` | Explore graph neighborhood |
| `getRules(options?)` | Get projected rules |
| `feedback(outcomes)` | Report feedback for learning |
| `connect()` / `disconnect()` | MCP lifecycle |

### `QortexEmbeddings`

Wraps a qortex-style embedding model (`.embed(texts)`) in LangChain's `Embeddings` interface.

```typescript
import { QortexEmbeddings } from "@peleke.s/langchain-qortex";

const embeddings = new QortexEmbeddings({ model: myQortexModel });
```

### Configuration

```typescript
interface QortexVectorStoreConfig {
  serverCommand?: string;     // Default: "uvx"
  serverArgs?: string[];      // Default: ["qortex", "mcp-serve"]
  serverEnv?: Record<string, string>;
  mcpClient?: Client;         // Pre-configured MCP client
  indexName?: string;          // Default: "default"
  domain?: string;             // Default: "default"
  feedbackSource?: string;    // Default: "langchain"
}
```

## Architecture

```
LangChain App
    |
    v
QortexVectorStore (extends VectorStore)
    |
    v
QortexMcpClient (MCP SDK, stdio transport)
    |
    v
qortex MCP Server (Python, spawned via uvx)
    |
    v
Knowledge Graph + Vector Index
```

Text-level search (`similaritySearch`) uses qortex's full pipeline: embedding + graph PPR + rules. Vector-level search (`similaritySearchVectorWithScore`) provides standard LangChain compatibility.

## Comparison with Python Version

| Feature | `langchain-qortex` (Python) | `@peleke.s/langchain-qortex` (TypeScript) |
|---------|----------------------------|----------------------------------------|
| Transport | Direct (LocalQortexClient) | MCP (stdio subprocess) |
| VectorStore | `langchain_core.vectorstores` | `@langchain/core/vectorstores` |
| Graph extras | `explore()`, `rules()`, `feedback()` | `explore()`, `getRules()`, `feedback()` |
| Embeddings | `QortexEmbeddings` | `QortexEmbeddings` |
| Retriever | `as_retriever()` | `asRetriever()` |

## Development

```bash
npm install
npm run build
npm test                          # Unit tests (mock MCP)
npm run test:e2e                  # E2E (requires uvx + qortex)
npm run test:dogfood              # Dogfood (full import path test)
npm run lint                      # TypeScript type check
```

## License

MIT
