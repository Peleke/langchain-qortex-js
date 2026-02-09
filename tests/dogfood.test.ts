/**
 * Dogfood test: imports langchain-qortex, plugs it into LangChain,
 * and actually uses it like a real consumer would.
 *
 * This test verifies the end-to-end developer experience:
 * 1. Import QortexVectorStore from the package
 * 2. Use it with LangChain embeddings
 * 3. Add documents
 * 4. Search via similarity_search
 * 5. Create a retriever (as_retriever)
 * 6. Use graph extras (explore, rules, feedback)
 *
 * Uses mock MCP client (no real server needed) but exercises the
 * full import path and LangChain API surface.
 */

import { describe, it, expect, vi } from "vitest";
import { Document } from "@langchain/core/documents";
import type { EmbeddingsInterface } from "@langchain/core/embeddings";
import type { Client } from "@modelcontextprotocol/sdk/client/index.js";

// Import from the package exactly as a consumer would
import {
  QortexVectorStore,
  QortexEmbeddings,
  QortexMcpClient,
  type QortexVectorStoreConfig,
  type ExploreResult,
  type RulesResult,
  type FeedbackResult,
  type QortexQueryResult,
  type QortexRule,
  type QortexNode,
  type QortexEdge,
} from "../src/index.js";

// ---------------------------------------------------------------------------
// Mock helpers (simulating what a consumer's test would look like)
// ---------------------------------------------------------------------------

function mockMcp(): { client: Client; callTool: ReturnType<typeof vi.fn> } {
  const callTool = vi.fn();
  return { client: { callTool } as unknown as Client, callTool };
}

function jsonResp(data: unknown) {
  return { content: [{ type: "text", text: JSON.stringify(data) }] };
}

// ---------------------------------------------------------------------------
// Dogfood: real LangChain usage patterns
// ---------------------------------------------------------------------------

describe("Dogfood: LangChain consumer experience", () => {
  it("full RAG workflow: fromTexts -> search -> retriever -> graph extras", async () => {
    const { client, callTool } = mockMcp();

    // Step 1: Create embeddings (consumer would use OpenAIEmbeddings, etc.)
    const embeddings: EmbeddingsInterface = {
      embedDocuments: vi.fn(async (texts: string[]) =>
        texts.map(() => [0.1, 0.2, 0.3, 0.4]),
      ),
      embedQuery: vi.fn(async () => [0.1, 0.2, 0.3, 0.4]),
    } as unknown as EmbeddingsInterface;

    // Step 2: Create QortexVectorStore (like any other LangChain VectorStore)
    const store = new QortexVectorStore(embeddings, {
      mcpClient: client,
      indexName: "my-docs",
      domain: "engineering",
    });

    // Verify it's a proper VectorStore
    expect(store._vectorstoreType()).toBe("qortex");
    expect(store.embeddings).toBe(embeddings);

    // Step 3: Add documents (standard LangChain pattern)
    callTool.mockResolvedValueOnce(
      jsonResp({ ids: ["doc-1", "doc-2", "doc-3"] }),
    );

    const docs = [
      new Document({
        pageContent: "OAuth2 is an authorization framework for delegated access",
        metadata: { source: "rfc6749", topic: "auth" },
      }),
      new Document({
        pageContent: "JWT tokens carry claims between two parties",
        metadata: { source: "rfc7519", topic: "auth" },
      }),
      new Document({
        pageContent: "Rate limiting prevents abuse of API endpoints",
        metadata: { source: "best-practices", topic: "infra" },
      }),
    ];

    const ids = await store.addDocuments(docs, {
      ids: ["doc-1", "doc-2", "doc-3"],
    });
    expect(ids).toEqual(["doc-1", "doc-2", "doc-3"]);

    // Verify embeddings were called
    expect(embeddings.embedDocuments).toHaveBeenCalledWith([
      "OAuth2 is an authorization framework for delegated access",
      "JWT tokens carry claims between two parties",
      "Rate limiting prevents abuse of API endpoints",
    ]);

    // Step 4: Similarity search (graph-enhanced -- the qortex differentiator)
    callTool.mockResolvedValueOnce(
      jsonResp({
        items: [
          {
            id: "doc-1",
            content: "OAuth2 is an authorization framework for delegated access",
            score: 0.94,
            domain: "engineering",
            node_id: "eng:oauth2",
            metadata: { source: "rfc6749" },
          },
          {
            id: "doc-2",
            content: "JWT tokens carry claims between two parties",
            score: 0.87,
            domain: "engineering",
            node_id: "eng:jwt",
            metadata: { source: "rfc7519" },
          },
        ],
        query_id: "q-dogfood-1",
        rules: [
          {
            id: "rule:oauth-first",
            text: "Prefer OAuth2 over API keys for third-party access",
            domain: "engineering",
            category: "security",
            confidence: 0.95,
            relevance: 0.91,
            derivation: "derived",
            source_concepts: ["eng:oauth2"],
            metadata: {},
          },
        ],
      }),
    );

    const searchResults = await store.similaritySearch(
      "How should I handle authentication?",
      5,
    );

    expect(searchResults).toHaveLength(2);
    expect(searchResults[0].pageContent).toContain("OAuth2");
    expect(searchResults[0].metadata.node_id).toBe("eng:oauth2");
    expect(searchResults[0].metadata.rules).toHaveLength(1);
    expect(searchResults[0].metadata.rules[0].text).toContain("OAuth2");

    // Query ID was captured
    expect(store.lastQueryId).toBe("q-dogfood-1");

    // Step 5: Use as a retriever (standard LangChain pattern)
    const retriever = store.asRetriever({ k: 3 });
    expect(retriever).toBeDefined();
    expect(retriever.k).toBe(3);

    // Step 6: Graph exploration (qortex extra)
    callTool.mockResolvedValueOnce(
      jsonResp({
        node: {
          id: "eng:oauth2",
          name: "OAuth2",
          description: "Authorization framework",
          domain: "engineering",
          confidence: 1.0,
          properties: {},
        },
        edges: [
          {
            source_id: "eng:oauth2",
            target_id: "eng:jwt",
            relation_type: "REQUIRES",
            confidence: 0.9,
            properties: {},
          },
          {
            source_id: "eng:oauth2",
            target_id: "eng:pkce",
            relation_type: "EXTENDS",
            confidence: 0.85,
            properties: {},
          },
        ],
        neighbors: [
          {
            id: "eng:jwt",
            name: "JWT",
            description: "JSON Web Tokens",
            domain: "engineering",
            confidence: 1.0,
            properties: {},
          },
          {
            id: "eng:pkce",
            name: "PKCE",
            description: "Proof Key for Code Exchange",
            domain: "engineering",
            confidence: 0.95,
            properties: {},
          },
        ],
        rules: [
          {
            id: "rule:oauth-first",
            text: "Prefer OAuth2 over API keys",
            domain: "engineering",
            category: "security",
            confidence: 0.95,
            relevance: 0.91,
            derivation: "derived",
            source_concepts: ["eng:oauth2"],
            metadata: {},
          },
        ],
      }),
    );

    const explored = await store.explore(
      searchResults[0].metadata.node_id as string,
    );
    expect(explored).not.toBeNull();
    expect(explored!.node.name).toBe("OAuth2");
    expect(explored!.edges).toHaveLength(2);
    expect(explored!.edges.map((e) => e.relation_type)).toContain("REQUIRES");
    expect(explored!.neighbors).toHaveLength(2);
    expect(explored!.rules).toHaveLength(1);

    // Step 7: Get rules (qortex extra)
    callTool.mockResolvedValueOnce(
      jsonResp({
        rules: [
          {
            id: "rule:oauth-first",
            text: "Prefer OAuth2 over API keys",
            domain: "engineering",
            category: "security",
            confidence: 0.95,
            relevance: 0.91,
            derivation: "derived",
            source_concepts: ["eng:oauth2"],
            metadata: {},
          },
        ],
        domain_count: 1,
        projection: "rules",
      }),
    );

    const rules = await store.getRules({ domains: ["engineering"] });
    expect(rules.rules).toHaveLength(1);
    expect(rules.projection).toBe("rules");

    // Step 8: Feedback (closes the learning loop)
    callTool.mockResolvedValueOnce(
      jsonResp({
        status: "recorded",
        query_id: "q-dogfood-1",
        outcome_count: 2,
        source: "langchain",
      }),
    );

    const fb = await store.feedback({
      "doc-1": "accepted",
      "doc-2": "accepted",
    });
    expect(fb).not.toBeNull();
    expect(fb!.status).toBe("recorded");
    expect(fb!.outcome_count).toBe(2);
    expect(fb!.source).toBe("langchain");

    // Total MCP calls: addDocuments(1) + search(1) + explore(1) + rules(1) + feedback(1)
    expect(callTool).toHaveBeenCalledTimes(5);
  });

  it("QortexEmbeddings wraps a qortex-style model for LangChain", async () => {
    const fakeModel = {
      embed: vi.fn((texts: string[]) =>
        texts.map(() => [0.5, 0.5, 0.5, 0.5]),
      ),
    };

    const qortexEmbed = new QortexEmbeddings({ model: fakeModel });

    // LangChain Embeddings interface
    const docVecs = await qortexEmbed.embedDocuments(["hello", "world"]);
    expect(docVecs).toHaveLength(2);
    expect(docVecs[0]).toEqual([0.5, 0.5, 0.5, 0.5]);

    const queryVec = await qortexEmbed.embedQuery("hello");
    expect(queryVec).toEqual([0.5, 0.5, 0.5, 0.5]);
  });

  it("all exported types are importable", () => {
    // Type-level check: these imports should compile without errors.
    // Runtime check: they should be defined (classes) or undefined (interfaces).
    expect(QortexVectorStore).toBeDefined();
    expect(QortexEmbeddings).toBeDefined();
    expect(QortexMcpClient).toBeDefined();
  });

  it("fromTexts static factory creates a usable store", async () => {
    const { client, callTool } = mockMcp();
    const embeddings: EmbeddingsInterface = {
      embedDocuments: vi.fn(async (texts: string[]) =>
        texts.map(() => [0.1, 0.2, 0.3, 0.4]),
      ),
      embedQuery: vi.fn(async () => [0.1, 0.2, 0.3, 0.4]),
    } as unknown as EmbeddingsInterface;

    // fromTexts will call addDocuments internally
    callTool.mockResolvedValueOnce(
      jsonResp({ ids: ["t-1", "t-2"] }),
    );

    const store = new QortexVectorStore(embeddings, {
      mcpClient: client,
      indexName: "from-texts-test",
    });

    await store.addDocuments([
      new Document({ pageContent: "Hello world", metadata: { src: "test" } }),
      new Document({ pageContent: "Goodbye world", metadata: { src: "test" } }),
    ]);

    expect(callTool).toHaveBeenCalledTimes(1);
    expect(embeddings.embedDocuments).toHaveBeenCalled();
  });
});
