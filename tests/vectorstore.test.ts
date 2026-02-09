/**
 * Unit tests for QortexVectorStore.
 *
 * Uses a mock MCP client to test VectorStore compliance, graph-enhanced
 * search, add operations, graph extras, and lifecycle.
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { Document } from "@langchain/core/documents";
import type { EmbeddingsInterface } from "@langchain/core/embeddings";
import type { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { QortexVectorStore } from "../src/vectorstore.js";

// ---------------------------------------------------------------------------
// Mock helpers
// ---------------------------------------------------------------------------

function createMockClient(): {
  client: Client;
  callTool: ReturnType<typeof vi.fn>;
} {
  const callTool = vi.fn();
  const client = { callTool } as unknown as Client;
  return { client, callTool };
}

function mockResponse(data: unknown) {
  return {
    content: [{ type: "text", text: JSON.stringify(data) }],
  };
}

function createFakeEmbeddings(): EmbeddingsInterface {
  return {
    embedDocuments: vi.fn(async (texts: string[]) =>
      texts.map(() => [0.1, 0.2, 0.3, 0.4]),
    ),
    embedQuery: vi.fn(async () => [0.1, 0.2, 0.3, 0.4]),
  } as unknown as EmbeddingsInterface;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("QortexVectorStore", () => {
  let store: QortexVectorStore;
  let callTool: ReturnType<typeof vi.fn>;
  let embeddings: EmbeddingsInterface;

  beforeEach(() => {
    const mock = createMockClient();
    callTool = mock.callTool;
    embeddings = createFakeEmbeddings();
    store = new QortexVectorStore(embeddings, {
      mcpClient: mock.client,
      indexName: "test-index",
      domain: "security",
    });
  });

  // -----------------------------------------------------------------------
  // VectorStore type
  // -----------------------------------------------------------------------

  it("reports vectorstore type as 'qortex'", () => {
    expect(store._vectorstoreType()).toBe("qortex");
  });

  it("exposes embeddings interface", () => {
    expect(store.embeddings).toBe(embeddings);
  });

  // -----------------------------------------------------------------------
  // addVectors
  // -----------------------------------------------------------------------

  describe("addVectors", () => {
    it("upserts vectors with metadata via MCP", async () => {
      callTool.mockResolvedValue(
        mockResponse({ ids: ["v1", "v2"] }),
      );

      const docs = [
        new Document({ pageContent: "OAuth2 auth", metadata: { source: "docs" } }),
        new Document({ pageContent: "JWT tokens", metadata: { source: "docs" } }),
      ];

      const ids = await store.addVectors(
        [[1, 0, 0, 0], [0, 1, 0, 0]],
        docs,
        { ids: ["v1", "v2"] },
      );

      expect(ids).toEqual(["v1", "v2"]);
      expect(callTool).toHaveBeenCalledWith({
        name: "qortex_vector_upsert",
        arguments: {
          index_name: "test-index",
          vectors: [[1, 0, 0, 0], [0, 1, 0, 0]],
          metadata: [
            { text: "OAuth2 auth", source: "docs" },
            { text: "JWT tokens", source: "docs" },
          ],
          ids: ["v1", "v2"],
        },
      });
    });

    it("auto-generates ids when not provided", async () => {
      callTool.mockResolvedValue(
        mockResponse({ ids: ["auto-1"] }),
      );

      const docs = [new Document({ pageContent: "test" })];
      await store.addVectors([[1, 0, 0, 0]], docs);

      expect(callTool).toHaveBeenCalledWith(
        expect.objectContaining({
          arguments: expect.objectContaining({
            ids: undefined,
          }),
        }),
      );
    });

    it("throws on error response", async () => {
      callTool.mockResolvedValue(
        mockResponse({ error: "Dimension mismatch" }),
      );

      const docs = [new Document({ pageContent: "test" })];
      await expect(
        store.addVectors([[1, 0]], docs),
      ).rejects.toThrow("Dimension mismatch");
    });
  });

  // -----------------------------------------------------------------------
  // addDocuments
  // -----------------------------------------------------------------------

  describe("addDocuments", () => {
    it("embeds documents then upserts", async () => {
      callTool.mockResolvedValue(
        mockResponse({ ids: ["d1", "d2"] }),
      );

      const docs = [
        new Document({ pageContent: "OAuth2 auth", metadata: { category: "auth" } }),
        new Document({ pageContent: "Rate limiting", metadata: { category: "infra" } }),
      ];

      const ids = await store.addDocuments(docs, { ids: ["d1", "d2"] });

      expect(ids).toEqual(["d1", "d2"]);
      expect(embeddings.embedDocuments).toHaveBeenCalledWith([
        "OAuth2 auth",
        "Rate limiting",
      ]);
      expect(callTool).toHaveBeenCalledTimes(1);
    });
  });

  // -----------------------------------------------------------------------
  // similaritySearchVectorWithScore (raw vector search)
  // -----------------------------------------------------------------------

  describe("similaritySearchVectorWithScore", () => {
    it("returns documents with scores from vector query", async () => {
      callTool.mockResolvedValue(
        mockResponse({
          results: [
            { id: "v1", score: 0.95, metadata: { text: "OAuth2 auth", source: "docs" } },
            { id: "v2", score: 0.82, metadata: { text: "JWT tokens", source: "docs" } },
          ],
        }),
      );

      const results = await store.similaritySearchVectorWithScore(
        [1, 0, 0, 0],
        2,
      );

      expect(results).toHaveLength(2);
      expect(results[0][0].pageContent).toBe("OAuth2 auth");
      expect(results[0][0].id).toBe("v1");
      expect(results[0][1]).toBe(0.95);
      expect(results[1][0].pageContent).toBe("JWT tokens");
      expect(results[1][1]).toBe(0.82);
    });

    it("passes filter to MCP", async () => {
      callTool.mockResolvedValue(mockResponse({ results: [] }));

      await store.similaritySearchVectorWithScore(
        [1, 0, 0, 0],
        5,
        { source: "handbook" },
      );

      expect(callTool).toHaveBeenCalledWith(
        expect.objectContaining({
          arguments: expect.objectContaining({
            filter: { source: "handbook" },
          }),
        }),
      );
    });

    it("throws on error", async () => {
      callTool.mockResolvedValue(
        mockResponse({ error: "Index not found" }),
      );

      await expect(
        store.similaritySearchVectorWithScore([1, 0, 0, 0], 5),
      ).rejects.toThrow("Index not found");
    });
  });

  // -----------------------------------------------------------------------
  // similaritySearch (graph-enhanced text search)
  // -----------------------------------------------------------------------

  describe("similaritySearch", () => {
    it("returns documents from text-level qortex_query", async () => {
      callTool.mockResolvedValue(
        mockResponse({
          items: [
            {
              id: "i-1",
              content: "OAuth2 authorization framework",
              score: 0.94,
              domain: "security",
              node_id: "sec:oauth",
              metadata: {},
            },
            {
              id: "i-2",
              content: "JWT token validation",
              score: 0.85,
              domain: "security",
              node_id: "sec:jwt",
              metadata: {},
            },
          ],
          query_id: "q-abc",
          rules: [],
        }),
      );

      const docs = await store.similaritySearch("authentication", 2);

      expect(docs).toHaveLength(2);
      expect(docs[0].pageContent).toBe("OAuth2 authorization framework");
      expect(docs[0].metadata.node_id).toBe("sec:oauth");
      expect(docs[0].metadata.domain).toBe("security");
      expect(docs[0].id).toBe("i-1");
    });

    it("sets lastQueryId", async () => {
      callTool.mockResolvedValue(
        mockResponse({
          items: [],
          query_id: "q-xyz",
          rules: [],
        }),
      );

      await store.similaritySearch("test");
      expect(store.lastQueryId).toBe("q-xyz");
    });
  });

  // -----------------------------------------------------------------------
  // similaritySearchWithScore (graph-enhanced with scores)
  // -----------------------------------------------------------------------

  describe("similaritySearchWithScore", () => {
    it("returns documents with scores and linked rules", async () => {
      callTool.mockResolvedValue(
        mockResponse({
          items: [
            {
              id: "i-1",
              content: "OAuth2 framework",
              score: 0.94,
              domain: "security",
              node_id: "sec:oauth",
              metadata: {},
            },
          ],
          query_id: "q-abc",
          rules: [
            {
              id: "rule:oauth",
              text: "Use OAuth2 for API access",
              domain: "security",
              category: "security",
              confidence: 1.0,
              relevance: 0.94,
              derivation: "explicit",
              source_concepts: ["sec:oauth"],
              metadata: {},
            },
          ],
        }),
      );

      const results = await store.similaritySearchWithScore("OAuth2 auth");

      expect(results).toHaveLength(1);
      const [doc, score] = results[0];
      expect(doc.pageContent).toBe("OAuth2 framework");
      expect(score).toBe(0.94);
      expect(doc.metadata.rules).toEqual([
        { id: "rule:oauth", text: "Use OAuth2 for API access", relevance: 0.94 },
      ]);
    });

    it("passes domain from filter", async () => {
      callTool.mockResolvedValue(
        mockResponse({ items: [], query_id: "q-1", rules: [] }),
      );

      await store.similaritySearchWithScore("test", 5, {
        domains: ["infra"],
        min_confidence: 0.5,
      });

      expect(callTool).toHaveBeenCalledWith(
        expect.objectContaining({
          arguments: expect.objectContaining({
            domains: ["infra"],
            min_confidence: 0.5,
          }),
        }),
      );
    });
  });

  // -----------------------------------------------------------------------
  // Qortex extras
  // -----------------------------------------------------------------------

  describe("explore", () => {
    it("returns graph neighborhood", async () => {
      callTool.mockResolvedValue(
        mockResponse({
          node: {
            id: "sec:oauth",
            name: "OAuth2",
            description: "Auth framework",
            domain: "security",
            confidence: 1.0,
            properties: {},
          },
          edges: [
            {
              source_id: "sec:oauth",
              target_id: "sec:jwt",
              relation_type: "REQUIRES",
              confidence: 0.9,
              properties: {},
            },
          ],
          neighbors: [
            {
              id: "sec:jwt",
              name: "JWT",
              description: "Signed tokens",
              domain: "security",
              confidence: 1.0,
              properties: {},
            },
          ],
          rules: [],
        }),
      );

      const result = await store.explore("sec:oauth");
      expect(result).not.toBeNull();
      expect(result!.node.name).toBe("OAuth2");
      expect(result!.edges).toHaveLength(1);
      expect(result!.edges[0].relation_type).toBe("REQUIRES");
      expect(result!.neighbors).toHaveLength(1);
    });

    it("returns null for missing node", async () => {
      callTool.mockResolvedValue(mockResponse({ node: null }));
      const result = await store.explore("nonexistent");
      expect(result).toBeNull();
    });
  });

  describe("getRules", () => {
    it("queries rules by domain", async () => {
      callTool.mockResolvedValue(
        mockResponse({
          rules: [
            {
              id: "rule:1",
              text: "Use OAuth2",
              domain: "security",
              category: "architectural",
              confidence: 1.0,
              relevance: 0.9,
              derivation: "explicit",
              source_concepts: ["sec:oauth"],
              metadata: {},
            },
          ],
          domain_count: 1,
          projection: "rules",
        }),
      );

      const result = await store.getRules({ domains: ["security"] });
      expect(result.rules).toHaveLength(1);
      expect(result.projection).toBe("rules");
    });
  });

  describe("feedback", () => {
    it("submits feedback outcomes for last query", async () => {
      // First, do a search to set lastQueryId
      callTool.mockResolvedValueOnce(
        mockResponse({
          items: [
            {
              id: "i-1",
              content: "OAuth2",
              score: 0.9,
              domain: "security",
              node_id: "sec:oauth",
              metadata: {},
            },
          ],
          query_id: "q-abc",
          rules: [],
        }),
      );
      await store.similaritySearch("auth");

      // Now submit feedback
      callTool.mockResolvedValueOnce(
        mockResponse({
          status: "recorded",
          query_id: "q-abc",
          outcome_count: 1,
          source: "langchain",
        }),
      );

      const result = await store.feedback({ "i-1": "accepted" });
      expect(result).not.toBeNull();
      expect(result!.status).toBe("recorded");
      expect(result!.outcome_count).toBe(1);
    });

    it("returns null when no query has been made", async () => {
      const result = await store.feedback({ "i-1": "accepted" });
      expect(result).toBeNull();
    });
  });

  // -----------------------------------------------------------------------
  // asRetriever (inherited from VectorStore)
  // -----------------------------------------------------------------------

  describe("asRetriever", () => {
    it("creates a retriever with default k", () => {
      const retriever = store.asRetriever();
      expect(retriever).toBeDefined();
      expect(retriever.k).toBe(4);
    });

    it("creates a retriever with custom k", () => {
      const retriever = store.asRetriever(10);
      expect(retriever.k).toBe(10);
    });
  });

  // -----------------------------------------------------------------------
  // Full VectorStore lifecycle
  // -----------------------------------------------------------------------

  describe("full VectorStore lifecycle", () => {
    it("addDocuments -> similaritySearch -> explore -> rules -> feedback", async () => {
      // 1. addDocuments (embed + upsert)
      callTool.mockResolvedValueOnce(
        mockResponse({ ids: ["d1", "d2", "d3"] }),
      );
      await store.addDocuments([
        new Document({ pageContent: "OAuth2 authorization", metadata: { category: "auth" } }),
        new Document({ pageContent: "JWT validation", metadata: { category: "auth" } }),
        new Document({ pageContent: "Rate limiting", metadata: { category: "infra" } }),
      ]);

      // 2. similaritySearch (text-level, graph-enhanced)
      callTool.mockResolvedValueOnce(
        mockResponse({
          items: [
            {
              id: "i-1",
              content: "OAuth2 authorization",
              score: 0.94,
              domain: "security",
              node_id: "sec:oauth",
              metadata: {},
            },
          ],
          query_id: "q-lifecycle",
          rules: [
            {
              id: "rule:oauth",
              text: "Use OAuth2 for API access",
              domain: "security",
              category: "security",
              confidence: 1.0,
              relevance: 0.94,
              derivation: "explicit",
              source_concepts: ["sec:oauth"],
              metadata: {},
            },
          ],
        }),
      );
      const docs = await store.similaritySearch("authentication", 5);
      expect(docs).toHaveLength(1);
      expect(docs[0].metadata.rules).toHaveLength(1);
      expect(store.lastQueryId).toBe("q-lifecycle");

      // 3. Explore the top result
      callTool.mockResolvedValueOnce(
        mockResponse({
          node: {
            id: "sec:oauth",
            name: "OAuth2",
            description: "Auth framework",
            domain: "security",
            confidence: 1.0,
            properties: {},
          },
          edges: [
            {
              source_id: "sec:oauth",
              target_id: "sec:jwt",
              relation_type: "REQUIRES",
              confidence: 0.9,
              properties: {},
            },
          ],
          neighbors: [
            {
              id: "sec:jwt",
              name: "JWT",
              description: "Tokens",
              domain: "security",
              confidence: 1.0,
              properties: {},
            },
          ],
          rules: [],
        }),
      );
      const explored = await store.explore(docs[0].metadata.node_id as string);
      expect(explored!.edges[0].relation_type).toBe("REQUIRES");

      // 4. Get rules
      callTool.mockResolvedValueOnce(
        mockResponse({
          rules: [
            {
              id: "rule:oauth",
              text: "Use OAuth2 for API access",
              domain: "security",
              category: "security",
              confidence: 1.0,
              relevance: 0.94,
              derivation: "explicit",
              source_concepts: ["sec:oauth"],
              metadata: {},
            },
          ],
          domain_count: 1,
          projection: "rules",
        }),
      );
      const rules = await store.getRules({ domains: ["security"] });
      expect(rules.rules).toHaveLength(1);

      // 5. Feedback
      callTool.mockResolvedValueOnce(
        mockResponse({
          status: "recorded",
          query_id: "q-lifecycle",
          outcome_count: 1,
          source: "langchain",
        }),
      );
      const fb = await store.feedback({ [docs[0].id!]: "accepted" });
      expect(fb!.status).toBe("recorded");

      expect(callTool).toHaveBeenCalledTimes(5);
    });
  });

  // -----------------------------------------------------------------------
  // Full graph-enhanced lifecycle
  // -----------------------------------------------------------------------

  describe("full graph-enhanced lifecycle", () => {
    it("search -> explore -> feedback -> improved retrieval", async () => {
      // 1. First search
      callTool.mockResolvedValueOnce(
        mockResponse({
          items: [
            {
              id: "i-1",
              content: "OAuth2",
              score: 0.80,
              domain: "security",
              node_id: "sec:oauth",
              metadata: {},
            },
            {
              id: "i-2",
              content: "RBAC",
              score: 0.60,
              domain: "security",
              node_id: "sec:rbac",
              metadata: {},
            },
          ],
          query_id: "q-1",
          rules: [],
        }),
      );
      const r1 = await store.similaritySearchWithScore("authentication");
      expect(r1).toHaveLength(2);

      // 2. Explore
      callTool.mockResolvedValueOnce(
        mockResponse({
          node: {
            id: "sec:oauth",
            name: "OAuth2",
            description: "Auth framework",
            domain: "security",
            confidence: 1.0,
            properties: {},
          },
          edges: [],
          neighbors: [],
          rules: [
            {
              id: "rule:1",
              text: "Use OAuth2",
              domain: "security",
              category: "security",
              confidence: 1.0,
              relevance: 0.8,
              derivation: "explicit",
              source_concepts: ["sec:oauth"],
              metadata: {},
            },
          ],
        }),
      );
      const explored = await store.explore(r1[0][0].metadata.node_id as string);
      expect(explored!.rules).toHaveLength(1);

      // 3. Feedback
      callTool.mockResolvedValueOnce(
        mockResponse({
          status: "recorded",
          query_id: "q-1",
          outcome_count: 2,
          source: "langchain",
        }),
      );
      await store.feedback({
        [r1[0][0].id!]: "accepted",
        [r1[1][0].id!]: "rejected",
      });

      // 4. Re-query (improved scores)
      callTool.mockResolvedValueOnce(
        mockResponse({
          items: [
            {
              id: "i-1",
              content: "OAuth2",
              score: 0.92,
              domain: "security",
              node_id: "sec:oauth",
              metadata: {},
            },
          ],
          query_id: "q-2",
          rules: [],
        }),
      );
      const r2 = await store.similaritySearchWithScore("authentication");
      expect(r2[0][1]).toBeGreaterThan(r1[0][1]);

      expect(callTool).toHaveBeenCalledTimes(4);
    });
  });
});
