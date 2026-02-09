/**
 * Real E2E integration test for @peleke/langchain-qortex.
 *
 * Spawns the actual qortex MCP server (via uvx), connects over stdio,
 * and runs the full VectorStore lifecycle through real MCP transport.
 * No mocks.
 *
 * Prerequisites:
 *   - uvx installed (pip install uv)
 *   - qortex >= 0.2.0 on PyPI (has qortex_vector_* tools)
 *
 * Run:
 *   npx vitest run tests/e2e.test.ts
 */

import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { Document } from "@langchain/core/documents";
import type { EmbeddingsInterface } from "@langchain/core/embeddings";
import { QortexVectorStore } from "../src/vectorstore.js";

const E2E_TIMEOUT = 60_000;

/** Deterministic fake embeddings for E2E testing. */
function createFakeEmbeddings(): EmbeddingsInterface {
  return {
    embedDocuments: async (texts: string[]) =>
      texts.map((_, i) => {
        const v = [0, 0, 0, 0];
        v[i % 4] = 1;
        return v;
      }),
    embedQuery: async () => [1, 0, 0, 0],
  } as unknown as EmbeddingsInterface;
}

describe("Real E2E: QortexVectorStore over stdio MCP", () => {
  let store: QortexVectorStore;
  const embeddings = createFakeEmbeddings();

  beforeAll(async () => {
    store = new QortexVectorStore(embeddings, {
      serverCommand: "uvx",
      serverArgs: ["qortex", "mcp-serve"],
      indexName: "e2e-langchain",
      domain: "e2e-test",
    });
    await store.connect();
  }, E2E_TIMEOUT);

  afterAll(async () => {
    await store.disconnect();
  });

  // ---------------------------------------------------------------------------
  // VectorStore abstract methods
  // ---------------------------------------------------------------------------

  it(
    "addVectors + similaritySearchVectorWithScore",
    async () => {
      const docs = [
        new Document({ pageContent: "OAuth2 authorization", metadata: { category: "auth" } }),
        new Document({ pageContent: "JWT validation", metadata: { category: "auth" } }),
        new Document({ pageContent: "Rate limiting", metadata: { category: "infra" } }),
      ];

      const ids = await store.addVectors(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
        docs,
        { ids: ["e2e-1", "e2e-2", "e2e-3"] },
      );
      expect(ids).toEqual(["e2e-1", "e2e-2", "e2e-3"]);

      const results = await store.similaritySearchVectorWithScore(
        [1, 0, 0, 0],
        2,
      );
      expect(results.length).toBeGreaterThanOrEqual(1);
      expect(results[0][0].id).toBe("e2e-1");
      expect(results[0][1]).toBeGreaterThan(0.9);
    },
    E2E_TIMEOUT,
  );

  it(
    "addDocuments embeds then upserts",
    async () => {
      const docs = [
        new Document({ pageContent: "Circuit breakers", metadata: { category: "infra" } }),
      ];

      const ids = await store.addDocuments(docs, { ids: ["e2e-4"] });
      expect(ids).toEqual(["e2e-4"]);
    },
    E2E_TIMEOUT,
  );

  // ---------------------------------------------------------------------------
  // Full lifecycle
  // ---------------------------------------------------------------------------

  it(
    "full lifecycle: add -> search -> retriever compatibility",
    async () => {
      // Search (vector-level)
      const vecResults = await store.similaritySearchVectorWithScore(
        [1, 0, 0, 0],
        3,
      );
      expect(vecResults.length).toBeGreaterThanOrEqual(1);

      // Retriever (inherited from VectorStore)
      const retriever = store.asRetriever(2);
      expect(retriever).toBeDefined();
      expect(retriever.k).toBe(2);
    },
    E2E_TIMEOUT,
  );
});
