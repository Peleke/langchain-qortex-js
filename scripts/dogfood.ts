#!/usr/bin/env npx tsx
/**
 * Dogfood script: imports langchain-qortex, plugs it into LangChain,
 * and actually runs a full workflow against a real qortex MCP server.
 *
 * This is NOT a test -- it's a real script that proves the package works
 * end-to-end as a consumer would use it.
 *
 * Prerequisites:
 *   - uvx installed
 *   - qortex >= 0.2.0 on PyPI
 *
 * Run:
 *   npx tsx scripts/dogfood.ts
 */

import { Document } from "@langchain/core/documents";
import type { EmbeddingsInterface } from "@langchain/core/embeddings";
import { QortexVectorStore, QortexEmbeddings } from "../src/index.js";

// Deterministic embeddings for dogfood (no OpenAI key needed)
class DogfoodEmbeddings implements Pick<EmbeddingsInterface, "embedDocuments" | "embedQuery"> {
  async embedDocuments(texts: string[]): Promise<number[][]> {
    return texts.map((t) => {
      // Simple hash-based embedding for reproducibility
      const hash = Array.from(t).reduce((h, c) => (h * 31 + c.charCodeAt(0)) | 0, 0);
      return [
        Math.sin(hash),
        Math.cos(hash),
        Math.sin(hash * 2),
        Math.cos(hash * 2),
      ];
    });
  }

  async embedQuery(text: string): Promise<number[]> {
    return (await this.embedDocuments([text]))[0];
  }
}

async function main() {
  console.log("=== langchain-qortex dogfood ===\n");

  const embeddings = new DogfoodEmbeddings() as unknown as EmbeddingsInterface;

  // 1. Create QortexVectorStore (spawns real MCP server)
  console.log("1. Creating QortexVectorStore...");
  const store = new QortexVectorStore(embeddings, {
    serverCommand: "uvx",
    serverArgs: ["qortex", "mcp-serve"],
    indexName: "dogfood-langchain",
    domain: "dogfood",
    feedbackSource: "langchain-dogfood",
  });
  await store.connect();
  console.log("   Connected to qortex MCP server.\n");

  // 2. Add documents (standard LangChain pattern)
  console.log("2. Adding documents...");
  const docs = [
    new Document({
      pageContent: "OAuth2 is an authorization framework for delegated access control",
      metadata: { source: "rfc6749", topic: "auth" },
    }),
    new Document({
      pageContent: "JWT tokens carry signed claims between two parties",
      metadata: { source: "rfc7519", topic: "auth" },
    }),
    new Document({
      pageContent: "Rate limiting prevents abuse of API endpoints through request throttling",
      metadata: { source: "best-practices", topic: "infra" },
    }),
    new Document({
      pageContent: "Circuit breakers prevent cascading failures in distributed systems",
      metadata: { source: "best-practices", topic: "infra" },
    }),
  ];

  const ids = await store.addDocuments(docs, {
    ids: ["df-1", "df-2", "df-3", "df-4"],
  });
  console.log(`   Added ${ids?.length ?? 0} documents: ${ids}\n`);

  // 3. Vector search (raw, LangChain standard)
  console.log("3. Vector search (similaritySearchVectorWithScore)...");
  const queryVec = await embeddings.embedQuery("authentication");
  const vecResults = await store.similaritySearchVectorWithScore(queryVec, 3);
  for (const [doc, score] of vecResults) {
    console.log(`   [${score.toFixed(3)}] ${doc.pageContent.slice(0, 60)}...`);
  }
  console.log();

  // 4. Retriever (inherited from VectorStore)
  console.log("4. Creating retriever (asRetriever)...");
  const retriever = store.asRetriever({ k: 3 });
  console.log(`   Retriever created with k=${retriever.k}\n`);

  // 5. Check VectorStore type
  console.log(`5. VectorStore type: ${store._vectorstoreType()}`);
  console.log(`   Has embeddings: ${!!store.embeddings}\n`);

  // 6. Cleanup
  console.log("6. Disconnecting...");
  await store.disconnect();
  console.log("   Done.\n");

  console.log("=== Dogfood complete. All LangChain patterns work. ===");
}

main().catch((err) => {
  console.error("Dogfood failed:", err);
  process.exit(1);
});
