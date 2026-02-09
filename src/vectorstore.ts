/**
 * QortexVectorStore: LangChain.js VectorStore backed by qortex knowledge graph.
 *
 * Drop-in replacement for MemoryVectorStore, Chroma, Pinecone, or any
 * LangChain VectorStore. Same API. Same chains. Same retriever. Plus
 * graph structure, rules, and feedback-driven learning.
 *
 * Text-level search (similaritySearch, similaritySearchWithScore) uses
 * qortex's full pipeline: embedding + graph PPR + rules. Vector-level
 * search (similaritySearchVectorWithScore) provides raw LangChain
 * compatibility via qortex_vector_query.
 *
 * Usage:
 *   import { QortexVectorStore } from "@peleke/langchain-qortex";
 *
 *   const store = await QortexVectorStore.fromTexts(
 *     texts, metadatas, embeddings, { indexName: "docs" }
 *   );
 *   const docs = await store.similaritySearch("authentication", 5);
 *   const retriever = store.asRetriever({ k: 10 });
 */

import { VectorStore } from "@langchain/core/vectorstores";
import { Document } from "@langchain/core/documents";
import type { EmbeddingsInterface } from "@langchain/core/embeddings";
import type { DocumentInterface } from "@langchain/core/documents";
import { QortexMcpClient, type QortexMcpClientConfig } from "./client.js";
import type {
  ExploreResult,
  RulesResult,
  FeedbackOutcome,
  FeedbackResult,
  QortexQueryResult,
  QortexRule,
} from "./types.js";

export interface QortexVectorStoreConfig extends QortexMcpClientConfig {
  /** Index name for vector operations (default: "default"). */
  indexName?: string;
  /** Default domain for text-level queries (default: "default"). */
  domain?: string;
  /** Source identifier for feedback events (default: "langchain"). */
  feedbackSource?: string;
}

export class QortexVectorStore extends VectorStore {
  declare FilterType: Record<string, unknown>;

  private mcp: QortexMcpClient;
  private indexName: string;
  private domain: string;
  private feedbackSource: string;
  private _lastQueryId: string | null = null;

  constructor(
    embeddings: EmbeddingsInterface,
    config: QortexVectorStoreConfig = {},
  ) {
    super(embeddings, config);
    this.mcp = new QortexMcpClient(config);
    this.indexName = config.indexName ?? "default";
    this.domain = config.domain ?? "default";
    this.feedbackSource = config.feedbackSource ?? "langchain";
  }

  _vectorstoreType(): string {
    return "qortex";
  }

  /** Ensure the MCP connection is established. */
  async connect(): Promise<void> {
    await this.mcp.connect();
  }

  /** Disconnect from the MCP server. */
  async disconnect(): Promise<void> {
    await this.mcp.disconnect();
  }

  /** The query_id from the most recent text-level search. */
  get lastQueryId(): string | null {
    return this._lastQueryId;
  }

  // ---------------------------------------------------------------------------
  // Abstract method implementations (required by VectorStore)
  // ---------------------------------------------------------------------------

  async addVectors(
    vectors: number[][],
    documents: DocumentInterface[],
    options?: { ids?: string[] },
  ): Promise<string[] | void> {
    const metadata = documents.map((doc) => ({
      text: doc.pageContent,
      ...doc.metadata,
    }));
    const ids = options?.ids ?? documents.map((doc) => doc.id).filter(Boolean) as string[];

    const result = (await this.mcp.callTool("qortex_vector_upsert", {
      index_name: this.indexName,
      vectors,
      metadata,
      ids: ids.length > 0 ? ids : undefined,
    })) as { ids?: string[]; error?: string };

    if (result.error) {
      throw new Error(result.error);
    }

    return result.ids;
  }

  async addDocuments(
    documents: DocumentInterface[],
    options?: { ids?: string[] },
  ): Promise<string[] | void> {
    const texts = documents.map((doc) => doc.pageContent);
    const vectors = await this.embeddings.embedDocuments(texts);
    return this.addVectors(vectors, documents, options);
  }

  async similaritySearchVectorWithScore(
    query: number[],
    k: number,
    filter?: this["FilterType"],
  ): Promise<[DocumentInterface, number][]> {
    const result = (await this.mcp.callTool("qortex_vector_query", {
      index_name: this.indexName,
      query_vector: query,
      top_k: k,
      filter: filter ?? undefined,
      include_vector: false,
    })) as {
      results?: Array<{
        id: string;
        score: number;
        metadata?: Record<string, unknown>;
      }>;
      error?: string;
    };

    if (result.error) {
      throw new Error(result.error);
    }

    return (result.results ?? []).map((item) => {
      const { text, ...meta } = (item.metadata ?? {}) as Record<string, unknown>;
      const doc = new Document({
        pageContent: (text as string) ?? "",
        metadata: { ...meta, score: item.score },
        id: item.id,
      });
      return [doc, item.score] as [DocumentInterface, number];
    });
  }

  // ---------------------------------------------------------------------------
  // Overridden methods: graph-enhanced text-level search
  // ---------------------------------------------------------------------------

  async similaritySearch(
    query: string,
    k: number = 4,
    filter?: this["FilterType"],
  ): Promise<DocumentInterface[]> {
    const docsAndScores = await this.similaritySearchWithScore(query, k, filter);
    return docsAndScores.map(([doc]) => doc);
  }

  async similaritySearchWithScore(
    query: string,
    k: number = 4,
    filter?: this["FilterType"],
  ): Promise<[DocumentInterface, number][]> {
    const domains = filter?.domains
      ? (filter.domains as string[])
      : [this.domain];
    const minConfidence =
      typeof filter?.min_confidence === "number" ? filter.min_confidence : 0.0;

    const result = (await this.mcp.callTool("qortex_query", {
      context: query,
      domains,
      top_k: k,
      min_confidence: minConfidence,
      mode: "auto",
    })) as QortexQueryResult;

    this._lastQueryId = result.query_id ?? null;

    return (result.items ?? []).map((item) => {
      const meta: Record<string, unknown> = {
        score: item.score,
        domain: item.domain,
        node_id: item.node_id,
        ...item.metadata,
      };

      if (result.rules?.length) {
        const linkedRules = result.rules
          .filter((r: QortexRule) => r.source_concepts?.includes(item.node_id))
          .map((r: QortexRule) => ({
            id: r.id,
            text: r.text,
            relevance: r.relevance,
          }));
        if (linkedRules.length > 0) {
          meta.rules = linkedRules;
        }
      }

      const doc = new Document({
        pageContent: item.content,
        metadata: meta,
        id: item.id,
      });
      return [doc, item.score] as [DocumentInterface, number];
    });
  }

  // ---------------------------------------------------------------------------
  // Qortex extras: graph exploration + rules + feedback
  // ---------------------------------------------------------------------------

  /**
   * Explore a concept's graph neighborhood.
   * Use node_id from search results metadata to navigate the graph.
   */
  async explore(
    nodeId: string,
    depth: number = 1,
  ): Promise<ExploreResult | null> {
    const result = (await this.mcp.callTool("qortex_explore", {
      node_id: nodeId,
      depth,
    })) as ExploreResult & { node: unknown };

    if (result.node === null) {
      return null;
    }

    return result;
  }

  /** Get projected rules from the knowledge graph. */
  async getRules(
    options: {
      domains?: string[];
      conceptIds?: string[];
      categories?: string[];
      includeDerived?: boolean;
      minConfidence?: number;
    } = {},
  ): Promise<RulesResult> {
    const result = (await this.mcp.callTool("qortex_rules", {
      domains: options.domains ?? undefined,
      concept_ids: options.conceptIds ?? undefined,
      categories: options.categories ?? undefined,
      include_derived: options.includeDerived ?? true,
      min_confidence: options.minConfidence ?? 0.0,
    })) as RulesResult;

    return result;
  }

  /**
   * Report feedback for retrieved items to improve future retrieval.
   * Accepted items get higher PPR teleportation probability; rejected lower.
   */
  async feedback(
    outcomes: Record<string, FeedbackOutcome>,
  ): Promise<FeedbackResult | null> {
    if (!this._lastQueryId) {
      return null;
    }

    const result = (await this.mcp.callTool("qortex_feedback", {
      query_id: this._lastQueryId,
      outcomes,
      source: this.feedbackSource,
    })) as FeedbackResult;

    return result;
  }

  // ---------------------------------------------------------------------------
  // Static factory methods
  // ---------------------------------------------------------------------------

  static async fromTexts(
    texts: string[],
    metadatas: object[] | object,
    embeddings: EmbeddingsInterface,
    config: QortexVectorStoreConfig = {},
  ): Promise<QortexVectorStore> {
    const store = new QortexVectorStore(embeddings, config);
    await store.connect();

    const metaArray = Array.isArray(metadatas)
      ? metadatas
      : texts.map(() => metadatas);

    const docs = texts.map(
      (text, i) =>
        new Document({
          pageContent: text,
          metadata: metaArray[i] as Record<string, unknown>,
        }),
    );

    await store.addDocuments(docs);
    return store;
  }

  static async fromDocuments(
    docs: DocumentInterface[],
    embeddings: EmbeddingsInterface,
    config: QortexVectorStoreConfig = {},
  ): Promise<QortexVectorStore> {
    const store = new QortexVectorStore(embeddings, config);
    await store.connect();
    await store.addDocuments(docs);
    return store;
  }
}
