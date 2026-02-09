/**
 * QortexEmbeddings: wrap a qortex-style embedding model in LangChain's
 * Embeddings interface.
 *
 * Any object with `.embed(texts) -> number[][]` works. This is a thin
 * utility -- most users will bring their own LangChain embeddings
 * (OpenAI, HuggingFace, etc.) and pass them to QortexVectorStore directly.
 */

import { Embeddings, type EmbeddingsParams } from "@langchain/core/embeddings";

export interface QortexEmbeddingsParams extends EmbeddingsParams {
  /** Any object with `.embed(texts: string[]) -> Promise<number[][]> | number[][]`. */
  model: { embed(texts: string[]): number[][] | Promise<number[][]> };
}

export class QortexEmbeddings extends Embeddings {
  private model: { embed(texts: string[]): number[][] | Promise<number[][]> };

  constructor(params: QortexEmbeddingsParams) {
    super(params);
    this.model = params.model;
  }

  async embedDocuments(texts: string[]): Promise<number[][]> {
    return await this.model.embed(texts);
  }

  async embedQuery(text: string): Promise<number[]> {
    const result = await this.model.embed([text]);
    return result[0];
  }
}
