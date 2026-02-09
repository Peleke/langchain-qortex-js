/**
 * Qortex-specific types for graph-enhanced vector store capabilities.
 *
 * These represent the graph structure that makes qortex different from
 * a plain vector store -- nodes, edges, rules, and feedback.
 */

/** A node in the qortex knowledge graph. */
export interface QortexNode {
  id: string;
  name: string;
  description: string;
  domain: string;
  confidence: number;
  properties: Record<string, unknown>;
}

/** A typed edge between two nodes. */
export interface QortexEdge {
  source_id: string;
  target_id: string;
  relation_type: string;
  confidence: number;
  properties: Record<string, unknown>;
}

/** A rule surfaced from the knowledge graph. */
export interface QortexRule {
  id: string;
  text: string;
  domain: string;
  category: string | null;
  confidence: number;
  relevance: number;
  derivation: string;
  source_concepts: string[];
  metadata: Record<string, unknown>;
}

/** Result of exploring a node's neighborhood. */
export interface ExploreResult {
  node: QortexNode;
  edges: QortexEdge[];
  rules: QortexRule[];
  neighbors: QortexNode[];
}

/** Result of a rules projection query. */
export interface RulesResult {
  rules: QortexRule[];
  domain_count: number;
  projection: string;
}

/** Outcome feedback for a query result. */
export type FeedbackOutcome = "accepted" | "rejected" | "partial";

/** Feedback submission result. */
export interface FeedbackResult {
  status: string;
  query_id: string;
  outcome_count: number;
  source: string;
}

/** Query result from text-level qortex_query. */
export interface QortexQueryItem {
  id: string;
  content: string;
  score: number;
  domain: string;
  node_id: string;
  metadata: Record<string, unknown>;
}

/** Full query result including rules. */
export interface QortexQueryResult {
  items: QortexQueryItem[];
  query_id: string;
  rules: QortexRule[];
}

/** Domain info. */
export interface QortexDomainInfo {
  name: string;
  description: string | null;
  concept_count: number;
  edge_count: number;
  rule_count: number;
}
