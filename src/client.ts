/**
 * MCP client wrapper for communicating with the qortex MCP server.
 *
 * Handles connection lifecycle and tool invocation. The qortex server
 * is spawned as a subprocess (stdio transport) or connected to an
 * existing server.
 */

import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

export interface QortexMcpClientConfig {
  /** Command to spawn the qortex MCP server (default: "uvx") */
  serverCommand?: string;
  /** Arguments for the server command (default: ["qortex", "mcp-serve"]) */
  serverArgs?: string[];
  /** Environment variables for the server process */
  serverEnv?: Record<string, string>;
  /** Pre-configured MCP client (skip spawning) */
  mcpClient?: Client;
}

export class QortexMcpClient {
  private client: Client | null = null;
  private transport: StdioClientTransport | null = null;
  private config: QortexMcpClientConfig;
  private _connected = false;

  constructor(config: QortexMcpClientConfig = {}) {
    this.config = config;
    if (config.mcpClient) {
      this.client = config.mcpClient;
      this._connected = true;
    }
  }

  get connected(): boolean {
    return this._connected;
  }

  async connect(): Promise<void> {
    if (this._connected) return;

    const command = this.config.serverCommand ?? "uvx";
    const args = this.config.serverArgs ?? ["qortex", "mcp-serve"];
    const env: Record<string, string> = {};
    for (const [k, v] of Object.entries(process.env)) {
      if (v !== undefined) env[k] = v;
    }
    if (this.config.serverEnv) {
      Object.assign(env, this.config.serverEnv);
    }

    this.transport = new StdioClientTransport({
      command,
      args,
      env,
    });

    this.client = new Client(
      { name: "langchain-qortex", version: "0.1.0" },
      { capabilities: {} },
    );

    await this.client.connect(this.transport);
    this._connected = true;
  }

  async disconnect(): Promise<void> {
    if (this.transport) {
      await this.transport.close();
      this.transport = null;
    }
    this.client = null;
    this._connected = false;
  }

  /**
   * Call a qortex MCP tool and return the parsed result.
   *
   * @param name - Tool name (e.g. "qortex_vector_query")
   * @param args - Tool arguments as a plain object
   * @returns Parsed JSON result from the tool
   */
  async callTool(
    name: string,
    args: Record<string, unknown>,
  ): Promise<unknown> {
    if (!this._connected) {
      await this.connect();
    }

    const result = await this.client!.callTool({
      name,
      arguments: args,
    });

    // MCP tool results come as content blocks
    const content = result.content as Array<{ type: string; text?: string }>;
    if (!content || content.length === 0) {
      return {};
    }

    const textBlock = content.find((c) => c.type === "text");
    if (!textBlock?.text) {
      return {};
    }

    // Handle MCP error responses that aren't valid JSON
    try {
      return JSON.parse(textBlock.text);
    } catch {
      if (result.isError) {
        throw new Error(`MCP tool error: ${textBlock.text}`);
      }
      return { raw: textBlock.text };
    }
  }
}
