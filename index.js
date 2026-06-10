/**
 * @langchain/ainative — LangChain.js integration for AINative/ZeroDB.
 *
 * Provides three LangChain-compatible classes:
 *   - AINativeChatModel   — wraps AINative's OpenAI-compatible chat API
 *   - AINativeEmbeddings   — wraps AINative's embeddings API
 *   - AINativeVectorStore  — uses ZeroDB vectors API for storage/search
 *
 * Auto-provisioning: on first use with no credentials, a free ZeroDB
 * project is created automatically. No signup, no credit card.
 */

import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const AINATIVE_BASE_URL = 'https://api.ainative.studio/api/v1';
const ZERODB_API_BASE = 'https://api.ainative.studio';
const INSTANT_DB_ENDPOINT = `${ZERODB_API_BASE}/api/v1/public/instant-db`;

const DEFAULT_CHAT_MODEL = 'meta-llama/Llama-3.3-70B-Instruct';
const DEFAULT_EMBEDDING_MODEL = 'BAAI/bge-small-en-v1.5';

// ---------------------------------------------------------------------------
// Auto-provisioning (shared with VectorStore)
// ---------------------------------------------------------------------------

async function autoProvision(source = 'langchain-ainative-js') {
  const resp = await fetch(INSTANT_DB_ENDPOINT, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ source }),
  });

  if (!resp.ok) {
    const text = await resp.text().catch(() => '');
    throw new Error(
      `Failed to auto-provision ZeroDB project (${resp.status}): ${text}\n` +
      'Set ZERODB_API_KEY and ZERODB_PROJECT_ID manually, or visit ' +
      'https://ainative.studio to create a free account.'
    );
  }

  const data = await resp.json();
  return {
    apiKey: data.api_key,
    projectId: data.project_id,
    claimUrl: data.claim_url || null,
  };
}

// ---------------------------------------------------------------------------
// AINativeChatModel
// ---------------------------------------------------------------------------

/**
 * LangChain ChatModel backed by AINative's OpenAI-compatible chat API.
 *
 * Uses the free tier by default (Meta Llama 3.3 70B).
 *
 * @example
 * ```js
 * import { AINativeChatModel } from '@langchain/ainative';
 *
 * const model = new AINativeChatModel({ model: 'meta-llama/Llama-3.3-70B-Instruct' });
 * const response = await model.invoke('What is RAG?');
 * console.log(response.content);
 * ```
 */
export class AINativeChatModel extends ChatOpenAI {
  constructor(opts = {}) {
    const apiKey = opts.apiKey || opts.openAIApiKey || process.env.AINATIVE_API_KEY || 'auto';
    const baseURL = opts.baseURL || AINATIVE_BASE_URL;

    super({
      modelName: opts.model || opts.modelName || DEFAULT_CHAT_MODEL,
      openAIApiKey: apiKey,
      configuration: { baseURL },
      ...opts,
      // Ensure these are not overridden by spread
      ...(opts.configuration ? {} : { configuration: { baseURL } }),
    });
  }
}

// ---------------------------------------------------------------------------
// AINativeEmbeddings
// ---------------------------------------------------------------------------

/**
 * LangChain Embeddings backed by AINative's OpenAI-compatible embeddings API.
 *
 * Uses BGE-small by default (free tier).
 *
 * @example
 * ```js
 * import { AINativeEmbeddings } from '@langchain/ainative';
 *
 * const embeddings = new AINativeEmbeddings();
 * const vectors = await embeddings.embedDocuments(['Hello world', 'Goodbye world']);
 * ```
 */
export class AINativeEmbeddings extends OpenAIEmbeddings {
  constructor(opts = {}) {
    const apiKey = opts.apiKey || opts.openAIApiKey || process.env.AINATIVE_API_KEY || 'auto';
    const baseURL = opts.baseURL || AINATIVE_BASE_URL;

    super({
      modelName: opts.model || opts.modelName || DEFAULT_EMBEDDING_MODEL,
      openAIApiKey: apiKey,
      configuration: { baseURL },
      ...opts,
      ...(opts.configuration ? {} : { configuration: { baseURL } }),
    });
  }
}

// ---------------------------------------------------------------------------
// AINativeVectorStore
// ---------------------------------------------------------------------------

/**
 * LangChain VectorStore backed by ZeroDB's vector API.
 *
 * Auto-provisions a free ZeroDB project if no credentials are provided.
 * Supports upsert, similarity search, and delete operations.
 *
 * @example
 * ```js
 * import { AINativeVectorStore } from '@langchain/ainative';
 *
 * const store = new AINativeVectorStore(new AINativeEmbeddings(), {
 *   apiKey: 'your-zerodb-api-key',
 *   projectId: 'your-project-id',
 * });
 *
 * await store.addDocuments([
 *   { pageContent: 'Hello world', metadata: { source: 'test' } },
 * ]);
 *
 * const results = await store.similaritySearch('Hello', 3);
 * ```
 */
export class AINativeVectorStore {
  /**
   * @param {import('@langchain/core/embeddings').Embeddings} embeddings
   * @param {Object} [opts]
   * @param {string} [opts.apiKey]     — ZeroDB API key (env: ZERODB_API_KEY)
   * @param {string} [opts.projectId]  — ZeroDB project ID (env: ZERODB_PROJECT_ID)
   * @param {string} [opts.namespace]  — Vector namespace (default: 'default')
   * @param {string} [opts.baseURL]    — API base URL
   */
  constructor(embeddings, opts = {}) {
    this.embeddings = embeddings;
    this._apiKey = opts.apiKey || process.env.ZERODB_API_KEY || '';
    this._projectId = opts.projectId || process.env.ZERODB_PROJECT_ID || '';
    this._namespace = opts.namespace || 'default';
    this._baseURL = opts.baseURL || ZERODB_API_BASE;
    this._provisionPromise = null;
  }

  // LangChain VectorStore interface requires this
  get lc_namespace() {
    return ['langchain', 'vectorstores', 'ainative'];
  }

  // -----------------------------------------------------------------------
  // Internal: ensure we have a project + API key
  // -----------------------------------------------------------------------

  async _ensureProvisioned() {
    if (this._apiKey && this._projectId) return;

    if (this._provisionPromise) {
      await this._provisionPromise;
      return;
    }

    this._provisionPromise = (async () => {
      const result = await autoProvision('langchain-ainative-js-vectorstore');
      this._projectId = result.projectId;
      this._apiKey = result.apiKey;

      if (result.claimUrl) {
        console.error(
          `[@langchain/ainative] ZeroDB project auto-provisioned (free, 72h trial).`
        );
        console.error(
          `[@langchain/ainative] Claim to keep permanently: ${result.claimUrl}`
        );
      }
    })();

    await this._provisionPromise;
  }

  _vectorsUrl() {
    return `${this._baseURL}/api/v1/public/zerodb/${this._projectId}/vectors`;
  }

  _headers() {
    return {
      'Content-Type': 'application/json',
      'X-API-Key': this._apiKey,
    };
  }

  async _request(url, options = {}) {
    const res = await fetch(url, {
      ...options,
      headers: { ...this._headers(), ...options.headers },
    });

    if (!res.ok) {
      const body = await res.text().catch(() => '');
      const err = new Error(`ZeroDB API error ${res.status}: ${body}`);
      err.statusCode = res.status;
      throw err;
    }

    const contentType = res.headers.get('content-type') || '';
    if (contentType.includes('application/json')) {
      return res.json();
    }
    return res;
  }

  // -----------------------------------------------------------------------
  // LangChain VectorStore interface
  // -----------------------------------------------------------------------

  /**
   * Add documents with their embeddings to the vector store.
   *
   * @param {Array<{pageContent: string, metadata?: object}>} documents
   * @param {Object} [options]
   * @param {string[]} [options.ids] — Custom IDs for each document
   * @returns {Promise<string[]>} — IDs of added documents
   */
  async addDocuments(documents, options = {}) {
    const texts = documents.map((doc) => doc.pageContent);
    const vectors = await this.embeddings.embedDocuments(texts);
    return this.addVectors(vectors, documents, options);
  }

  /**
   * Add pre-computed vectors with their documents.
   *
   * @param {number[][]} vectors
   * @param {Array<{pageContent: string, metadata?: object}>} documents
   * @param {Object} [options]
   * @param {string[]} [options.ids]
   * @returns {Promise<string[]>}
   */
  async addVectors(vectors, documents, options = {}) {
    await this._ensureProvisioned();

    const ids = options.ids || documents.map((_, i) => `doc_${Date.now()}_${i}`);

    const items = vectors.map((vector, i) => ({
      id: ids[i],
      vector,
      metadata: {
        ...(documents[i].metadata || {}),
        pageContent: documents[i].pageContent,
      },
    }));

    await this._request(this._vectorsUrl(), {
      method: 'POST',
      body: JSON.stringify({
        items,
        namespace: this._namespace,
      }),
    });

    return ids;
  }

  /**
   * Search for documents similar to a query string.
   *
   * @param {string} query
   * @param {number} [k=4]
   * @param {object} [filter]
   * @returns {Promise<Array<{pageContent: string, metadata: object}>>}
   */
  async similaritySearch(query, k = 4, filter) {
    const results = await this.similaritySearchWithScore(query, k, filter);
    return results.map(([doc]) => doc);
  }

  /**
   * Search for documents with similarity scores.
   *
   * @param {string} query
   * @param {number} [k=4]
   * @param {object} [filter]
   * @returns {Promise<Array<[{pageContent: string, metadata: object}, number]>>}
   */
  async similaritySearchWithScore(query, k = 4, filter) {
    const queryVector = await this.embeddings.embedQuery(query);
    return this.similaritySearchVectorWithScore(queryVector, k, filter);
  }

  /**
   * Search using a pre-computed vector.
   *
   * @param {number[]} vector
   * @param {number} [k=4]
   * @param {object} [filter]
   * @returns {Promise<Array<[{pageContent: string, metadata: object}, number]>>}
   */
  async similaritySearchVectorWithScore(vector, k = 4, filter) {
    await this._ensureProvisioned();

    const payload = {
      vector,
      top_k: k,
      namespace: this._namespace,
      include_metadata: true,
    };

    if (filter) {
      payload.filter = filter;
    }

    const data = await this._request(`${this._vectorsUrl()}/search`, {
      method: 'POST',
      body: JSON.stringify(payload),
    });

    const results = Array.isArray(data) ? data : data.results || [];

    return results.map((item) => {
      const metadata = { ...(item.metadata || {}) };
      const pageContent = metadata.pageContent || '';
      delete metadata.pageContent;

      return [
        { pageContent, metadata },
        item.score ?? item.similarity ?? 0,
      ];
    });
  }

  /**
   * Delete documents by their IDs.
   *
   * @param {Object} params
   * @param {string[]} params.ids
   */
  async delete(params) {
    await this._ensureProvisioned();

    if (!params?.ids?.length) return;

    await this._request(`${this._vectorsUrl()}/delete`, {
      method: 'POST',
      body: JSON.stringify({
        ids: params.ids,
        namespace: this._namespace,
      }),
    });
  }

  /**
   * Create a vector store from documents (static factory method).
   *
   * @param {Array<{pageContent: string, metadata?: object}>} docs
   * @param {import('@langchain/core/embeddings').Embeddings} embeddings
   * @param {Object} [opts]
   * @returns {Promise<AINativeVectorStore>}
   */
  static async fromDocuments(docs, embeddings, opts = {}) {
    const store = new AINativeVectorStore(embeddings, opts);
    await store.addDocuments(docs);
    return store;
  }

  /**
   * Create a vector store from pre-computed vectors (static factory method).
   *
   * @param {Array<[{pageContent: string, metadata?: object}, number[]]>} pairs
   * @param {import('@langchain/core/embeddings').Embeddings} embeddings
   * @param {Object} [opts]
   * @returns {Promise<AINativeVectorStore>}
   */
  static async fromTexts(texts, metadatas, embeddings, opts = {}) {
    const docs = texts.map((text, i) => ({
      pageContent: text,
      metadata: Array.isArray(metadatas) ? metadatas[i] || {} : metadatas || {},
    }));
    return AINativeVectorStore.fromDocuments(docs, embeddings, opts);
  }
}

// ---------------------------------------------------------------------------
// Convenience: re-export auto-provision for advanced use
// ---------------------------------------------------------------------------

export { autoProvision };

// Default export for convenience
export default {
  AINativeChatModel,
  AINativeEmbeddings,
  AINativeVectorStore,
  autoProvision,
};
