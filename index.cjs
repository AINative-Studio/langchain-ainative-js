/**
 * @langchain/ainative — CommonJS entry point.
 *
 * LangChain.js integration for AINative/ZeroDB.
 * Provides ChatModel, Embeddings, and VectorStore classes.
 */

'use strict';

const AINATIVE_BASE_URL = 'https://api.ainative.studio/api/v1';
const ZERODB_API_BASE = 'https://api.ainative.studio';
const INSTANT_DB_ENDPOINT = `${ZERODB_API_BASE}/api/v1/public/instant-db`;

const DEFAULT_CHAT_MODEL = 'meta-llama/Llama-3.3-70B-Instruct';
const DEFAULT_EMBEDDING_MODEL = 'BAAI/bge-small-en-v1.5';

// ---------------------------------------------------------------------------
// Auto-provisioning
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
// AINativeChatModel (CJS wrapper)
// ---------------------------------------------------------------------------

class AINativeChatModel {
  constructor(opts = {}) {
    const apiKey = opts.apiKey || opts.openAIApiKey || process.env.AINATIVE_API_KEY || 'auto';
    const baseURL = opts.baseURL || AINATIVE_BASE_URL;

    // Lazy-load @langchain/openai to avoid top-level ESM issues
    const { ChatOpenAI } = require('@langchain/openai');

    const parentOpts = {
      modelName: opts.model || opts.modelName || DEFAULT_CHAT_MODEL,
      openAIApiKey: apiKey,
      configuration: { baseURL },
      ...opts,
    };
    if (!opts.configuration) {
      parentOpts.configuration = { baseURL };
    }

    // Return an instance of ChatOpenAI with AINative config
    const instance = new ChatOpenAI(parentOpts);
    return instance;
  }
}

// ---------------------------------------------------------------------------
// AINativeEmbeddings (CJS wrapper)
// ---------------------------------------------------------------------------

class AINativeEmbeddings {
  constructor(opts = {}) {
    const apiKey = opts.apiKey || opts.openAIApiKey || process.env.AINATIVE_API_KEY || 'auto';
    const baseURL = opts.baseURL || AINATIVE_BASE_URL;

    const { OpenAIEmbeddings } = require('@langchain/openai');

    const parentOpts = {
      modelName: opts.model || opts.modelName || DEFAULT_EMBEDDING_MODEL,
      openAIApiKey: apiKey,
      configuration: { baseURL },
      ...opts,
    };
    if (!opts.configuration) {
      parentOpts.configuration = { baseURL };
    }

    const instance = new OpenAIEmbeddings(parentOpts);
    return instance;
  }
}

// ---------------------------------------------------------------------------
// AINativeVectorStore (CJS)
// ---------------------------------------------------------------------------

class AINativeVectorStore {
  constructor(embeddings, opts = {}) {
    this.embeddings = embeddings;
    this._apiKey = opts.apiKey || process.env.ZERODB_API_KEY || '';
    this._projectId = opts.projectId || process.env.ZERODB_PROJECT_ID || '';
    this._namespace = opts.namespace || 'default';
    this._baseURL = opts.baseURL || ZERODB_API_BASE;
    this._provisionPromise = null;
  }

  get lc_namespace() {
    return ['langchain', 'vectorstores', 'ainative'];
  }

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

  async addDocuments(documents, options = {}) {
    const texts = documents.map((doc) => doc.pageContent);
    const vectors = await this.embeddings.embedDocuments(texts);
    return this.addVectors(vectors, documents, options);
  }

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

  async similaritySearch(query, k = 4, filter) {
    const results = await this.similaritySearchWithScore(query, k, filter);
    return results.map(([doc]) => doc);
  }

  async similaritySearchWithScore(query, k = 4, filter) {
    const queryVector = await this.embeddings.embedQuery(query);
    return this.similaritySearchVectorWithScore(queryVector, k, filter);
  }

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

  static async fromDocuments(docs, embeddings, opts = {}) {
    const store = new AINativeVectorStore(embeddings, opts);
    await store.addDocuments(docs);
    return store;
  }

  static async fromTexts(texts, metadatas, embeddings, opts = {}) {
    const docs = texts.map((text, i) => ({
      pageContent: text,
      metadata: Array.isArray(metadatas) ? metadatas[i] || {} : metadatas || {},
    }));
    return AINativeVectorStore.fromDocuments(docs, embeddings, opts);
  }
}

// ---------------------------------------------------------------------------
// Exports
// ---------------------------------------------------------------------------

module.exports = {
  AINativeChatModel,
  AINativeEmbeddings,
  AINativeVectorStore,
  autoProvision,
};
