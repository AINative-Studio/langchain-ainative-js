/**
 * @langchain/ainative unit tests.
 *
 * All HTTP calls are mocked — no real API calls.
 * Tests cover AINativeChatModel, AINativeEmbeddings, and AINativeVectorStore.
 */

const {
  AINativeVectorStore,
  autoProvision,
} = require('../index.cjs');

// ---------------------------------------------------------------------------
// Mock fetch
// ---------------------------------------------------------------------------

const mockResponses = [];

function pushMock(status, body, contentType = 'application/json') {
  mockResponses.push({ status, body, contentType });
}

function createMockFetch() {
  return jest.fn(async (url, opts) => {
    const mock = mockResponses.shift();
    if (!mock) throw new Error(`Unexpected fetch call: ${url}`);

    return {
      ok: mock.status >= 200 && mock.status < 300,
      status: mock.status,
      headers: {
        get: (name) => {
          if (name === 'content-type') return mock.contentType;
          return null;
        },
      },
      json: async () =>
        typeof mock.body === 'string' ? JSON.parse(mock.body) : mock.body,
      text: async () =>
        typeof mock.body === 'string' ? mock.body : JSON.stringify(mock.body),
    };
  });
}

beforeEach(() => {
  mockResponses.length = 0;
  globalThis.fetch = createMockFetch();
});

afterEach(() => {
  delete globalThis.fetch;
});

// ---------------------------------------------------------------------------
// Fake embeddings for testing (avoids needing @langchain/openai in tests)
// ---------------------------------------------------------------------------

class FakeEmbeddings {
  constructor() {
    this._dimension = 3;
  }

  async embedDocuments(texts) {
    return texts.map((_, i) => [0.1 * (i + 1), 0.2 * (i + 1), 0.3 * (i + 1)]);
  }

  async embedQuery(text) {
    return [0.1, 0.2, 0.3];
  }
}

// ---------------------------------------------------------------------------
// autoProvision
// ---------------------------------------------------------------------------

describe('autoProvision', () => {
  test('returns credentials on success', async () => {
    pushMock(200, {
      api_key: 'prov-key',
      project_id: 'prov-proj',
      claim_url: 'https://zerodb.ai/claim/abc',
    });

    const result = await autoProvision();
    expect(result.apiKey).toBe('prov-key');
    expect(result.projectId).toBe('prov-proj');
    expect(result.claimUrl).toBe('https://zerodb.ai/claim/abc');
  });

  test('throws on API error', async () => {
    pushMock(500, 'Internal Server Error');

    await expect(autoProvision()).rejects.toThrow('Failed to auto-provision');
  });

  test('sends correct source parameter', async () => {
    pushMock(200, { api_key: 'k', project_id: 'p' });

    await autoProvision('custom-source');

    const [, opts] = globalThis.fetch.mock.calls[0];
    expect(JSON.parse(opts.body).source).toBe('custom-source');
  });
});

// ---------------------------------------------------------------------------
// AINativeVectorStore — constructor
// ---------------------------------------------------------------------------

describe('AINativeVectorStore constructor', () => {
  test('uses provided credentials', () => {
    const store = new AINativeVectorStore(new FakeEmbeddings(), {
      apiKey: 'test-key',
      projectId: 'test-proj',
    });

    expect(store._apiKey).toBe('test-key');
    expect(store._projectId).toBe('test-proj');
    expect(store._namespace).toBe('default');
  });

  test('reads from environment variables', () => {
    process.env.ZERODB_API_KEY = 'env-key';
    process.env.ZERODB_PROJECT_ID = 'env-proj';

    const store = new AINativeVectorStore(new FakeEmbeddings());
    expect(store._apiKey).toBe('env-key');
    expect(store._projectId).toBe('env-proj');

    delete process.env.ZERODB_API_KEY;
    delete process.env.ZERODB_PROJECT_ID;
  });

  test('accepts custom namespace', () => {
    const store = new AINativeVectorStore(new FakeEmbeddings(), {
      apiKey: 'k',
      projectId: 'p',
      namespace: 'my-docs',
    });
    expect(store._namespace).toBe('my-docs');
  });

  test('lc_namespace returns correct path', () => {
    const store = new AINativeVectorStore(new FakeEmbeddings(), {
      apiKey: 'k',
      projectId: 'p',
    });
    expect(store.lc_namespace).toEqual(['langchain', 'vectorstores', 'ainative']);
  });
});

// ---------------------------------------------------------------------------
// AINativeVectorStore — auto-provisioning
// ---------------------------------------------------------------------------

describe('AINativeVectorStore auto-provisioning', () => {
  test('provisions when no credentials', async () => {
    // Mock provision
    pushMock(200, {
      project_id: 'auto-proj',
      api_key: 'auto-key',
      claim_url: 'https://zerodb.ai/claim/xyz',
    });
    // Mock vector upsert
    pushMock(201, { status: 'success' });

    const store = new AINativeVectorStore(new FakeEmbeddings());
    await store.addDocuments([{ pageContent: 'test', metadata: {} }]);

    expect(store._projectId).toBe('auto-proj');
    expect(store._apiKey).toBe('auto-key');
  });

  test('skips provisioning when credentials exist', async () => {
    // Mock vector upsert only (no provision call)
    pushMock(201, { status: 'success' });

    const store = new AINativeVectorStore(new FakeEmbeddings(), {
      apiKey: 'existing',
      projectId: 'existing-proj',
    });
    await store.addDocuments([{ pageContent: 'test', metadata: {} }]);

    // Only one fetch call (upsert), not two (provision + upsert)
    expect(globalThis.fetch).toHaveBeenCalledTimes(1);
  });

  test('deduplicates concurrent provisioning', async () => {
    pushMock(200, {
      project_id: 'dedup-proj',
      api_key: 'dedup-key',
    });
    pushMock(201, { status: 'success' });
    pushMock(201, { status: 'success' });

    const store = new AINativeVectorStore(new FakeEmbeddings());

    // Two concurrent operations that both need provisioning
    await Promise.all([
      store.addDocuments([{ pageContent: 'a', metadata: {} }]),
      store.addDocuments([{ pageContent: 'b', metadata: {} }]),
    ]);

    // Only one provisioning call
    const provisionCalls = globalThis.fetch.mock.calls.filter(([url]) =>
      url.includes('instant-db')
    );
    expect(provisionCalls).toHaveLength(1);
  });
});

// ---------------------------------------------------------------------------
// AINativeVectorStore — addDocuments
// ---------------------------------------------------------------------------

describe('AINativeVectorStore addDocuments', () => {
  test('embeds and upserts documents', async () => {
    pushMock(201, { status: 'success' });

    const store = new AINativeVectorStore(new FakeEmbeddings(), {
      apiKey: 'k',
      projectId: 'p',
    });

    const ids = await store.addDocuments([
      { pageContent: 'Hello world', metadata: { source: 'test' } },
      { pageContent: 'Goodbye world', metadata: { source: 'test2' } },
    ]);

    expect(ids).toHaveLength(2);

    const [url, opts] = globalThis.fetch.mock.calls[0];
    expect(url).toContain('/api/v1/public/zerodb/p/vectors');

    const body = JSON.parse(opts.body);
    expect(body.items).toHaveLength(2);
    expect(body.items[0].vector).toEqual([0.1, 0.2, 0.3]);
    expect(body.items[0].metadata.pageContent).toBe('Hello world');
    expect(body.items[0].metadata.source).toBe('test');
    expect(body.namespace).toBe('default');
  });

  test('uses custom IDs when provided', async () => {
    pushMock(201, { status: 'success' });

    const store = new AINativeVectorStore(new FakeEmbeddings(), {
      apiKey: 'k',
      projectId: 'p',
    });

    const ids = await store.addDocuments(
      [{ pageContent: 'doc1', metadata: {} }],
      { ids: ['custom-id-1'] }
    );

    expect(ids).toEqual(['custom-id-1']);

    const body = JSON.parse(globalThis.fetch.mock.calls[0][1].body);
    expect(body.items[0].id).toBe('custom-id-1');
  });
});

// ---------------------------------------------------------------------------
// AINativeVectorStore — addVectors
// ---------------------------------------------------------------------------

describe('AINativeVectorStore addVectors', () => {
  test('upserts pre-computed vectors', async () => {
    pushMock(201, { status: 'success' });

    const store = new AINativeVectorStore(new FakeEmbeddings(), {
      apiKey: 'k',
      projectId: 'p',
      namespace: 'custom-ns',
    });

    await store.addVectors(
      [[1.0, 2.0, 3.0]],
      [{ pageContent: 'pre-computed', metadata: { manual: true } }]
    );

    const body = JSON.parse(globalThis.fetch.mock.calls[0][1].body);
    expect(body.items[0].vector).toEqual([1.0, 2.0, 3.0]);
    expect(body.namespace).toBe('custom-ns');
  });
});

// ---------------------------------------------------------------------------
// AINativeVectorStore — similaritySearch
// ---------------------------------------------------------------------------

describe('AINativeVectorStore similaritySearch', () => {
  test('returns documents without scores', async () => {
    pushMock(200, {
      results: [
        {
          id: 'v1',
          score: 0.95,
          metadata: { pageContent: 'Hello world', source: 'test' },
        },
        {
          id: 'v2',
          score: 0.8,
          metadata: { pageContent: 'Goodbye world', source: 'test2' },
        },
      ],
    });

    const store = new AINativeVectorStore(new FakeEmbeddings(), {
      apiKey: 'k',
      projectId: 'p',
    });

    const results = await store.similaritySearch('hello', 2);

    expect(results).toHaveLength(2);
    expect(results[0].pageContent).toBe('Hello world');
    expect(results[0].metadata.source).toBe('test');
    // pageContent should NOT be in metadata
    expect(results[0].metadata.pageContent).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// AINativeVectorStore — similaritySearchWithScore
// ---------------------------------------------------------------------------

describe('AINativeVectorStore similaritySearchWithScore', () => {
  test('returns documents with scores', async () => {
    pushMock(200, {
      results: [
        {
          id: 'v1',
          score: 0.95,
          metadata: { pageContent: 'Match!', tag: 'a' },
        },
      ],
    });

    const store = new AINativeVectorStore(new FakeEmbeddings(), {
      apiKey: 'k',
      projectId: 'p',
    });

    const results = await store.similaritySearchWithScore('test', 1);

    expect(results).toHaveLength(1);
    expect(results[0][0].pageContent).toBe('Match!');
    expect(results[0][1]).toBe(0.95);
  });

  test('handles array response format', async () => {
    pushMock(200, [
      {
        id: 'v1',
        similarity: 0.9,
        metadata: { pageContent: 'Array format' },
      },
    ]);

    const store = new AINativeVectorStore(new FakeEmbeddings(), {
      apiKey: 'k',
      projectId: 'p',
    });

    const results = await store.similaritySearchWithScore('test', 1);
    expect(results[0][0].pageContent).toBe('Array format');
    expect(results[0][1]).toBe(0.9);
  });

  test('passes filter to API', async () => {
    pushMock(200, { results: [] });

    const store = new AINativeVectorStore(new FakeEmbeddings(), {
      apiKey: 'k',
      projectId: 'p',
    });

    await store.similaritySearchWithScore('test', 5, { type: 'article' });

    const body = JSON.parse(globalThis.fetch.mock.calls[0][1].body);
    expect(body.filter).toEqual({ type: 'article' });
    expect(body.top_k).toBe(5);
  });
});

// ---------------------------------------------------------------------------
// AINativeVectorStore — similaritySearchVectorWithScore
// ---------------------------------------------------------------------------

describe('AINativeVectorStore similaritySearchVectorWithScore', () => {
  test('searches with pre-computed vector', async () => {
    pushMock(200, {
      results: [
        {
          id: 'v1',
          score: 0.99,
          metadata: { pageContent: 'Direct vector search' },
        },
      ],
    });

    const store = new AINativeVectorStore(new FakeEmbeddings(), {
      apiKey: 'k',
      projectId: 'p',
    });

    const results = await store.similaritySearchVectorWithScore(
      [1.0, 2.0, 3.0],
      1
    );

    expect(results[0][0].pageContent).toBe('Direct vector search');
    expect(results[0][1]).toBe(0.99);

    const body = JSON.parse(globalThis.fetch.mock.calls[0][1].body);
    expect(body.vector).toEqual([1.0, 2.0, 3.0]);
  });
});

// ---------------------------------------------------------------------------
// AINativeVectorStore — delete
// ---------------------------------------------------------------------------

describe('AINativeVectorStore delete', () => {
  test('deletes by IDs', async () => {
    pushMock(200, { status: 'success' });

    const store = new AINativeVectorStore(new FakeEmbeddings(), {
      apiKey: 'k',
      projectId: 'p',
    });

    await store.delete({ ids: ['id1', 'id2'] });

    const [url, opts] = globalThis.fetch.mock.calls[0];
    expect(url).toContain('/vectors/delete');

    const body = JSON.parse(opts.body);
    expect(body.ids).toEqual(['id1', 'id2']);
    expect(body.namespace).toBe('default');
  });

  test('no-ops when no IDs provided', async () => {
    const store = new AINativeVectorStore(new FakeEmbeddings(), {
      apiKey: 'k',
      projectId: 'p',
    });

    await store.delete({ ids: [] });
    await store.delete(null);

    expect(globalThis.fetch).not.toHaveBeenCalled();
  });
});

// ---------------------------------------------------------------------------
// AINativeVectorStore — static factory methods
// ---------------------------------------------------------------------------

describe('AINativeVectorStore static methods', () => {
  test('fromDocuments creates store and adds docs', async () => {
    pushMock(201, { status: 'success' });

    const store = await AINativeVectorStore.fromDocuments(
      [{ pageContent: 'factory doc', metadata: { x: 1 } }],
      new FakeEmbeddings(),
      { apiKey: 'k', projectId: 'p' }
    );

    expect(store).toBeInstanceOf(AINativeVectorStore);
    expect(globalThis.fetch).toHaveBeenCalledTimes(1);
  });

  test('fromTexts creates store from text array', async () => {
    pushMock(201, { status: 'success' });

    const store = await AINativeVectorStore.fromTexts(
      ['text1', 'text2'],
      [{ src: 'a' }, { src: 'b' }],
      new FakeEmbeddings(),
      { apiKey: 'k', projectId: 'p' }
    );

    expect(store).toBeInstanceOf(AINativeVectorStore);

    const body = JSON.parse(globalThis.fetch.mock.calls[0][1].body);
    expect(body.items).toHaveLength(2);
    expect(body.items[0].metadata.pageContent).toBe('text1');
    expect(body.items[0].metadata.src).toBe('a');
  });

  test('fromTexts handles shared metadata object', async () => {
    pushMock(201, { status: 'success' });

    const store = await AINativeVectorStore.fromTexts(
      ['text1'],
      { shared: true },
      new FakeEmbeddings(),
      { apiKey: 'k', projectId: 'p' }
    );

    const body = JSON.parse(globalThis.fetch.mock.calls[0][1].body);
    expect(body.items[0].metadata.shared).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// AINativeVectorStore — error handling
// ---------------------------------------------------------------------------

describe('AINativeVectorStore error handling', () => {
  test('API errors are thrown with status code', async () => {
    pushMock(403, { error: 'Forbidden' });

    const store = new AINativeVectorStore(new FakeEmbeddings(), {
      apiKey: 'bad-key',
      projectId: 'p',
    });

    await expect(store.similaritySearch('test')).rejects.toThrow(
      'ZeroDB API error 403'
    );
  });

  test('headers include API key', async () => {
    pushMock(200, { results: [] });

    const store = new AINativeVectorStore(new FakeEmbeddings(), {
      apiKey: 'my-secret-key',
      projectId: 'p',
    });

    await store.similaritySearch('test');

    const [, opts] = globalThis.fetch.mock.calls[0];
    expect(opts.headers['X-API-Key']).toBe('my-secret-key');
  });
});
