# @langchain/ainative

LangChain.js integration for [AINative](https://ainative.studio) and [ZeroDB](https://zerodb.ai). Chat models, embeddings, and vector store — all backed by AINative's free-tier AI infrastructure.

**Zero config.** Auto-provisions a free project on first use. No signup, no credit card.

## Install

```bash
npm install @langchain/ainative @langchain/openai
```

## Quick Start

### Chat Model

```javascript
import { AINativeChatModel } from '@langchain/ainative';

const model = new AINativeChatModel();
// Uses meta-llama/Llama-3.3-70B-Instruct by default (free tier)

const response = await model.invoke('What is retrieval-augmented generation?');
console.log(response.content);
```

### Embeddings

```javascript
import { AINativeEmbeddings } from '@langchain/ainative';

const embeddings = new AINativeEmbeddings();
// Uses BAAI/bge-small-en-v1.5 by default (free tier)

const vectors = await embeddings.embedDocuments([
  'Hello world',
  'LangChain is great',
]);
console.log(vectors[0].length); // 384
```

### Vector Store (RAG)

```javascript
import { AINativeChatModel, AINativeEmbeddings, AINativeVectorStore } from '@langchain/ainative';

// Create embeddings + vector store
const embeddings = new AINativeEmbeddings();
const vectorStore = new AINativeVectorStore(embeddings, {
  // Optional: auto-provisions if omitted
  apiKey: process.env.ZERODB_API_KEY,
  projectId: process.env.ZERODB_PROJECT_ID,
});

// Add documents
await vectorStore.addDocuments([
  { pageContent: 'AINative provides free LLM access.', metadata: { source: 'docs' } },
  { pageContent: 'ZeroDB is a vector database for AI agents.', metadata: { source: 'docs' } },
]);

// Similarity search
const results = await vectorStore.similaritySearch('free AI tools', 3);
console.log(results);
```

### Full RAG Chain

```javascript
import { AINativeChatModel, AINativeEmbeddings, AINativeVectorStore } from '@langchain/ainative';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { RunnableSequence } from '@langchain/core/runnables';

const model = new AINativeChatModel();
const embeddings = new AINativeEmbeddings();
const vectorStore = new AINativeVectorStore(embeddings);

// Build RAG chain
const retriever = {
  invoke: async (query) => {
    const docs = await vectorStore.similaritySearch(query, 3);
    return docs.map((d) => d.pageContent).join('\n');
  },
};

const prompt = ChatPromptTemplate.fromTemplate(
  'Answer based on context:\n{context}\n\nQuestion: {question}'
);

const chain = RunnableSequence.from([
  {
    context: (input) => retriever.invoke(input.question),
    question: (input) => input.question,
  },
  prompt,
  model,
  new StringOutputParser(),
]);

const answer = await chain.invoke({ question: 'What is ZeroDB?' });
console.log(answer);
```

## Configuration

### Chat Model Options

| Option | Default | Description |
|--------|---------|-------------|
| `model` | `meta-llama/Llama-3.3-70B-Instruct` | Model name |
| `apiKey` | `process.env.AINATIVE_API_KEY` | AINative API key |
| `baseURL` | `https://api.ainative.studio/api/v1` | API base URL |
| `temperature` | `0.7` | Sampling temperature |
| `maxTokens` | - | Max tokens in response |

### Embeddings Options

| Option | Default | Description |
|--------|---------|-------------|
| `model` | `BAAI/bge-small-en-v1.5` | Embedding model |
| `apiKey` | `process.env.AINATIVE_API_KEY` | AINative API key |
| `baseURL` | `https://api.ainative.studio/api/v1` | API base URL |

### Vector Store Options

| Option | Default | Description |
|--------|---------|-------------|
| `apiKey` | `process.env.ZERODB_API_KEY` | ZeroDB API key |
| `projectId` | `process.env.ZERODB_PROJECT_ID` | ZeroDB project ID |
| `namespace` | `default` | Vector namespace |
| `baseURL` | `https://api.ainative.studio` | API base URL |

## Auto-Provisioning

When no API key is provided, the package automatically creates a free ZeroDB project:

1. Makes a POST to `/api/v1/public/instant-db`
2. Returns credentials (API key + project ID)
3. Project has a 72-hour trial — claim it at the provided URL to keep permanently

Set `ZERODB_API_KEY` and `ZERODB_PROJECT_ID` environment variables to skip provisioning.

## Environment Variables

| Variable | Description |
|----------|-------------|
| `AINATIVE_API_KEY` | API key for chat/embeddings |
| `ZERODB_API_KEY` | API key for vector store |
| `ZERODB_PROJECT_ID` | ZeroDB project ID |

## CommonJS Support

```javascript
const { AINativeChatModel, AINativeEmbeddings, AINativeVectorStore } = require('@langchain/ainative');
```

## License

MIT

---

## Zero-Config Auto-Provisioning

**No account needed to start.** On first run, this package automatically:

1. Provisions a **free ZeroDB instance** (~800ms)
2. Saves credentials locally (`.env`, `.mcp.json`)
3. Prints a **claim link** in your terminal

### The 72-Hour Window

Your auto-provisioned instance is **free for 72 hours**. To keep your data permanently, visit the **claim URL** printed in your terminal.

### For AI Agents

If a coding agent (Claude Code, Cursor, Windsurf) installs this package:
- Auto-provisions without human intervention
- The **claim URL** appears in agent output — visit it to take ownership
- `.claude/CLAUDE.md` instructs agents to surface the claim link to you
- Check `.env` or `~/.ainative/config.json` for your project ID if you miss it
