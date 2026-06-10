# @langchain/ainative

LangChain.js integration for AINative/ZeroDB.

## Rules

- This package has ZERO runtime dependencies (peer-depends on @langchain/openai)
- ES module (index.js) and CommonJS (index.cjs) entry points
- AINativeChatModel and AINativeEmbeddings extend ChatOpenAI and OpenAIEmbeddings
- AINativeVectorStore implements the LangChain VectorStore interface directly
- Auto-provisioning uses POST /api/v1/public/instant-db
- ZeroDB vector API: /api/v1/public/zerodb/{project_id}/vectors
- Never store credentials in code or tests
- Tests use mocked fetch — no real API calls in CI
