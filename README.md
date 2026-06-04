# Build an AI Agent with Genkit — from zero

A hands-on workshop. You'll build a Genkit agent step by step, growing a single
file (`src/index.ts`) from a one-line flow into a stateful chat agent with tools,
web grounding, and RAG. Copy each block, run it, see the result.

**You'll need:** Node.js 20+, a terminal, and a Gemini API key
(free at <https://aistudio.google.com> → *Get API key*).

---

## Step 0 — Project setup

Create an empty project and install dependencies.

```bash
mkdir genkit-workshop && cd genkit-workshop
npm init -y
npm pkg set type=module
npm install -D typescript tsx
npx tsc --init
npm install genkit @genkit-ai/google-genai
npm install -g genkit-cli
mkdir src
```

Set your API key (this terminal session only):

```bash
export GEMINI_API_KEY=AIza...your_key
```

**Check:** `genkit --version` prints a version number.

> **Why:** Genkit is a normal Node project — the library is a dependency, the
> `genkit` CLI is a separate global tool that runs a local Dev UI over your code.

---

## Step 1 — Your first flow

Create `src/index.ts`:

```typescript
import { genkit, z } from 'genkit';
import { googleAI } from '@genkit-ai/google-genai';

const ai = genkit({
  plugins: [googleAI()],
  model: googleAI.model('gemini-3.5-flash'),
});

export const cityGuide = ai.defineFlow(
  {
    name: 'cityGuide',
    inputSchema: z.object({ query: z.string() }),
    outputSchema: z.object({ tips: z.string() }),
  },
  async ({ query }) => {
    const { text } = await ai.generate(
      `You are a travel assistant. Answer concisely.\n\nUser: ${query}`,
    );
    return { tips: text };
  },
);
```

Start the Dev UI:

```bash
genkit start -- npx tsx --watch src/index.ts
```

**Check:** browser opens `http://localhost:4000`. Click **Flows → cityGuide**,
enter `{"query": "What to do in Berlin?"}`, press **Run**. You get an answer.
Open the **Trace** tab — you can see latency and token counts.

> **Why:** A *flow* is a typed function (zod in/out) that the Dev UI can run and
> trace. `--watch` reloads on every save — keep the Dev UI open from now on.

---

## Step 2 — Give the model a tool

The model can call *your* code. Add a tool **above** `cityGuide`:

```typescript
const getWeather = ai.defineTool(
  {
    name: 'getWeather',
    description: 'Get current weather for a city',
    inputSchema: z.object({ city: z.string() }),
    outputSchema: z.string(),
  },
  async ({ city }) => `${city}: 18°C, sunny`,
);
```

Then add a new flow that uses it (below `cityGuide`):

```typescript
export const cityGuideTool = ai.defineFlow(
  {
    name: 'cityGuideTool',
    inputSchema: z.object({ query: z.string() }),
    outputSchema: z.object({ tips: z.string() }),
  },
  async ({ query }) => {
    const { text } = await ai.generate({
      prompt: `You are a helpful travel assistant. Answer directly using your own knowledge. You also have tools — call them only when they help, otherwise just answer.\n\nUser: ${query}`,
      tools: [getWeather],
    });
    return { tips: text };
  },
);
```

**Check:** run `cityGuideTool` with `{"query": "What's the weather in Berlin?"}`.
In the Trace you'll see the model **call getWeather**, get your result, and use it.

> **Why:** Genkit gives the model only the tool's *description*. When it decides
> to call, the request comes back to **your machine**, Genkit runs your code, and
> returns the result. This is a *client-side tool* — it runs where your code runs.

---

## Step 3 — Add Google Search grounding

`googleSearch` is a **built-in** tool that runs inside Gemini. Add a new flow:

```typescript
export const cityGuideLive = ai.defineFlow(
  {
    name: 'cityGuideLive',
    inputSchema: z.object({ query: z.string() }),
    outputSchema: z.object({ tips: z.string() }),
  },
  async ({ query }) => {
    const { text } = await ai.generate({
      prompt: `You are a helpful travel assistant. Answer directly using your own knowledge. You also have tools — call them only when they help, otherwise just answer.\n\nUser: ${query}`,
      tools: [getWeather],                              // your tool
      config: {
        tools: [{ googleSearch: {} }],                  // built-in grounding tool
        toolConfig: { includeServerSideToolInvocations: true },
      },
    });
    return { tips: text };
  },
);
```

**Check:** run with `{"query": "What's the weather in Berlin and any events there this week?"}`.
The model uses `getWeather` for weather and `googleSearch` for events — both in one answer.

> **Why:** Two slots, two execution places. `tools: [getWeather]` runs **your**
> code. `config.tools: [{ googleSearch: {} }]` is a *server-side* tool that runs
> **inside Gemini at Google** — you can't put your function there, and you can't
> run Google Search locally. `includeServerSideToolInvocations` makes the
> server-side calls visible in the Trace. (Requires Gemini 3.x — older models
> can't combine your tools with built-in ones.)

---

## Step 4 — RAG: search YOUR documents

Now the agent searches a private knowledge base. **Three changes:**

**4a.** Update imports at the top of the file:

```typescript
import {
  devLocalVectorstore,
  devLocalIndexerRef,
  devLocalRetrieverRef,
} from '@genkit-ai/dev-local-vectorstore';
import { Document } from 'genkit/retriever';
```

**4b.** Install the plugin and add it to the config:

```bash
npm install @genkit-ai/dev-local-vectorstore
```

```typescript
const ai = genkit({
  plugins: [
    googleAI(),
    devLocalVectorstore([
      { indexName: 'knowledge', embedder: googleAI.embedder('gemini-embedding-001') },
    ]),
  ],
  model: googleAI.model('gemini-3.5-flash'),
});
```

**4c.** Add the indexer, a seed flow, a RAG tool, and the agent (at the bottom):

```typescript
const indexer = devLocalIndexerRef('knowledge');
const retriever = devLocalRetrieverRef('knowledge');

// Run this ONCE to fill the vector store.
export const seedDocs = ai.defineFlow(
  { name: 'seedDocs', inputSchema: z.object({}), outputSchema: z.object({ indexed: z.number() }) },
  async () => {
    const data = [
      'AgentCamp Oslo 2026 is a free, community-driven AI learning event on June 2, 2026, hosted at Microsoft in Oslo.',
      'AgentCamp Oslo 2026 has two tracks: one session track and one workshop track.',
      'AgentCamp Oslo 2026 is part of the Global AI Community. The call for speakers closed on April 30, 2026.',
      'At AgentCamp Oslo 2026, Sasha Denisov gives a talk about Genkit — building AI agents with Google\'s Genkit framework.',
    ];
    const documents = data.map((text) => Document.fromText(text));
    await ai.index({ indexer, documents });
    return { indexed: documents.length };
  },
);

// RAG as a tool — the model decides when to search your knowledge base.
const knowledgeTool = ai.defineTool(
  {
    name: 'knowledgeTool',
    description: 'Search the private knowledge base about AgentCamp Oslo 2026.',
    inputSchema: z.object({ query: z.string() }),
    outputSchema: z.array(z.string()),
  },
  async ({ query }) => {
    const docs = await ai.retrieve({ retriever, query, options: { k: 3 } });
    return docs.map((d) => d.text);
  },
);

export const assistant = ai.defineFlow(
  {
    name: 'assistant',
    inputSchema: z.object({ query: z.string() }),
    outputSchema: z.object({ answer: z.string() }),
  },
  async ({ query }) => {
    const { text } = await ai.generate({
      prompt: `You are a helpful assistant. Answer directly using your own knowledge. You also have tools — weather, web search, and a knowledge base about AgentCamp Oslo. Use each tool at most once, only when it helps. Give a final answer; do not repeat tool calls.\n\nUser: ${query}`,
      tools: [getWeather, knowledgeTool],
      maxTurns: 10,
      config: {
        tools: [{ googleSearch: {} }],
        toolConfig: { includeServerSideToolInvocations: true },
      },
    });
    return { answer: text };
  },
);
```

**Check:**
1. Run `seedDocs` with `{}` → returns `{"indexed": 4}`.
2. Run `assistant` with `{"query": "When is AgentCamp Oslo and what is Sasha talking about?"}`
   → answers from your documents (the model can't know this otherwise).

> **Why:** The embedder is attached to the index name `knowledge` *once*, in the
> config. `indexerRef`/`retrieverRef` reference it by that name — so indexing and
> search always use the same embedder (different embedders = broken search). The
> agent now has three knowledge sources — your code, Google, your documents — and
> it routes each question itself. `maxTurns` caps the tool-call loop.

---

## Step 5 — Stateful chat: add memory

Same agent, but it remembers the conversation.

**5a.** Add the import at the top:

```typescript
import type { MessageData } from 'genkit/beta';
```

**5b.** Add the chat flow at the bottom:

```typescript
const SYSTEM_PROMPT =
  'You are a helpful assistant. Answer directly using your own knowledge. You also have tools — weather, web search, and a knowledge base about AgentCamp Oslo. Use each tool at most once per turn, only when it helps. Give a final answer; do not repeat tool calls.';

const chatHistory: Record<string, MessageData[]> = {};

export const chatAssistant = ai.defineFlow(
  {
    name: 'chatAssistant',
    inputSchema: z.object({ sessionId: z.string(), message: z.string() }),
    outputSchema: z.object({ answer: z.string() }),
  },
  async ({ sessionId, message }) => {
    // Gemini requires the system message to be first; seed it once per session.
    const history = chatHistory[sessionId] ?? [
      { role: 'system', content: [{ text: SYSTEM_PROMPT }] },
    ];
    history.push({ role: 'user', content: [{ text: message }] });

    const response = await ai.generate({
      messages: history,
      tools: [getWeather, knowledgeTool],
      maxTurns: 10,
      config: {
        tools: [{ googleSearch: {} }],
        toolConfig: { includeServerSideToolInvocations: true },
      },
    });

    chatHistory[sessionId] = response.messages;
    return { answer: response.text };
  },
);
```

**Check:** run `chatAssistant` twice with the **same** `sessionId`:
1. `{"sessionId": "demo", "message": "What's the weather in Berlin?"}`
2. `{"sessionId": "demo", "message": "And what events are happening there?"}`

The second message says "there" — the model knows it means Berlin, from memory.

> **Why:** Instead of `prompt:` (one shot) we send `messages:` — the whole
> conversation. The model "remembers" because we pass it the full history every
> turn. The system message lives as the *first* entry in the history (Gemini
> requires it there), seeded once per session.

---

## Step 6 — Deploy to production (Firebase Cloud Functions)

So far everything ran locally. Now ship a flow as a real HTTPS endpoint with
`onCallGenkit`, which wraps a flow into a callable Cloud Function.

> **Prerequisites:** a Firebase project on the **Blaze** (pay-as-you-go) plan —
> Cloud Functions require billing. Free Spark plan won't deploy.

**6a.** Install the Firebase CLI and log in:

```bash
npm install -g firebase-tools
firebase login
```

**6b.** Initialize Functions in a new folder (keep your workshop project intact):

```bash
mkdir genkit-deploy && cd genkit-deploy
firebase init functions
```

Choose: **TypeScript**, your Firebase project, install dependencies = **yes**.
This creates a `functions/` folder with its own `package.json`.

**6c.** Add Genkit to the functions package:

```bash
cd functions
npm install genkit @genkit-ai/google-genai
```

**6d.** Store your API key as a secret (not an env var — Cloud Functions can't
see your terminal):

```bash
firebase functions:secrets:set GEMINI_API_KEY
```

Paste your `AIza...` key when prompted.

**6e.** Replace `functions/src/index.ts` with a deployable flow:

```typescript
import { genkit, z } from 'genkit';
import { googleAI } from '@genkit-ai/google-genai';
import { onCallGenkit } from 'firebase-functions/https';
import { defineSecret } from 'firebase-functions/params';

const apiKey = defineSecret('GEMINI_API_KEY');

const ai = genkit({
  plugins: [googleAI()],
  model: googleAI.model('gemini-3.5-flash'),
});

const cityGuideFlow = ai.defineFlow(
  {
    name: 'cityGuide',
    inputSchema: z.object({ query: z.string() }),
    outputSchema: z.object({ tips: z.string() }),
  },
  async ({ query }) => {
    const { text } = await ai.generate(
      `You are a travel assistant. Answer concisely.\n\nUser: ${query}`,
    );
    return { tips: text };
  },
);

// Wrap the flow as a callable Cloud Function. The secret is injected at runtime.
// NOTE: no auth here — see genkit.dev/docs/js/auth before going public.
export const cityGuide = onCallGenkit({ secrets: [apiKey] }, cityGuideFlow);
```

**6f.** Deploy:

```bash
firebase deploy --only functions
```

**Check:** the CLI prints a Function URL like
`https://us-central1-<project>.cloudfunctions.net/cityGuide`. Your flow now runs
in the cloud — call it from any app via the Firebase callable SDK.

> **Why:** `onCallGenkit` turns a flow into an HTTPS callable function with built-in
> support for streaming, auth policies, and App Check. The key moves from a local
> `export` to `defineSecret` → Cloud Secret Manager, because the deployed function
> has no access to your shell. Same flow code, production execution context.

> **Note — why deploy `cityGuide` and not the full RAG agent?** Our RAG used an
> *in-memory* vector store (`devLocalVectorstore`). That's perfect for local dev,
> but in the cloud every function instance starts empty and cold — the index
> wouldn't persist. For production RAG you'd swap it for a persistent vector store
> (Firestore vector search, Pinecone, etc.). So we deploy the simplest flow to show
> the `onCallGenkit` mechanics without that infrastructure detour.

---

## Step 7 — Deploy anywhere (Express + Docker)

Firebase is one option. But a Genkit flow is just Node — wrap it in an Express
server with `startFlowServer`, put it in a container, and ship it to **any** cloud
(Cloud Run, AWS, Render, Fly.io, a plain VM).

**7a.** In a fresh folder, set up a Node project and add Genkit + the Express plugin:

```bash
mkdir genkit-server && cd genkit-server
npm init -y
npm pkg set type=module
npm install -D typescript
npm install genkit @genkit-ai/google-genai @genkit-ai/express
npx tsc --init
mkdir src
```

**7b.** Create `src/index.ts` — same flow, but served over HTTP:

```typescript
import { genkit, z } from 'genkit';
import { googleAI } from '@genkit-ai/google-genai';
import { startFlowServer } from '@genkit-ai/express';

const ai = genkit({
  plugins: [googleAI()],
  model: googleAI.model('gemini-3.5-flash'),
});

const cityGuide = ai.defineFlow(
  {
    name: 'cityGuide',
    inputSchema: z.object({ query: z.string() }),
    outputSchema: z.object({ tips: z.string() }),
  },
  async ({ query }) => {
    const { text } = await ai.generate(
      `You are a travel assistant. Answer concisely.\n\nUser: ${query}`,
    );
    return { tips: text };
  },
);

// Serve the flow as an HTTP endpoint. Reads PORT from the environment.
startFlowServer({ flows: [cityGuide] });
```

**7c.** Add `build` and `start` scripts:

```bash
npm pkg set scripts.build="tsc"
npm pkg set scripts.start="node lib/index.js"
```

Make sure `tsconfig.json` outputs to `lib/` (`"outDir": "lib"`).

**7d.** Run it locally to confirm:

```bash
export GEMINI_API_KEY=AIza...your_key
npm run build && npm start
```

**Check:** the server prints a port (default 3400). Call it:

```bash
curl -X POST http://localhost:3400/cityGuide \
  -H "Content-Type: application/json" \
  -d '{"data": {"query": "What to do in Oslo?"}}'
```

You get a JSON response. Your flow is now a plain HTTP API.

**7e.** Add a `Dockerfile`:

```dockerfile
FROM node:20-slim
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build
ENV PORT=8080
EXPOSE 8080
CMD ["npm", "start"]
```

**7f.** Build the image and deploy to any container host. Example — Google Cloud Run:

```bash
gcloud run deploy genkit-server --source . \
  --update-secrets=GEMINI_API_KEY=<your-secret-name>:latest \
  --allow-unauthenticated
```

…or push the image to any registry and run it on AWS, Render, Fly.io, or your own
server — the container is the same everywhere.

**Check:** the platform gives you a public URL. `curl` it the same way as 7d.

> **Why:** `startFlowServer` turns your flows into a standard Express app — no
> vendor lock-in. The container is portable: any host that runs Docker runs your
> agent. The key comes from an environment variable (`GEMINI_API_KEY`), injected
> by the platform (Cloud Run secret, Docker `-e`, etc.). This is the difference
> from Step 6: Firebase is the managed, batteries-included path; this is the
> bring-your-own-cloud path. Same flow, your choice of infrastructure.

---

## You built

| Flow | Capability |
|------|------------|
| `cityGuide` | a typed flow |
| `cityGuideTool` | + your code as a tool |
| `cityGuideLive` | + live Google grounding |
| `assistant` | + RAG over your own documents |
| `chatAssistant` | + memory across messages |
| `cityGuide` (Firebase) | + live on Firebase Cloud Functions |
| `cityGuide` (container) | + portable to any cloud via Docker |

One agent. Three knowledge sources. It routes every question itself, remembers
the conversation — and ships to the cloud, managed or bring-your-own. All with
full tracing, no frontend.
