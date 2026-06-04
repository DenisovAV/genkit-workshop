import { genkit, z } from 'genkit';
import { googleAI } from '@genkit-ai/google-genai';
import {
  devLocalVectorstore,
  devLocalIndexerRef,
  devLocalRetrieverRef,
} from '@genkit-ai/dev-local-vectorstore';
import { Document } from 'genkit/retriever';
import type { MessageData } from 'genkit/beta';

const ai = genkit({
  plugins: [
    googleAI(),
    devLocalVectorstore([
      { indexName: 'knowledge', embedder: googleAI.embedder('gemini-embedding-001') },
    ]),
  ],
  model: googleAI.model('gemini-3.5-flash'),
});

// === Step 1: base flow — a simple travel assistant ===
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

// === Step 2: tool calling — getWeather is YOUR code, called by the model ===
const getWeather = ai.defineTool(
  { name: 'getWeather', description: 'Get current weather for a city',
    inputSchema: z.object({ city: z.string() }), outputSchema: z.string() },
  async ({ city }) => `${city}: 18°C, sunny`,
);

export const cityGuideTool = ai.defineFlow(
  {
    name: 'cityGuideTool',
    inputSchema: z.object({ query: z.string() }),
    outputSchema: z.object({ tips: z.string() }),
  },
  async ({ query }) => {
    const { text } = await ai.generate({
      prompt: `You are a helpful travel assistant. Answer directly using your own knowledge. You also have tools (e.g. weather, web search) — call them only when they help, otherwise just answer.\n\nUser: ${query}`,
      tools: [getWeather],
    });
    return { tips: text };
  },
);

// === Step 3: grounding — your tool + built-in googleSearch (runs in Gemini) ===
export const cityGuideLive = ai.defineFlow(
  {
    name: 'cityGuideLive',
    inputSchema: z.object({ query: z.string() }),
    outputSchema: z.object({ tips: z.string() }),
  },
  async ({ query }) => {
    const { text } = await ai.generate({
      prompt: `You are a helpful travel assistant. Answer directly using your own knowledge. You also have tools (e.g. weather, web search) — call them only when they help, otherwise just answer.\n\nUser: ${query}`,
      tools: [getWeather],
      config: {
        tools: [{ googleSearch: {} }],
        toolConfig: { includeServerSideToolInvocations: true },
      },
    });
    return { tips: text };
  },
);

// === Step 4: Agentic RAG — run seedDocs ONCE, then assistant ===
// Three knowledge sources: your code + Google + your private documents.
const indexer = devLocalIndexerRef('knowledge');
const retriever = devLocalRetrieverRef('knowledge');

export const seedDocs = ai.defineFlow(
  { name: 'seedDocs', inputSchema: z.object({}), outputSchema: z.object({ indexed: z.number() }) },
  async () => {
    const data = [
      'AgentCamp Oslo 2026 is a free, community-driven AI learning event taking place on June 2, 2026, hosted at Microsoft in Oslo.',
      'AgentCamp Oslo 2026 has two tracks: one session track and one workshop track, focused on hands-on AI agent development.',
      'AgentCamp Oslo 2026 is part of the Global AI Community. The call for speakers closed on April 30, 2026.',
      'At AgentCamp Oslo 2026, Sasha Denisov gives a talk about Genkit — building AI agents with Google\'s Genkit framework.',
    ];
    const documents = data.map((text) => Document.fromText(text));
    await ai.index({ indexer, documents });
    return { indexed: documents.length };
  },
);

const knowledgeTool = ai.defineTool(
  {
    name: 'knowledgeTool',
    description: 'Search the private knowledge base about AgentCamp Oslo 2026 and its talks.',
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
      prompt: `You are a helpful assistant. Answer directly using your own knowledge. You also have tools — weather, web search, and a knowledge base about AgentCamp Oslo. Use each tool at most once, only when it helps (use the knowledge base for AgentCamp Oslo questions). After gathering what you need, give a final answer. Do not repeat tool calls.\n\nUser: ${query}`,
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

// === Step 5: stateful chat — same agent, but remembers the conversation ===
// Use the SAME sessionId across messages to keep context.
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
