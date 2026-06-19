// Single source of truth for the 6 RAG endpoints — their backend path and
// which request-shaping / UI capabilities they support. Reused by:
//   - ragApi.js's askEndpoint()  → how to shape the request body per endpoint
//   - ModeSelector.jsx           → what to render in the endpoint/strictness picker
//   - useEndpointAccents.js      → the 6 accent slots map 1:1 to these 6 entries
//
// Deliberately a plain data file — no axios/fetch/React imports — so any of
// the above can pull it in without dragging in unrelated dependencies, and
// adding a 7th endpoint later only means touching this one file.

export const ENDPOINTS = [
  { id: 'ask',       label: 'Simple',    path: '/rag/ask',           supportsStream: false, supportsWeb: false, supportsStrictness: false },
  { id: 'langchain', label: 'LangChain', path: '/rag/ask/langchain', supportsStream: true,  supportsWeb: false, supportsStrictness: false, legacySSE: true },
  { id: 'agent',     label: 'Agent',     path: '/rag/ask/agent',     supportsStream: true,  supportsWeb: false, supportsStrictness: false },
  { id: 'react',     label: 'ReAct',     path: '/rag/react/ask',     supportsStream: true,  supportsWeb: true,  supportsStrictness: true },
  { id: 'planner',   label: 'Planner',   path: '/rag/planner/ask',   supportsStream: true,  supportsWeb: true,  supportsStrictness: true },
  { id: 'research',  label: 'Research',  path: '/rag/research/ask', supportsStream: true,  supportsWeb: true,  supportsStrictness: true },
]

export const STRICTNESS_LEVELS = ['strict', 'balanced', 'creative']

// The app today always calls /rag/ask/langchain (streaming or not — see the
// current ragApi.js). Defaulting useResearchMode to 'langchain' preserves
// that behavior so shipping this doesn't silently switch anyone's default
// experience to a different endpoint. Change this if you'd rather default
// to something else (e.g. 'ask' for the simplest/cheapest call, or 'agent'
// to lead with the new thinking UI).
export const DEFAULT_ENDPOINT_ID = 'langchain'
export const DEFAULT_STRICTNESS = 'balanced'

export function getEndpointConfig(endpointId) {
  return ENDPOINTS.find(e => e.id === endpointId) || null
}

export function isValidStrictness(value) {
  return STRICTNESS_LEVELS.includes(value)
}
