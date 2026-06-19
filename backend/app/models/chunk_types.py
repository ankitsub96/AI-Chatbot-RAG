from typing import TypedDict, Literal


class ChunkDict(TypedDict):
    id: str
    document_id: str
    page: int | None
    section: str | None
    chunk_type: str | None
    parent_id: str | None
    child_id: str | None
    chunk_index: int | None
    text: str
    chunk_metadata: dict | None


class HybridResultDict(ChunkDict):
    hybrid_score: float


class GroupedParentDict(TypedDict):
    parent_id: str
    score: float
    chunks: list[ChunkDict]


StrictnessLevel = Literal["strict", "balanced", "creative"]


class ToolResult(TypedDict):
    name: str
    success: bool
    summary: str
    top_chunks: list[ChunkDict]
    full_context: str
    data: dict
    truncated_result: str
    ts: float


class DocumentSearchResult(ToolResult):
    name: Literal["document_search"]


class PageLookupResult(ToolResult):
    name: Literal["page_lookup"]


class WebSearchResult(ToolResult):
    name: Literal["web_search"]


class QueryExpanderResult(ToolResult):
    name: Literal["query_expander"]


class QuestionDecomposerResult(ToolResult):
    name: Literal["question_decomposer"]


class AnswerGeneratorResult(ToolResult):
    name: Literal["answer_generator"]


class AnswerSynthesizerResult(ToolResult):
    name: Literal["answer_synthesizer"]


class AnswerEvaluatorResult(ToolResult):
    name: Literal["answer_evaluator"]


class MemorySearchResult(ToolResult):
    name: Literal["memory_search"]


AnyToolResult = (
    DocumentSearchResult
    | PageLookupResult
    | WebSearchResult
    | QueryExpanderResult
    | QuestionDecomposerResult
    | AnswerGeneratorResult
    | AnswerSynthesizerResult
    | AnswerEvaluatorResult
    | MemorySearchResult
)


class NodeOutput(TypedDict, total=False):
    session_id: str
    question: str
    document_ids: list[str]
    use_web: bool
    strictness: StrictnessLevel
    memory_context: str
    memory_messages: list[dict]
    query_embedding: list[float]
    tool_history: list[dict]
    collected_context: str
    full_answer: str
    eval_result: dict | None
    iteration: int
    trace: list[dict]
    last_thought: dict
    cache_hit: bool
    cache_answer: str | None


class CacheCheckOutput(TypedDict, total=False):
    cache_hit: bool
    cache_answer: str | None
    query_embedding: list[float]
    last_thought: dict
    trace: list[dict]


class CacheHitOutput(TypedDict, total=False):
    last_thought: dict
    trace: list[dict]


class MemoryMessage(TypedDict):
    type: Literal["human", "ai"]
    content: str


class MemoryRetrievalOutput(TypedDict, total=False):
    memory_context: str
    memory_messages: list[MemoryMessage]
    last_thought: dict
    trace: list[dict]


class SaveAndReturnOutput(TypedDict, total=False):
    last_thought: dict
    trace: list[dict]
