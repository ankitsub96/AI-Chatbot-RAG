import operator
from typing import Annotated
from typing_extensions import TypedDict
from app.models.chunk_types import StrictnessLevel, MemoryMessage


# =====================================================
# REACT STATES
# =====================================================

class ReactState(TypedDict):
    session_id: str
    question: str
    document_ids: list[str]
    use_web: bool
    strictness: StrictnessLevel
    memory_context: str
    memory_messages: list[MemoryMessage]
    query_embedding: list[float]
    tool_history: Annotated[list[dict], operator.add]
    collected_context: str
    full_answer: str
    eval_result: dict | None
    iteration: int
    trace: Annotated[list[dict], operator.add]
    last_thought: dict
    cache_hit: bool
    cache_answer: str | None


# =====================================================
# PLANNER STATES
# =====================================================

class PlannerState(TypedDict):
    session_id: str
    question: str
    document_ids: list[str]
    use_web: bool
    strictness: StrictnessLevel
    memory_context: str
    memory_messages: list[MemoryMessage]
    query_embedding: list[float]
    plan: list[str]
    step_contexts: Annotated[list[str], operator.add]
    sub_answers: Annotated[list[str], operator.add]
    full_answer: str
    eval_result: dict | None
    trace: Annotated[list[dict], operator.add]
    last_thought: dict
    cache_hit: bool
    cache_answer: str | None


# =====================================================
# HYBRID STATES
# =====================================================

class StepState(TypedDict):
    sub_question: str
    document_ids: list[str]
    use_web: bool
    strictness: StrictnessLevel
    memory_context: str
    tool_history: Annotated[list[dict], operator.add]
    collected_context: str
    sub_answer: str
    iteration: int


class HybridState(TypedDict):
    session_id: str
    question: str
    document_ids: list[str]
    use_web: bool
    strictness: StrictnessLevel
    memory_context: str
    memory_messages: list[MemoryMessage]
    query_embedding: list[float]
    plan: list[str]
    step_contexts: Annotated[list[str], operator.add]
    sub_answers: Annotated[list[str], operator.add]
    full_answer: str
    eval_result: dict | None
    trace: Annotated[list[dict], operator.add]
    last_thought: dict
    cache_hit: bool
    cache_answer: str | None


# =====================================================
# REACT NODE OUTPUTS
# =====================================================

class ReactCacheCheckOutput(TypedDict, total=False):
    cache_hit: bool
    cache_answer: str | None
    query_embedding: list[float]
    last_thought: dict
    trace: list[dict]

class ReactNodeOutput(TypedDict, total=False):
    tool_history: list[dict]
    collected_context: str
    full_answer: str
    eval_result: dict | None
    iteration: int
    last_thought: dict
    trace: list[dict]


# =====================================================
# PLANNER NODE OUTPUTS
# =====================================================

class PlannerNodeOutput(TypedDict, total=False):
    plan: list[str]
    step_contexts: list[str]
    sub_answers: list[str]
    full_answer: str
    eval_result: dict | None
    last_thought: dict
    trace: list[dict]


# =====================================================
# STEP NODE OUTPUTS
# =====================================================

class StepReactOutput(TypedDict, total=False):
    tool_history: list[dict]
    last_thought: dict
    trace: list[dict]

class StepToolOutput(TypedDict, total=False):
    collected_context: str
    tool_history: list[dict]
    last_thought: dict
    trace: list[dict]

class StepAnswerOutput(TypedDict, total=False):
    sub_answer: str
    last_thought: dict


# =====================================================
# HYBRID NODE OUTPUTS
# =====================================================

class PlanCreationOutput(TypedDict, total=False):
    plan: list[str]
    last_thought: dict
    trace: list[dict]

class StepExecutionOutput(TypedDict, total=False):
    sub_answers: list[str]
    step_contexts: list[str]
    last_thought: dict
    trace: list[dict]

class SynthesisOutput(TypedDict, total=False):
    full_answer: str
    last_thought: dict
    trace: list[dict]

class EvaluationOutput(TypedDict, total=False):
    eval_result: dict
    last_thought: dict
    trace: list[dict]