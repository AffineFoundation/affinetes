from pydantic import BaseModel
from uuid import UUID
from typing import List, Optional, Any
from enum import Enum

# Inference
class InferenceMessage(BaseModel):
    role: str
    content: str

class InferenceToolMode(Enum):
    NONE = "none"
    AUTO = "auto"
    REQUIRED = "required"

class InferenceToolParameterType(Enum):
    BOOLEAN = "boolean"
    INTEGER = "integer"
    NUMBER = "number"
    STRING = "string"
    ARRAY = "array"
    OBJECT = "object"
    
class InferenceToolParameter(BaseModel):
    type: InferenceToolParameterType
    name: str
    description: str
    required: bool = False

class InferenceTool(BaseModel):
    name: str
    description: str
    parameters: List[InferenceToolParameter]

class InferenceRequest(BaseModel):
    evaluation_run_id: UUID
    model: str
    temperature: float
    messages: List[InferenceMessage]
    tool_mode: Optional[InferenceToolMode] = InferenceToolMode.NONE
    tools: Optional[List[InferenceTool]] = None

class InferenceToolCallArgument(BaseModel):
    name: str
    value: Any
    
class InferenceToolCall(BaseModel):
    name: str
    arguments: List[InferenceToolCallArgument]

class InferenceResponse(BaseModel):
    content: str
    tool_calls: List[InferenceToolCall]