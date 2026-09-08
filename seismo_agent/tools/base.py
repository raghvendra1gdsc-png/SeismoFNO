"""
seismo_agent/tools/base.py — Abstract Base Tool Interface.

Defines the contract for all deterministic engineering tools. Compatible with
standard OpenAI / NVIDIA Nemotron function calling specifications.
"""

from abc import ABC, abstractmethod
from typing import Type, Dict, Any, Union, Optional
from pydantic import BaseModel, ValidationError


class ToolExecutionError(Exception):
    """Raised when a deterministic engineering tool encounters an unrecoverable execution failure."""
    def __init__(self, tool_name: str, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(f"[{tool_name}] {message}")
        self.tool_name = tool_name
        self.message = message
        self.details = details or {}


class BaseTool(ABC):
    """
    Abstract base class for all deterministic engineering tools in SeismoAgent.

    Attributes:
        name: Unique string identifier for tool invocation.
        description: Comprehensive description for LLM reasoning and selection.
        input_schema: Pydantic model defining valid input parameters.
        output_schema: Pydantic model defining structured result payload.
    """

    name: str
    description: str
    input_schema: Type[BaseModel]
    output_schema: Type[BaseModel]

    @abstractmethod
    def _run(self, request: BaseModel) -> BaseModel:
        """Internal execution logic receiving a validated input model."""
        pass

    def execute(self, request: Union[BaseModel, Dict[str, Any]]) -> BaseModel:
        """
        Public execution entry point with automatic input validation.

        Args:
            request: Either an instance of `input_schema` or a dictionary of arguments.

        Returns:
            Validated instance of `output_schema`.

        Raises:
            ValidationError: If input parameters violate schema constraints.
            ToolExecutionError: If tool execution encounters a physical or numerical error.
        """
        if isinstance(request, dict):
            try:
                validated_req = self.input_schema.model_validate(request)
            except ValidationError as e:
                raise ValidationError.from_exception_data(
                    title=f"Validation failed for tool '{self.name}'",
                    line_errors=e.errors()
                ) from e
        elif isinstance(request, self.input_schema):
            validated_req = request
        else:
            raise TypeError(
                f"Tool '{self.name}' expected input of type {self.input_schema.__name__} or dict, got {type(request).__name__}"
            )

        try:
            result = self._run(validated_req)
            if not isinstance(result, self.output_schema):
                result = self.output_schema.model_validate(result)
            return result
        except Exception as e:
            if isinstance(e, (ValidationError, ToolExecutionError)):
                raise
            raise ToolExecutionError(
                tool_name=self.name,
                message=str(e),
                details={"input_request": validated_req.model_dump()}
            ) from e

    def to_tool_spec(self) -> Dict[str, Any]:
        """
        Generate OpenAI / NVIDIA Nemotron compatible function tool specification.
        """
        schema = self.input_schema.model_json_schema()
        # Clean schema title/description to avoid redundancy
        schema.pop("title", None)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description.strip(),
                "parameters": schema,
            }
        }
