"""Convert Python function signatures to OpenAI function-tool JSON schema."""

from __future__ import annotations

import inspect
import typing
from typing import Any, Callable, get_args, get_origin


def _python_type_to_json_schema(annotation: Any) -> dict:
    """Map a Python type annotation to a JSON Schema type descriptor."""
    if annotation is inspect.Parameter.empty:
        return {"type": "string"}

    origin = get_origin(annotation)
    args = get_args(annotation)

    # list[X] or List[X]
    if origin is list:
        schema: dict = {"type": "array"}
        if args:
            schema["items"] = _python_type_to_json_schema(args[0])
        return schema

    # Union types: X | Y or Optional[X] (= Union[X, None])
    if origin is typing.Union:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            # Optional[X] — just use X's schema
            return _python_type_to_json_schema(non_none[0])
        # True union (e.g. str | list[str]) — use anyOf
        return {"anyOf": [_python_type_to_json_schema(a) for a in non_none]}

    # typing.Literal["a", "b", "c"]
    if origin is typing.Literal:
        values = list(args)
        if all(isinstance(v, str) for v in values):
            return {"type": "string", "enum": values}
        if all(isinstance(v, int) for v in values):
            return {"type": "integer", "enum": values}
        return {"enum": values}

    # Primitive types
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
    }
    json_type = type_map.get(annotation)
    if json_type:
        return {"type": json_type}

    return {"type": "string"}


def function_to_tool_schema(func: Callable) -> dict:
    """Convert a Python function to an OpenAI function-tool JSON schema.

    Extracts the function name, first line of docstring as description,
    and parameter types/defaults from the signature.

    Args:
        func: A Python function with type annotations and a docstring.

    Returns:
        A dict matching the OpenAI function tool schema format.
    """
    sig = inspect.signature(func)
    docstring = inspect.getdoc(func) or ""
    # Use everything before "Examples:" as the description
    description = docstring.split("Examples:")[0].strip()

    properties = {}
    required = []

    for name, param in sig.parameters.items():
        prop = _python_type_to_json_schema(param.annotation)

        if param.default is not inspect.Parameter.empty:
            prop["default"] = param.default
        else:
            required.append(name)

        properties[name] = prop

    schema = {
        "type": "function",
        "name": func.__name__,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
        },
    }
    if required:
        schema["parameters"]["required"] = required

    return schema
