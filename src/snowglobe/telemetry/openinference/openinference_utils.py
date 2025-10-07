"""
Borrowed heavily from openinference-instrumentation-guardrails.
Source: https://github.com/Arize-ai/openinference/blob/main/python/instrumentation/openinference-instrumentation-guardrails/src/openinference/instrumentation/guardrails/_wrap_guard_call.py#L29-L45
"""
from enum import Enum
from typing import Any, Iterator, List, Mapping, Optional, Tuple
from opentelemetry.util.types import AttributeValue


def _flatten(
    mapping: Optional[Mapping[str, Any]],
) -> Iterator[Tuple[str, AttributeValue]]:
    if not mapping:
        return
    for key, value in mapping.items():
        if value is None:
            continue
        if isinstance(value, Mapping):
            for sub_key, sub_value in _flatten(value):
                yield f"{key}.{sub_key}", sub_value
        elif isinstance(value, List) and any(
            isinstance(item, Mapping) for item in value
        ):
            for index, sub_mapping in enumerate(value):
                for sub_key, sub_value in _flatten(sub_mapping):
                    yield f"{key}.{index}.{sub_key}", sub_value
        else:
            if isinstance(value, Enum):
                value = value.value
            yield key, value
