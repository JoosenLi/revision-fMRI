"""Shared statistical contracts; no replacement for the audited R fits yet."""
from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence


@dataclass(frozen=True)
class StatisticalSpec:
    protocol_id: str
    formula: str
    reference_levels: Mapping[str, str]
    source_revision: str
    inference_method: str


class StatisticalAdapter(Protocol):
    """Future backend must match the reference design matrix and inference rules."""
    def compare(self, records: Sequence[Mapping[str, object]],
                specification: StatisticalSpec) -> Mapping[str, object]: ...
