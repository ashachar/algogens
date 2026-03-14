"""
Base validator interface and common validation utilities.
Each benchmark implements its own validator subclass.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class ValidationResult:
    decision: str  # accepted / rejected / clamped / transformed
    original_proposal: dict | None
    validated_proposal: dict | None
    reasons: list[str] = field(default_factory=list)
    clamped_fields: list[str] = field(default_factory=list)


class BaseValidator(ABC):
    """Base class for all benchmark validators."""

    def __init__(self, name: str, strength: str = "full"):
        """
        Args:
            name: validator name
            strength: "full", "weak", or "none"
        """
        self.name = name
        self.strength = strength
        self.log: list[dict] = []

    def validate(self, proposal: dict | None, context: dict) -> ValidationResult:
        """Validate a proposal. Returns a ValidationResult."""
        if proposal is None:
            result = ValidationResult(
                decision="rejected",
                original_proposal=None,
                validated_proposal=None,
                reasons=["proposal is None (API error or no response)"],
            )
            self._log(result, context)
            return result

        if self.strength == "none":
            result = ValidationResult(
                decision="accepted",
                original_proposal=proposal,
                validated_proposal=proposal,
                reasons=["no-validator mode: all proposals accepted"],
            )
            self._log(result, context)
            return result

        # Schema check
        schema_result = self._check_schema(proposal)
        if schema_result is not None:
            self._log(schema_result, context)
            return schema_result

        if self.strength == "weak":
            result = self._weak_validate(proposal, context)
        else:
            result = self._full_validate(proposal, context)

        self._log(result, context)
        return result

    @abstractmethod
    def _check_schema(self, proposal: dict) -> ValidationResult | None:
        """Check that proposal has correct schema. Return None if valid."""
        ...

    @abstractmethod
    def _full_validate(self, proposal: dict, context: dict) -> ValidationResult:
        """Full validation with all constraints."""
        ...

    def _weak_validate(self, proposal: dict, context: dict) -> ValidationResult:
        """Weak validation: only schema check, accept everything else."""
        return ValidationResult(
            decision="accepted",
            original_proposal=proposal,
            validated_proposal=proposal,
            reasons=["weak validator: schema valid, no further checks"],
        )

    def _log(self, result: ValidationResult, context: dict):
        self.log.append({
            "validator": self.name,
            "strength": self.strength,
            "decision": result.decision,
            "reasons": result.reasons,
            "clamped_fields": result.clamped_fields,
            "instance_id": context.get("instance_id", "unknown"),
        })

    def get_stats(self) -> dict:
        total = len(self.log)
        if total == 0:
            return {"total": 0}
        decisions = {}
        for entry in self.log:
            d = entry["decision"]
            decisions[d] = decisions.get(d, 0) + 1
        return {
            "total": total,
            "decisions": decisions,
            "acceptance_rate": decisions.get("accepted", 0) / total,
            "clamped_rate": decisions.get("clamped", 0) / total,
            "rejection_rate": decisions.get("rejected", 0) / total,
        }
