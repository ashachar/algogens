"""
Proposal client: wraps the Anthropic Messages API with tool use for
generating structured proposals from Claude.

All benchmark proposal generation goes through this module.
"""
import json
import time
import os
import subprocess
from dataclasses import dataclass, field, asdict
from typing import Any

import anthropic


@dataclass
class InferenceLog:
    benchmark: str
    instance_id: str
    model_id: str
    prompt_template_version: str
    tool_schema_version: str
    raw_tool_input: dict | None
    validator_decision: str  # accepted / rejected / clamped / transformed
    latency_ms: float
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    retry_count: int
    error_class: str | None
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


# Pricing per million tokens (as of 2026-03)
MODEL_PRICING = {
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
    "claude-opus-4-6": {"input": 5.0, "output": 25.0},
    "claude-haiku-4-5": {"input": 1.0, "output": 5.0},
}


def estimate_cost(model_id: str, input_tokens: int, output_tokens: int) -> float:
    pricing = MODEL_PRICING.get(model_id, {"input": 3.0, "output": 15.0})
    return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000


def _refresh_oauth_token() -> str | None:
    """Refresh OAuth token from macOS keychain."""
    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-s", "Claude Code-credentials", "-g"],
            capture_output=True, text=True, timeout=10
        )
        for line in result.stderr.split("\n"):
            if line.startswith("password:"):
                raw = line.split("password: ", 1)[1].strip('"')
                data = json.loads(raw)
                return data["claudeAiOauth"]["accessToken"]
    except Exception:
        pass
    return None


class ProposalClient:
    def __init__(self, model_id: str = "claude-sonnet-4-6", max_retries: int = 2,
                 temperature: float = 0.0, api_key: str | None = None):
        self.model_id = model_id
        self.max_retries = max_retries
        self.temperature = temperature
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=self._api_key)
        self.logs: list[InferenceLog] = []

    def _refresh_client(self):
        """Refresh the API client with a new OAuth token."""
        new_token = _refresh_oauth_token()
        if new_token:
            self._api_key = new_token
            self.client = anthropic.Anthropic(api_key=new_token)
            return True
        return False

    def propose(
        self,
        benchmark: str,
        instance_id: str,
        system_prompt: str,
        user_prompt: str,
        tool_definition: dict,
        tool_name: str,
        prompt_version: str = "v1",
        schema_version: str = "v1",
    ) -> tuple[dict | None, InferenceLog]:
        """
        Send a proposal request to Claude using tool use.
        Returns (tool_input_dict_or_None, log).
        """
        retries = 0
        last_error = None

        while retries <= self.max_retries:
            t0 = time.time()
            try:
                response = self.client.messages.create(
                    model=self.model_id,
                    max_tokens=4096,
                    temperature=self.temperature,
                    system=system_prompt,
                    tools=[tool_definition],
                    tool_choice={"type": "tool", "name": tool_name},
                    messages=[{"role": "user", "content": user_prompt}],
                )
                latency_ms = (time.time() - t0) * 1000

                # Extract tool use block
                tool_input = None
                for block in response.content:
                    if block.type == "tool_use":
                        tool_input = block.input
                        break

                usage = response.usage
                log = InferenceLog(
                    benchmark=benchmark,
                    instance_id=instance_id,
                    model_id=response.model,
                    prompt_template_version=prompt_version,
                    tool_schema_version=schema_version,
                    raw_tool_input=tool_input,
                    validator_decision="pending",
                    latency_ms=latency_ms,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    estimated_cost_usd=estimate_cost(
                        self.model_id, usage.input_tokens, usage.output_tokens
                    ),
                    retry_count=retries,
                    error_class=None,
                    cache_creation_input_tokens=getattr(usage, 'cache_creation_input_tokens', 0) or 0,
                    cache_read_input_tokens=getattr(usage, 'cache_read_input_tokens', 0) or 0,
                )
                self.logs.append(log)
                time.sleep(1.0)  # Rate-limit guard between calls
                return tool_input, log

            except anthropic.RateLimitError as e:
                latency_ms = (time.time() - t0) * 1000
                last_error = "RateLimitError"
                retries += 1
                time.sleep(5 * (2 ** retries))  # 10s, 20s backoff
            except anthropic.APIStatusError as e:
                latency_ms = (time.time() - t0) * 1000
                last_error = f"APIStatusError_{e.status_code}"
                if e.status_code == 401:
                    # Token expired — try refreshing
                    if self._refresh_client():
                        retries += 1
                        time.sleep(1)
                    else:
                        break
                elif e.status_code >= 500:
                    retries += 1
                    time.sleep(2 ** retries)
                else:
                    break
            except anthropic.APIConnectionError:
                latency_ms = (time.time() - t0) * 1000
                last_error = "APIConnectionError"
                retries += 1
                time.sleep(2 ** retries)
            except Exception as e:
                latency_ms = (time.time() - t0) * 1000
                last_error = type(e).__name__
                break

        # All retries exhausted or non-retryable error
        log = InferenceLog(
            benchmark=benchmark,
            instance_id=instance_id,
            model_id=self.model_id,
            prompt_template_version=prompt_version,
            tool_schema_version=schema_version,
            raw_tool_input=None,
            validator_decision="error",
            latency_ms=latency_ms,
            input_tokens=0,
            output_tokens=0,
            estimated_cost_usd=0.0,
            retry_count=retries,
            error_class=last_error,
        )
        self.logs.append(log)
        return None, log

    def get_all_logs(self) -> list[dict]:
        return [asdict(log) for log in self.logs]

    def get_cost_summary(self) -> dict:
        total_cost = sum(l.estimated_cost_usd for l in self.logs)
        total_input = sum(l.input_tokens for l in self.logs)
        total_output = sum(l.output_tokens for l in self.logs)
        total_calls = len(self.logs)
        errors = sum(1 for l in self.logs if l.error_class is not None)
        return {
            "total_calls": total_calls,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_cost_usd": round(total_cost, 6),
            "error_count": errors,
            "model_id": self.model_id,
        }
