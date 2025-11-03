"""Utility helpers for NutriBench prompt optimization.

This module provides reusable building blocks for the optimization workflow:

* Dataset loading utilities
* Evaluation data structures and metrics
* Resilient LLM client wrappers for Gemini and GPT models
* Prompt critique/edit helpers inspired by ProTeGi
* Convenience functions for logging and artifact persistence
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import google.generativeai as genai
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google.api_core import exceptions as gcore_exceptions
from google.generativeai.types import RequestOptions
from openai import OpenAI
from tenacity import (  # type: ignore[import-untyped]
	Retrying,
	before_sleep_log,
	retry,
	retry_if_exception_type,
	stop_after_attempt,
	wait_exponential,
	wait_random_exponential,
)
from tqdm import tqdm


SEED = 42
DEFAULT_PROVIDER = "gemini"
DEFAULT_GEMINI_MODEL = "models/gemini-2.5-pro"
DEFAULT_GPT_MODEL = "gpt-4o-mini"
ACCURACY_THRESHOLD = 7.5
DEFAULT_GEMINI_ENDPOINT = os.getenv(
	"GEMINI_API_ENDPOINT",
	"generativelanguage.googleapis.com",
)
DEFAULT_GEMINI_TIMEOUT_SECONDS = float(os.getenv("GEMINI_REQUEST_TIMEOUT", "180"))
MAX_MEAL_DESCRIPTION_CHARS = int(os.getenv("MAX_MEAL_DESCRIPTION_CHARS", "1500"))
GEMINI_REQUESTS_PER_BATCH = int(os.getenv("GEMINI_REQUESTS_PER_BATCH", "12"))
GEMINI_BATCH_COOLDOWN_SECONDS = float(os.getenv("GEMINI_BATCH_COOLDOWN_SECONDS", "1.5"))

load_dotenv()


@dataclass(slots=True)
class SampleEvaluation:
	"""Stores evaluation details for a single NutriBench example."""

	meal_description: str
	actual_carb: float
	prediction: float
	error: float
	response_text: str
	metadata: dict[str, object] | None = None


@dataclass(slots=True)
class PromptEvaluation:
	"""Aggregate evaluation metrics for a prompt."""

	prompt: str
	provider: str
	model: str
	metrics: dict[str, float]
	samples: list[SampleEvaluation]
	iteration: int
	evaluation_size: int
	parent_id: str | None = None
	candidate_id: str | None = None
	metadata: dict[str, object] | None = None


@dataclass(slots=True)
class PromptCandidate:
	"""Container for prompt variants tracked during beam search."""

	id: str
	prompt: str
	iteration: int
	parent_id: str | None = None
	critique: str | None = None
	variant_index: int = 0
	evaluation: PromptEvaluation | None = None

	@property
	def score(self) -> float:
		if self.evaluation is None:
			return float("inf")
		return self.evaluation.metrics.get("mae", float("inf"))

class LLMError(RuntimeError):
	"""Raised when an LLM call fails permanently."""


class LLMRetryableError(LLMError):
	"""Raised when an LLM call should be retried with the same request."""


class LLMClient:
	"""Abstracts interactions with a chat or text generation model."""

	def __init__(self, provider: str, model: str, max_output_tokens: int = 1024) -> None:
		self.provider = provider
		self.model = model
		self.max_output_tokens = max_output_tokens
		self._last_metadata: dict[str, object] | None = None

	def generate(self, prompt: str, temperature: float) -> str:
		raise NotImplementedError

	def get_last_response_metadata(self) -> dict[str, object] | None:
		if self._last_metadata is None:
			return None
		return dict(self._last_metadata)

	def _set_last_metadata(self, metadata: dict[str, object] | None) -> None:
		self._last_metadata = metadata.copy() if metadata else None


class GeminiClient(LLMClient):
	"""Client wrapper around the Google Gemini SDK with resilient retries."""

	_retryable_exceptions = (
		gcore_exceptions.DeadlineExceeded,
		gcore_exceptions.ServiceUnavailable,
		gcore_exceptions.ResourceExhausted,
		gcore_exceptions.TooManyRequests,
		gcore_exceptions.InternalServerError,
		gcore_exceptions.Cancelled,
		gcore_exceptions.Unknown,
		gcore_exceptions.Aborted,
		gcore_exceptions.GoogleAPICallError,
	)

	def __init__(self, model: str = DEFAULT_GEMINI_MODEL, max_output_tokens: int = 1024) -> None:
		api_key = os.getenv("GOOGLE_API_KEY")
		if not api_key:
			msg = "GOOGLE_API_KEY environment variable is required for Gemini usage."
			raise LLMError(msg)
		genai.configure(  # type: ignore[attr-defined]
			api_key=api_key,
			transport="grpc",
			client_options={"api_endpoint": DEFAULT_GEMINI_ENDPOINT},
		)
		self._model = genai.GenerativeModel(model)  # type: ignore[attr-defined]
		self._timeout = DEFAULT_GEMINI_TIMEOUT_SECONDS
		self._max_attempts = int(os.getenv("GEMINI_MAX_ATTEMPTS", "6"))
		self._logger = logging.getLogger("nutribench.gemini")
		super().__init__(provider="gemini", model=model, max_output_tokens=max_output_tokens)

	def generate(self, prompt: str, temperature: float) -> str:  # type: ignore[override]
		self._set_last_metadata(None)
		max_tokens = max(512, min(self.max_output_tokens, 8192))
		upper_bound = 8192
		while True:
			text, metadata = self._call_with_retry(prompt, temperature, max_tokens)
			finish_reason = str(metadata.get("finish_reason") or "").upper()
			content_present = bool(metadata.get("content_present"))
			if finish_reason == "MAX_TOKENS" and not content_present and max_tokens < upper_bound:
				next_tokens = min(int(max_tokens * 1.5), upper_bound)
				if next_tokens > max_tokens:
					self._logger.debug(
						"Gemini response truncated at %d tokens; retrying with %d tokens.",
						max_tokens,
						next_tokens,
					)
					max_tokens = next_tokens
					continue
			self._set_last_metadata(metadata)
			if not content_present:
				message = metadata.get("finish_message") or "Gemini response did not contain any text content."
				raise LLMError(
					f"{message} | finish_reason={finish_reason or 'unknown'} | response_id={metadata.get('response_id')}"
				)
			return text

	def _call_with_retry(
		self,
		prompt: str,
		temperature: float,
		max_tokens: int,
	) -> tuple[str, dict[str, object]]:
		retrying = Retrying(
			reraise=True,
			stop=stop_after_attempt(self._max_attempts),
			wait=wait_random_exponential(multiplier=1, max=30),
			retry=retry_if_exception_type(self._retryable_exceptions + (LLMRetryableError,)),
			before_sleep=before_sleep_log(self._logger, logging.WARNING),
		)
		for attempt in retrying:
			with attempt:
				start_time = time.monotonic()
				text, metadata = self._generate_once(prompt, temperature, max_tokens)
				metadata["attempt_number"] = attempt.retry_state.attempt_number
				metadata["retry_count"] = attempt.retry_state.attempt_number - 1
				metadata["latency_seconds"] = time.monotonic() - start_time
				metadata["max_output_tokens_requested"] = max_tokens
				return text, metadata
		raise RuntimeError("Gemini retry loop exhausted without returning a response.")

	def _generate_once(
		self,
		prompt: str,
		temperature: float,
		max_tokens: int,
	) -> tuple[str, dict[str, object]]:
		config = genai.types.GenerationConfig(  # type: ignore[attr-defined]
			temperature=temperature,
			max_output_tokens=max_tokens,
			candidate_count=1,
		)
		request_content = [{"role": "user", "parts": [{"text": prompt}]}]
		request_options = RequestOptions(timeout=self._timeout,)
		try:
			stream = self._model.generate_content(
				request_content,
				generation_config=config,
				stream=True,
				request_options=request_options,
			)
		except gcore_exceptions.MethodNotImplemented as exc:
			raise LLMError(
				"Gemini streaming endpoint is not implemented for the selected model. "
				"Switch to a compatible model or disable streaming."
			) from exc
		except gcore_exceptions.InvalidArgument as exc:
			raise LLMError(f"Gemini rejected the request: {exc}") from exc

		parts_text: list[str] = []
		finish_reason: str | None = None
		finish_message: str | None = None
		prompt_feedback: Any | None = None
		response_id: str | None = None
		safety_ratings: list[dict[str, object]] = []
		usage_dict: dict[str, object] | None = None
		candidate_token_count: int | None = None
		chunk_count = 0

		for event in stream:
			chunk_count += 1
			response_id = getattr(event, "response_id", response_id)
			prompt_feedback = getattr(event, "prompt_feedback", prompt_feedback)
			usage = getattr(event, "usage_metadata", None)
			if usage:
				usage_dict = self._maybe_to_dict(usage)
			candidates = getattr(event, "candidates", None) or []
			for candidate in candidates:
				finish_reason = getattr(candidate, "finish_reason", finish_reason)
				finish_message = getattr(candidate, "finish_message", finish_message)
				candidate_token_count = getattr(candidate, "token_count", candidate_token_count)
				safety = getattr(candidate, "safety_ratings", None)
				if safety:
					safety_ratings = [self._maybe_to_dict(item) or {} for item in safety]
				content = getattr(candidate, "content", None)
				if content:
					for part in getattr(content, "parts", []) or []:
						text = self._extract_part_text(part)
						if text:
							parts_text.append(text)

		finish_reason_norm = self._normalize_finish_reason(finish_reason)
		metadata: dict[str, object] = {
			"response_id": response_id,
			"finish_reason": finish_reason_norm or finish_reason or "",
			"finish_message": finish_message or "",
			"prompt_feedback": self._maybe_to_dict(prompt_feedback) or {},
			"safety_ratings": safety_ratings,
			"usage_metadata": usage_dict or {},
			"token_count": candidate_token_count,
			"content_present": bool(parts_text),
			"chunk_count": chunk_count,
		}

		prompt_feedback_dict = metadata.get("prompt_feedback")
		block_reason = (
			prompt_feedback_dict.get("block_reason")
			if isinstance(prompt_feedback_dict, dict)
			else None
		)
		if block_reason:
			metadata["prompt_block_reason"] = block_reason
			self._set_last_metadata(metadata)
			raise LLMError(
				"Gemini blocked the prompt | "
				f"block_reason={block_reason} | response_id={response_id}"
			)

		if parts_text:
			return "\n".join(filter(None, parts_text)).strip(), metadata

		self._set_last_metadata(metadata)
		if finish_reason_norm == "MAX_TOKENS":
			return "", metadata

		non_retryable = {
			"SAFETY",
			"BLOCKLIST",
			"PROHIBITED_CONTENT",
			"SPII",
			"RECITATION",
			"IMAGE_SAFETY",
		}
		if finish_reason_norm and finish_reason_norm in non_retryable:
			raise LLMError(
				"Gemini response blocked without content."
				f" finish_reason={finish_reason_norm}"
			)

		raise LLMRetryableError(
			"Gemini response did not contain any text content."
			f" finish_reason={finish_reason_norm or finish_reason or 'unknown'}"
		)

	@staticmethod
	def _normalize_finish_reason(value: Any) -> str | None:
		if value is None:
			return None
		if isinstance(value, str):
			return value.upper()
		text = str(value)
		return text.split(".")[-1].upper()

	@staticmethod
	def _maybe_to_dict(obj: object) -> dict[str, object] | None:
		if obj is None:
			return None
		if isinstance(obj, dict):
			return obj
		if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
			try:
				return obj.to_dict()  # type: ignore[assignment]
			except Exception:  # noqa: BLE001 - best-effort conversion
				return None
		return None

	@staticmethod
	def _extract_part_text(part: object) -> str | None:
		"""Best-effort extraction of textual content from a Gemini part object."""

		if part is None:
			return None
		if hasattr(part, "text"):
			text = getattr(part, "text", None)
			if isinstance(text, str) and text.strip():
				return text.strip()
		part_dict: dict[str, object] | None = None
		if isinstance(part, dict):
			part_dict = part
		elif hasattr(part, "to_dict") and callable(getattr(part, "to_dict")):
			try:
				part_dict = part.to_dict()  # type: ignore[assignment]
			except Exception:  # noqa: BLE001 - fallback only
				part_dict = None
		if not part_dict:
			return None
		text_value = part_dict.get("text")
		if isinstance(text_value, str) and text_value.strip():
			return text_value.strip()
		for alt_key in ("functionCall", "function_call", "struct", "structValue", "json"):
			if alt_key in part_dict:
				try:
					return json.dumps(part_dict[alt_key], sort_keys=True)
				except Exception:  # noqa: BLE001 - fallback serialization
					return str(part_dict[alt_key])
		return None


class OpenAIClient(LLMClient):
	"""Client wrapper around the OpenAI Chat Completions API with retries."""

	def __init__(self, model: str = DEFAULT_GPT_MODEL, max_output_tokens: int = 512) -> None:
		api_key = os.getenv("OPENAI_API_KEY")
		if not api_key:
			msg = "OPENAI_API_KEY environment variable is required for GPT usage."
			raise LLMError(msg)
		self._client = OpenAI(api_key=api_key)
		super().__init__(provider="openai", model=model, max_output_tokens=max_output_tokens)

	@retry(
		reraise=True,
		stop=stop_after_attempt(5),
		wait=wait_exponential(multiplier=1, min=1, max=10),
		retry=retry_if_exception_type(Exception),
	)
	def generate(self, prompt: str, temperature: float) -> str:  # type: ignore[override]
		response = self._client.chat.completions.create(
			model=self.model,
			temperature=temperature,
			max_tokens=self.max_output_tokens,
			messages=[{"role": "user", "content": prompt}],
		)
		usage = getattr(response, "usage", None)
		if usage is not None:
			usage_dict: dict[str, object]
			try:
				usage_dict = usage.model_dump()  # type: ignore[assignment,no-any-unimported]
			except Exception:  # noqa: BLE001 - fallback for older SDKs
				usage_dict = getattr(usage, "to_dict", lambda: {})()
		else:
			usage_dict = {}
		finish_reason = None
		if response.choices:
			finish_reason = getattr(response.choices[0], "finish_reason", None)
		content = response.choices[0].message.content if response.choices else None
		self._set_last_metadata(
			{
				"finish_reason": finish_reason or "",
				"usage_metadata": usage_dict,
				"content_present": bool(content and content.strip()),
			}
		)
		if content is None:
			raise LLMError("OpenAI response did not contain any text choices.")
		return content.strip()


class OfflineClient(LLMClient):
	"""Deterministic heuristic model used for dry-run testing without API calls."""

	def __init__(self) -> None:
		super().__init__(provider="offline", model="heuristic", max_output_tokens=128)

	def _extract_meal(self, prompt: str) -> str:
		match = re.search(r"Meal:\s*(.+)", prompt, re.IGNORECASE | re.DOTALL)
		if match:
			return match.group(1).strip()
		return prompt.strip()

	def _estimate_carb(self, meal: str) -> float:
		word_count = max(1, len(meal.split()))
		digit_sum = sum(int(ch) for ch in meal if ch.isdigit()) or 1
		return min(120.0, 3.2 * (word_count ** 0.8) + digit_sum)

	def generate(self, prompt: str, temperature: float) -> str:  # type: ignore[override]
		meal = self._extract_meal(prompt)
		estimate = self._estimate_carb(meal)
		noise = (temperature or 0.0) * 2.0
		result = f"Approximately {estimate + noise:.1f} grams"
		self._set_last_metadata(
			{
				"finish_reason": "STOP",
				"content_present": True,
			}
		)
		return result


def get_llm_client(provider: str, model: str | None = None) -> LLMClient:
	"""Factory for obtaining an :class:`LLMClient` given a provider string."""

	normalized = provider.strip().lower()
	if normalized == "gemini":
		return GeminiClient(model=model or DEFAULT_GEMINI_MODEL)
	if normalized in {"openai", "gpt"}:
		return OpenAIClient(model=model or DEFAULT_GPT_MODEL)
	if normalized in {"offline", "mock"}:
		return OfflineClient()
	msg = f"Unsupported provider '{provider}'. Choose 'gemini', 'openai', or 'offline'."
	raise ValueError(msg)


def extract_numeric_value(text: str) -> float:
	"""Extract the last floating-point number from ``text``.

	Returns ``math.nan`` if no numeric value is found so the caller can decide how
	to handle invalid model outputs.
	"""

	matches = re.findall(r"[-+]?[0-9]*\.?[0-9]+", text)
	if not matches:
		return math.nan
	return float(matches[-1])


def sanitize_meal_description(meal_description: str) -> str:
	"""Normalize whitespace and clip long meal descriptions for prompt safety."""

	collapsed = re.sub(r"\s+", " ", meal_description or "").strip()
	if len(collapsed) <= MAX_MEAL_DESCRIPTION_CHARS:
		return collapsed
	return collapsed[: MAX_MEAL_DESCRIPTION_CHARS - 3].rstrip() + "..."


def compute_metrics(samples: Sequence[SampleEvaluation]) -> dict[str, float]:
	"""Compute standard evaluation metrics from sample results."""

	errors = np.array([s.error for s in samples], dtype=float)
	actuals = np.array([s.actual_carb for s in samples], dtype=float)
	preds = np.array([s.prediction for s in samples], dtype=float)

	mae = float(np.nanmean(errors))
	rmse = float(np.sqrt(np.nanmean((preds - actuals) ** 2)))
	within_threshold = float(np.nanmean(errors <= ACCURACY_THRESHOLD))
	corr = float(np.corrcoef(actuals, preds)[0, 1]) if len(samples) > 1 else float("nan")

	return {
		"mae": mae,
		"rmse": rmse,
		"acc_within_7_5": within_threshold,
		"corr": corr,
	}


def load_split(path: Path) -> pd.DataFrame:
	"""Load a CSV split with standard column validation."""

	df = pd.read_csv(path)
	for column in ("meal_description", "carb"):
		if column not in df.columns:
			msg = f"Required column '{column}' missing from {path}."
			raise ValueError(msg)
	return df


def select_samples(df: pd.DataFrame, sample_size: int, seed: int = SEED) -> pd.DataFrame:
	"""Sample a subset of the dataframe without replacement."""

	sample_size = min(sample_size, len(df))
	return df.sample(n=sample_size, random_state=seed, replace=False)


def evaluate_prompt(
	client: LLMClient,
	prompt_template: str,
	df: pd.DataFrame,
	sample_size: int,
	iteration: int,
	temperature: float = 0.0,
	seed_offset: int = 0,
	show_progress: bool = True,
) -> PromptEvaluation:
	"""Evaluate a prompt on a subset of examples and compute metrics."""

	subset = select_samples(df, sample_size, seed=SEED + iteration + seed_offset)
	samples: list[SampleEvaluation] = []

	iterator = subset.iterrows()
	if show_progress:
		iterator = tqdm(iterator, total=len(subset), desc="Evaluating", unit="meal")

	requests_sent = 0

	for _, row in iterator:
		meal_text = sanitize_meal_description(row["meal_description"])
		filled_prompt = prompt_template.format(meal_description=meal_text)
		sample_metadata: dict[str, object] | None = None
		error_message: str | None = None
		try:
			response = client.generate(filled_prompt, temperature=temperature)
			had_error = False
		except Exception as exc:  # noqa: BLE001 - broad except to keep evaluation running
			had_error = True
			error_message = str(exc)
			logging.warning(
				"LLM generation failed for sample | iteration=%s | error=%s",
				iteration,
				exc,
			)
			response = f"[ERROR] {exc}"
		finally:
			if hasattr(client, "get_last_response_metadata"):
				metadata = client.get_last_response_metadata()
				if metadata:
					sample_metadata = metadata
		if had_error:
			prediction = 0.0
			if sample_metadata is None:
				sample_metadata = {}
			sample_metadata.setdefault("error", error_message or "UNKNOWN")
		else:
			prediction = extract_numeric_value(response)
			if math.isnan(prediction):
				prediction = 0.0
			elif sample_metadata is None:
				sample_metadata = {}
		error = abs(prediction - float(row["carb"]))
		samples.append(
			SampleEvaluation(
				meal_description=row["meal_description"],
				actual_carb=float(row["carb"]),
				prediction=float(prediction),
				error=float(error),
				response_text=response,
				metadata=sample_metadata,
			)
		)
		requests_sent += 1
		if (
			GEMINI_REQUESTS_PER_BATCH > 0
			and requests_sent % GEMINI_REQUESTS_PER_BATCH == 0
		):
			time.sleep(GEMINI_BATCH_COOLDOWN_SECONDS)

	metrics = compute_metrics(samples)
	finish_counts: Counter[str] = Counter()
	retry_tally = 0
	latencies: list[float] = []
	for sample in samples:
		if not sample.metadata:
			continue
		finish_reason_value = sample.metadata.get("finish_reason")
		if isinstance(finish_reason_value, str) and finish_reason_value:
			finish_counts[finish_reason_value.upper()] += 1
		retry_value = sample.metadata.get("retry_count")
		if isinstance(retry_value, (int, float)):
			retry_tally += int(retry_value)
		latency_value = sample.metadata.get("latency_seconds")
		if isinstance(latency_value, (int, float)):
			latencies.append(float(latency_value))

	error_sample_count = sum(1 for sample in samples if sample.response_text.startswith("[ERROR]"))
	avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

	metadata_payload: dict[str, object] = {
		"seed_offset": seed_offset,
		"sample_indices": subset.index.tolist(),
		"error_sample_count": error_sample_count,
		"total_retry_count": retry_tally,
		"average_latency_seconds": avg_latency,
	}
	if finish_counts:
		metadata_payload["finish_reason_counts"] = dict(finish_counts)
	return PromptEvaluation(
		prompt=prompt_template,
		provider=client.provider,
		model=client.model,
		metrics=metrics,
		samples=samples,
		iteration=iteration,
		evaluation_size=len(subset),
		metadata=metadata_payload,
	)


def _format_example(sample: SampleEvaluation) -> str:
	def _truncate(text: str, max_len: int) -> str:
		if len(text) <= max_len:
			return text
		return text[: max_len - 1] + "…"

	lines = [
		f"Meal: {_truncate(sample.meal_description.strip(), 140)}",
		f"Prediction {sample.prediction:.1f} g vs actual {sample.actual_carb:.1f} g",
	]
	response_text = sample.response_text.strip()
	if response_text.startswith("[ERROR]"):
		lines.append("Model response failed; treat as missing.")
	elif response_text:
		lines.append("Model output excerpt: " + _truncate(response_text, 60))
	return "\n".join(lines)


def generate_gradient(
	client: LLMClient,
	prompt_text: str,
	evaluation: PromptEvaluation,
	top_k: int,
) -> str:
	"""Produce a textual gradient (critique) using the largest-error samples."""

	sorted_samples = sorted(evaluation.samples, key=lambda s: s.error, reverse=True)
	worst_cases = sorted_samples[: min(top_k, 3, len(sorted_samples))]
	examples = "\n\n".join(_format_example(sample) for sample in worst_cases)
	if len(examples) > 2500:
		examples = examples[:2500] + "\n…(truncated)…"
	critique_prompt = (
		"You are reviewing a prompt that estimates carbohydrate content.\n"
		"Prompt under review:\n"
		'"""' "\n"
		f"{prompt_text}\n"
		'"""' "\n\n"
		"Challenging cases (max 3) with high errors:\n"
		f"{examples}\n\n"
		"Analyze the systematic issues and respond with EXACTLY 3 bullet points.\n"
		"Each bullet must stay under 15 words and include a concrete instruction or guardrail."
	)

	try:
		return client.generate(critique_prompt, temperature=0.7)
	except LLMError as exc:  # pragma: no cover - rare fallback
		logging.warning(
			"Gemini critique generation failed; using heuristic fallback | error=%s",
			exc,
		)
		return _heuristic_gradient(worst_cases)


def _heuristic_gradient(samples: Sequence[SampleEvaluation]) -> str:
	"""Fallback textual gradient if the LLM cannot supply one."""

	if not samples:
		return (
			"- Require the assistant to echo serving details before estimating.\n"
			"- Ask for intermediate carb reasoning steps referencing food types.\n"
			"- Enforce a single numeric gram answer, no extra words."
		)

	under_count = sum(1 for s in samples if s.prediction < s.actual_carb)
	over_count = sum(1 for s in samples if s.prediction > s.actual_carb)
	avg_error = sum(s.error for s in samples) / max(1, len(samples))
	bullets: list[str] = []

	if under_count >= over_count:
		bullets.append("Highlight high-sugar meals and require larger baseline carb estimates.")
	if over_count > 0:
		bullets.append("Add a reminder to temper carbs for low-carb snacks and drinks.")
	if avg_error > 25:
		bullets.append("Request step-by-step carb breakdown before giving the final number.")
	bullets.append("Demand the answer be a single number ending with 'grams'.")

	return "\n".join(f"- {point}" for point in bullets[:3])


def edit_prompt(
	client: LLMClient,
	prompt_text: str,
	critique: str,
	num_variants: int = 1,
	temperature: float = 0.7,
) -> list[str]:
	"""Use the textual gradient to synthesize one or more improved prompts."""

	variant_instruction = (
		f"Return {num_variants} improved prompts, each separated by a blank line."
		if num_variants > 1
		else "Return only the improved prompt without additional commentary."
	)

	edit_prompt = (
		"You are an expert prompt engineer applying ProTeGi to improve prompts.\n"
		"Current prompt:\n"
		'"""' "\n"
		f"{prompt_text}\n"
		'"""' "\n\n"
		"Critique:\n"
		f"{critique}\n\n"
		"Rewrite the prompt so that it:\n"
		"- preserves the original objective (predict carbohydrate grams)\n"
		"- enforces a single numeric output (grams)\n"
		"- encourages reasoning or intermediate estimation steps when helpful\n"
		"- clarifies handling of missing quantities or multi-item meals\n\n"
		f"{variant_instruction}"
	)

	raw_response = client.generate(edit_prompt, temperature=temperature)
	return _parse_prompt_variants(raw_response, expected=num_variants)


def _parse_prompt_variants(text: str, expected: int) -> list[str]:
	"""Extract individual prompt candidates from model output."""

	if not text.strip():
		return []

	lines = text.splitlines()
	variants: list[str] = []
	current: list[str] = []

	for line in lines:
		if re.match(r"^\s*(?:\d+[\).:-]?\s+|-+\s+|\*\s+)", line):
			if current:
				variants.append("\n".join(current).strip())
			current = []
			line = re.sub(r"^\s*(?:\d+[\).:-]?\s+|-+\s+|\*\s+)", "", line)
		if line.strip():
			current.append(line.rstrip())

	if current:
		variants.append("\n".join(current).strip())

	if not variants:
		return [text.strip()]

	if expected > 0 and len(variants) > expected:
		return variants[:expected]

	return variants


def ensure_directory(path: Path) -> None:
	path.mkdir(parents=True, exist_ok=True)


def save_prompt(text: str, path: Path) -> None:
	ensure_directory(path.parent)
	path.write_text(text.strip() + "\n", encoding="utf-8")


def save_jsonl(records: Iterable[dict[str, object]], path: Path) -> None:
	ensure_directory(path.parent)
	with path.open("a", encoding="utf-8") as file:
		for record in records:
			file.write(json.dumps(record) + "\n")


def save_evaluation(evaluation: PromptEvaluation, path: Path) -> None:
	ensure_directory(path.parent)
	payload = {
		"timestamp": datetime.utcnow().isoformat(timespec="seconds"),
		"provider": evaluation.provider,
		"model": evaluation.model,
		"iteration": evaluation.iteration,
		"evaluation_size": evaluation.evaluation_size,
		"metrics": evaluation.metrics,
		"samples": [asdict(sample) for sample in evaluation.samples],
		"metadata": evaluation.metadata,
	}
	path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def log_evaluation_summary(logger: logging.Logger, evaluation: PromptEvaluation, context: str) -> None:
	"""Emit debug-friendly diagnostics for an evaluation result."""

	metadata = evaluation.metadata or {}
	finish_counts = metadata.get("finish_reason_counts") if isinstance(metadata, dict) else None
	if isinstance(finish_counts, dict) and finish_counts:
		logger.debug("%s finish reasons: %s", context, finish_counts)
	error_count = metadata.get("error_sample_count") if isinstance(metadata, dict) else None
	if isinstance(error_count, int) and error_count > 0:
		logger.warning(
			"%s encountered %d error samples (responses replaced with 0 g).",
			context,
			error_count,
		)
	avg_latency = metadata.get("average_latency_seconds") if isinstance(metadata, dict) else None
	if isinstance(avg_latency, (int, float)) and avg_latency > 0:
		logger.debug("%s average latency: %.2fs", context, float(avg_latency))


def setup_logging(log_dir: Path, verbose: bool = False) -> logging.Logger:
	ensure_directory(log_dir)
	log_path = log_dir / "optimization.log"
	level = logging.DEBUG if verbose else logging.INFO
	logging.basicConfig(
		level=level,
		format="%(asctime)s | %(levelname)s | %(message)s",
		handlers=[
			logging.FileHandler(log_path, encoding="utf-8"),
			logging.StreamHandler(),
		],
	)
	logger = logging.getLogger("nutribench")
	logger.setLevel(level)
	logger.info("Logging initialized. Writing detailed logs to %s", log_path)
	return logger


def timestamped_filename(prefix: str, suffix: str) -> str:
	return f"{prefix}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}{suffix}"


def plot_metric_progress(
	history: Sequence[PromptEvaluation],
	output_path: Path,
	metric: str = "mae",
) -> None:
	"""Generate a simple line plot showing metric progression across iterations."""

	ensure_directory(output_path.parent)
	import matplotlib.pyplot as plt  # Imported lazily to keep CLI start-up light

	iterations = [h.iteration for h in history]
	values = [h.metrics.get(metric, float("nan")) for h in history]

	plt.figure(figsize=(6, 4))
	plt.plot(iterations, values, marker="o", linestyle="-", color="#007aff")
	plt.title(f"Prompt optimization progress ({metric.upper()})")
	plt.xlabel("Iteration")
	plt.ylabel(metric.upper())
	plt.grid(True, linestyle="--", alpha=0.4)
	plt.tight_layout()
	plt.savefig(output_path, dpi=200)
	plt.close()


def select_top_candidates(
	candidates: Sequence[PromptCandidate],
	beam_width: int,
) -> list[PromptCandidate]:
	"""Return the top-performing candidates based on MAE."""

	ordered = sorted(candidates, key=lambda candidate: candidate.score)
	return ordered[:beam_width]


def candidate_to_record(candidate: PromptCandidate) -> dict[str, object]:
	"""Serialize a prompt candidate for JSON logging."""

	evaluation = candidate.evaluation
	metrics = evaluation.metrics if evaluation else {}
	return {
		"id": candidate.id,
		"iteration": candidate.iteration,
		"parent_id": candidate.parent_id,
		"variant_index": candidate.variant_index,
		"critique": candidate.critique,
		"prompt": candidate.prompt,
		"metrics": metrics,
		"score": candidate.score,
		"provider": evaluation.provider if evaluation else None,
		"model": evaluation.model if evaluation else None,
		"evaluation_size": evaluation.evaluation_size if evaluation else None,
		"evaluation_metadata": evaluation.metadata if evaluation else None,
	}
