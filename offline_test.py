"""Offline smoke test — no network, no Metaculus token, no submission.

Builds 3 in-memory questions (binary / multiple_choice / numeric), pushes them
through the real FallTemplateBot2026 pipeline with a deterministic canned LLM
(subclasses GeneralLlm so the framework accepts it), and asserts the outputs.

Validates: prompt building -> forecast -> structure_output parsing ->
NumericDistribution CDF -> aggregation -> ForecastReport + run_log journal.

Usage:
  .venv/Scripts/python.exe offline_test.py                 # canned backend
  .venv/Scripts/python.exe offline_test.py --backend ollama [--model ollama_chat/qwen3:1.7b]
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time

import dotenv

dotenv.load_dotenv()

from bot_helpers import silence_noisy_dependencies

silence_noisy_dependencies()

from forecasting_tools import (  # noqa: E402
    BinaryQuestion,
    GeneralLlm,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    PredictedOptionList,
)

import run_log  # noqa: E402
from main import FallTemplateBot2026  # noqa: E402

MC_OPTIONS = ["Red", "Blue", "Green"]
# Numeric question bounds: days until 2027-01-01, 0..100, open bounds
NUMERIC_LOWER, NUMERIC_UPPER = 0, 100
NUMERIC_PERCENTILES = [
    {"percentile": 0.1, "value": 45},
    {"percentile": 0.2, "value": 55},
    {"percentile": 0.4, "value": 62},
    {"percentile": 0.6, "value": 68},
    {"percentile": 0.8, "value": 74},
    {"percentile": 0.9, "value": 80},
]


class CannedForecastLlm(GeneralLlm):
    """Deterministic 'reasoning' generator. Subclasses GeneralLlm purely so
    ForecastBot.get_llm(..., guarantee_type='llm') passes it through."""

    def __init__(self) -> None:
        super().__init__(model="canned/forecast-mock", allowed_tries=1)

    async def invoke(self, prompt, system_prompt: str | None = None) -> str:  # type: ignore[override]
        text = str(prompt)
        if MC_OPTIONS[0] in text and "options" in text.lower():
            return (
                "(a) Resolution is near-immediate.\n"
                "(b) Status quo: Red is most likely by construction.\n"
                "(c) Unexpected outcome: Green.\n"
                "Final answer:\n"
                "Red: 60%\nBlue: 30%\nGreen: 10%"
            )
        if "Percentile 90" in text:
            return (
                "(a) Days until 2027-01-01, about 66 as of the prompt date.\n"
                "(b) Nothing changes the calendar; ~66 days.\n"
                "(c) Trend continues: 66 days.\n"
                "(d) Experts agree: 66 days.\n"
                "(e) Low outcome: leap-second oddities, 45 days.\n"
                "(f) High outcome: calendar rewrite, 80 days.\n"
                "Percentile 10: 45\nPercentile 20: 55\nPercentile 40: 62\n"
                "Percentile 60: 68\nPercentile 80: 74\nPercentile 90: 80"
            )
        # default: binary
        return (
            "(a) Outcome known tomorrow morning.\n"
            "(b) Status quo: the sun rises, Yes.\n"
            "(c) No outcome: impossible heavy cloud cover does not stop sunrise.\n"
            "(d) Yes outcome: sunrise before 8 AM as usual.\n"
            "Probability: 98%"
        )


class CannedParserLlm(GeneralLlm):
    """Returns the exact JSON each output_type expects, keyed off the schema
    text that structure_output embeds in its prompt."""

    def __init__(self) -> None:
        super().__init__(model="canned/parser-mock", allowed_tries=1)

    async def invoke(self, prompt, system_prompt: str | None = None) -> str:  # type: ignore[override]
        import json

        text = str(prompt)
        # structure_output embeds the output class name in its prompt; classify
        # on that (field names are not always present in the instruction text).
        if "BinaryPrediction" in text:
            return json.dumps({"prediction_in_decimal": 0.98})
        if "PredictedOptionList" in text:
            return json.dumps(
                {
                    "predicted_options": [
                        {"option_name": "Red", "probability": 0.6},
                        {"option_name": "Blue", "probability": 0.3},
                        {"option_name": "Green", "probability": 0.1},
                    ]
                }
            )
        if "percentile" in text.lower():
            return json.dumps(NUMERIC_PERCENTILES)
        raise AssertionError(f"canned parser cannot classify prompt:\n{text[:400]}")


def build_questions():
    binary = BinaryQuestion(
        question_text=(
            "Will the sun rise in Shanghai before 8:00 AM local time tomorrow "
            "(astronomical sunrise visible above the horizon)?"
        ),
        background_info="Astronomical fact test.",
        resolution_criteria="Resolves Yes if sunrise time in Shanghai tomorrow is before 08:00 local.",
        fine_print="",
    )
    multiple_choice = MultipleChoiceQuestion(
        question_text="A ball is drawn from an urn containing 60% red, 30% blue and 10% green balls. Which color is drawn?",
        background_info="Single draw, well-mixed urn.",
        resolution_criteria="Resolves to the drawn color.",
        fine_print="",
        options=MC_OPTIONS,
    )
    numeric = NumericQuestion(
        question_text="How many days remain between 2026-10-27 and 2027-01-01 (inclusive of neither endpoint)?",
        background_info="Calendar arithmetic.",
        resolution_criteria="Resolves to the number of whole days between the two dates.",
        fine_print="",
        upper_bound=NUMERIC_UPPER,
        lower_bound=NUMERIC_LOWER,
        open_upper_bound=True,
        open_lower_bound=True,
    )
    return [binary, multiple_choice, numeric]


def build_bot(backend: str, model: str | None) -> FallTemplateBot2026:
    predictions = 5
    if backend == "ollama":
        m = model or "ollama_chat/qwen3:1.7b"
        llm = GeneralLlm(model=m, temperature=0.3, timeout=600, allowed_tries=1)
        llms = {"default": llm, "parser": llm, "summarizer": llm, "researcher": "no_research"}
        predictions = 1  # CPU speed
    elif backend == "kimi":
        # Reads DEFAULT_LLM_MODEL/_API_BASE/_API_KEY from .env (paid API, ~¥1 for 3 questions)
        import os

        m = model or os.environ["DEFAULT_LLM_MODEL"]
        llm = GeneralLlm(
            model=m,
            api_base=os.environ["DEFAULT_LLM_API_BASE"],
            api_key=os.environ["DEFAULT_LLM_API_KEY"],
            # K3 is a reasoning model: only temperature=1 is accepted, so omit it
            timeout=300,
            allowed_tries=2,
        )
        llms = {"default": llm, "parser": llm, "summarizer": llm, "researcher": "no_research"}
        predictions = 1  # cost control for smoke test
    else:
        llms = {
            "default": CannedForecastLlm(),
            "parser": CannedParserLlm(),
            "summarizer": CannedForecastLlm(),
            "researcher": "no_research",
        }
    return FallTemplateBot2026(
        research_reports_per_question=1,
        predictions_per_research_report=predictions,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=False,  # hard guarantee: nothing is posted
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=True,
        llms=llms,
    )


def check(reports) -> list[str]:
    problems = []
    if len(reports) != 3:
        return [f"expected 3 reports, got {len(reports)}"]
    bin_r, mc_r, num_r = reports

    if isinstance(bin_r, BaseException):
        problems.append(f"binary raised: {bin_r!r}")
    else:
        p = bin_r.prediction
        if not (isinstance(p, float) and 0.01 <= p <= 0.99):
            problems.append(f"binary prediction out of range/type: {p!r}")

    if isinstance(mc_r, BaseException):
        problems.append(f"multiple_choice raised: {mc_r!r}")
    else:
        p = mc_r.prediction
        if not isinstance(p, PredictedOptionList):
            problems.append(f"MC prediction wrong type: {type(p).__name__}")
        else:
            total = sum(o.probability for o in p.predicted_options)
            if abs(total - 1.0) > 0.01 or len(p.predicted_options) != 3:
                problems.append(f"MC options invalid: n={len(p.predicted_options)} sum={total}")

    if isinstance(num_r, BaseException):
        problems.append(f"numeric raised: {num_r!r}")
    else:
        p = num_r.prediction
        if not isinstance(p, NumericDistribution):
            problems.append(f"numeric prediction wrong type: {type(p).__name__}")
        else:
            cdf = p.get_cdf()
            if len(cdf) != 201:
                problems.append(f"CDF length {len(cdf)} != 201")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["canned", "ollama", "kimi"], default="canned")
    parser.add_argument("--model", default=None, help="model override (kimi/ollama backends)")
    args = parser.parse_args()

    bot = build_bot(args.backend, args.model)
    questions = build_questions()
    run = run_log.start_run(f"offline_{args.backend}", publish=False, targets=["in-memory:3"])

    t0 = time.monotonic()
    reports = asyncio.run(bot.forecast_questions(questions, return_exceptions=True))
    elapsed = time.monotonic() - t0

    run_log.record_reports(run, reports)
    run_log.attach_model_usage(run)

    problems = check(reports)
    print()
    print("=" * 70)
    print(f"OFFLINE TEST ({args.backend} backend): {'PASS' if not problems else 'FAIL'} "
          f"in {elapsed:.1f}s, publish=False throughout")
    for r in reports:
        if isinstance(r, BaseException):
            print(f"  ❌ {type(r).__name__}: {str(r)[:200]}")
        else:
            print(f"  ✅ {type(r.question).__name__:22s} -> {str(r.prediction)[:90]}")
    for p in problems:
        print(f"  ⚠️  {p}")
    print("=" * 70)

    exit_code = run_log.finish_run(run, reports)
    if problems:
        return 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
