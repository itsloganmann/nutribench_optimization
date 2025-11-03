"""ProTeGi-style prompt optimization for the NutriBench dataset."""

from __future__ import annotations

import itertools
import json
import logging
from pathlib import Path
from typing import Optional

import typer

from . import utils


app = typer.Typer(add_completion=False, no_args_is_help=True)


def _load_prompt(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def _record_history(history_path: Path, record: dict[str, object]) -> None:
    utils.save_jsonl([record], history_path)


@app.command()
def run(
    train_path: Path = typer.Option(
        Path("data/train.csv"),
        help="Path to the training split CSV produced by split_data.py.",
    ),
    val_path: Path = typer.Option(
        Path("data/val.csv"),
        help="Path to the validation split CSV produced by split_data.py.",
    ),
    prompt_path: Path = typer.Option(
        Path("prompts/baseline_prompt.txt"),
        help="Baseline prompt template file (must contain {meal_description}).",
    ),
    provider: str = typer.Option(
        utils.DEFAULT_PROVIDER,
        help="LLM provider to use ('gemini', 'openai', or 'offline').",
    ),
    model: Optional[str] = typer.Option(
        None,
        help="Override the default model for the selected provider.",
    ),
    iterations: int = typer.Option(
        4,
        min=1,
        help="Maximum number of optimization iterations to execute.",
    ),
    sample_size: int = typer.Option(
        120,
        min=10,
        help="Number of training samples to evaluate per candidate each iteration.",
    ),
    val_sample_size: int = typer.Option(
        1000,
        min=100,
        help="Number of validation samples for the final evaluation (capped at dataset size).",
    ),
    top_k: int = typer.Option(
        5,
        min=3,
        help="Number of highest-error samples used to produce critiques.",
    ),
    beam_width: int = typer.Option(
        3,
        min=1,
        help="Number of prompt candidates to retain per iteration (beam width).",
    ),
    beam_expansion: int = typer.Option(
        2,
        min=1,
        help="Number of improved variants to generate for each candidate.",
    ),
    eval_temperature: float = typer.Option(
        0.0,
        help="Temperature for evaluation calls (use 0.0 for deterministic scoring).",
    ),
    generation_temperature: float = typer.Option(
        0.7,
        help="Temperature for critique-driven prompt edits.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run/--live-run",
        help="Use the offline heuristic model instead of external APIs.",
    ),
    progress: bool = typer.Option(
        True,
        "--progress/--no-progress",
        help="Display tqdm progress bars during evaluations.",
    ),
    max_iterations_without_improvement: int = typer.Option(
        3,
        min=1,
        help="Early-stop after this many consecutive iterations without MAE improvement.",
    ),
    results_dir: Path = typer.Option(
        Path("results"),
        help="Directory for storing evaluation artifacts.",
    ),
    prompts_dir: Path = typer.Option(
        Path("prompts"),
        help="Directory for storing evolving prompt variants.",
    ),
    logs_dir: Path = typer.Option(
        Path("logs"),
        help="Directory for optimization logs.",
    ),
    verbose: bool = typer.Option(False, help="Enable verbose logging output."),
) -> None:
    """Run automatic prompt optimization with gradient critiques and beam search."""

    logger = utils.setup_logging(logs_dir, verbose=verbose)
    if dry_run and provider != "offline":
        logger.info("Dry-run requested; switching provider to 'offline'.")
        provider = "offline"
        model = None

    logger.info(
        "Starting prompt optimization | provider=%s | model=%s | beam_width=%d | beam_expansion=%d",
        provider,
        model,
        beam_width,
        beam_expansion,
    )

    train_df = utils.load_split(train_path)
    val_df = utils.load_split(val_path)
    prompt_text = _load_prompt(prompt_path)

    sample_size = min(sample_size, len(train_df))
    validation_size = min(val_sample_size, len(val_df))
    if sample_size < len(train_df):
        logger.info(
            "Training evaluation will sample %d meals out of %d available.",
            sample_size,
            len(train_df),
        )
    if validation_size < val_sample_size:
        logger.warning(
            "Validation set only contains %d meals (requested %d).",
            validation_size,
            val_sample_size,
        )

    client = utils.get_llm_client(provider=provider, model=model)

    utils.ensure_directory(results_dir)
    utils.ensure_directory(prompts_dir)

    history: list[utils.PromptEvaluation] = []
    best_overall: utils.PromptCandidate | None = None
    no_improvement_streak = 0
    iterations_completed = 0

    history_path = results_dir / "optimization_history.jsonl"
    candidate_history_path = results_dir / "candidate_history.jsonl"

    candidate_counter = itertools.count(1)
    evaluation_counter = itertools.count(1)

    def make_candidate(
        prompt: str,
        iteration_index: int,
        parent_id: str | None,
        variant_index: int,
        candidate_id: str | None = None,
    ) -> utils.PromptCandidate:
        identifier = candidate_id or f"iter{iteration_index:02d}_cand{next(candidate_counter):03d}"
        return utils.PromptCandidate(
            id=identifier,
            prompt=prompt.strip(),
            iteration=iteration_index,
            parent_id=parent_id,
            variant_index=variant_index,
        )

    current_beam: list[utils.PromptCandidate] = [
        make_candidate(
            prompt_text,
            iteration_index=1,
            parent_id=None,
            variant_index=0,
            candidate_id="baseline",
        )
    ]
    logger.info("Baseline prompt loaded from %s", prompt_path)

    for iteration in range(1, iterations + 1):
        iterations_completed = iteration
        logger.info("=== Iteration %d/%d (beam width=%d) ===", iteration, iterations, beam_width)

        iteration_dir = results_dir / f"iteration_{iteration:02d}"
        utils.ensure_directory(iteration_dir)

        candidate_pool: list[utils.PromptCandidate] = []
        candidate_records: list[dict[str, object]] = []
        seen_prompts: set[str] = set()

        for beam_index, candidate in enumerate(current_beam, start=1):
            normalized_prompt = candidate.prompt.strip()
            if normalized_prompt in seen_prompts:
                logger.debug("Skipping duplicate prompt in beam: %s", candidate.id)
                continue
            seen_prompts.add(normalized_prompt)

            eval_order = next(evaluation_counter)
            evaluation = utils.evaluate_prompt(
                client=client,
                prompt_template=candidate.prompt,
                df=train_df,
                sample_size=sample_size,
                iteration=iteration,
                temperature=eval_temperature,
                seed_offset=beam_index,
                show_progress=progress,
            )
            evaluation.candidate_id = candidate.id
            evaluation.parent_id = candidate.parent_id
            candidate.evaluation = evaluation
            utils.log_evaluation_summary(
                logger,
                evaluation,
                f"candidate {candidate.id} (iteration {iteration})",
            )
            candidate_pool.append(candidate)

            eval_path = iteration_dir / f"{candidate.id}_evaluation.json"
            utils.save_evaluation(evaluation, eval_path)

            prompt_snapshot_path = prompts_dir / f"iteration_{iteration:02d}_{candidate.id}.txt"
            utils.save_prompt(candidate.prompt, prompt_snapshot_path)

            record = utils.candidate_to_record(candidate)
            record.update(
                {
                    "evaluation_path": str(eval_path),
                    "prompt_path": str(prompt_snapshot_path),
                    "stage": "beam",
                    "evaluation_order": eval_order,
                }
            )

            critique = utils.generate_gradient(
                client=client,
                prompt_text=candidate.prompt,
                evaluation=evaluation,
                top_k=top_k,
            )
            candidate.critique = critique
            record["critique"] = critique

            variant_prompts = utils.edit_prompt(
                client=client,
                prompt_text=candidate.prompt,
                critique=critique,
                num_variants=beam_expansion,
                temperature=generation_temperature,
            )

            generated_candidate_ids: list[str] = []
            for variant_index, variant_prompt in enumerate(variant_prompts, start=1):
                variant_text = variant_prompt.strip()
                if not variant_text:
                    logger.debug("Skipping empty variant from candidate %s", candidate.id)
                    continue
                if variant_text in seen_prompts:
                    logger.debug("Skipping duplicate variant prompt for %s", candidate.id)
                    continue
                seen_prompts.add(variant_text)

                variant_candidate = make_candidate(
                    variant_text,
                    iteration_index=iteration,
                    parent_id=candidate.id,
                    variant_index=variant_index,
                )
                variant_candidate.critique = critique

                variant_eval_order = next(evaluation_counter)
                variant_evaluation = utils.evaluate_prompt(
                    client=client,
                    prompt_template=variant_candidate.prompt,
                    df=train_df,
                    sample_size=sample_size,
                    iteration=iteration,
                    temperature=eval_temperature,
                    seed_offset=beam_index * beam_expansion + variant_index + 10,
                    show_progress=False,
                )
                variant_evaluation.candidate_id = variant_candidate.id
                variant_evaluation.parent_id = variant_candidate.parent_id
                variant_candidate.evaluation = variant_evaluation
                utils.log_evaluation_summary(
                    logger,
                    variant_evaluation,
                    f"candidate {variant_candidate.id} (iteration {iteration})",
                )
                candidate_pool.append(variant_candidate)

                variant_eval_path = iteration_dir / f"{variant_candidate.id}_evaluation.json"
                utils.save_evaluation(variant_evaluation, variant_eval_path)

                variant_prompt_path = prompts_dir / f"iteration_{iteration:02d}_{variant_candidate.id}.txt"
                utils.save_prompt(variant_candidate.prompt, variant_prompt_path)

                variant_record = utils.candidate_to_record(variant_candidate)
                variant_record.update(
                    {
                        "evaluation_path": str(variant_eval_path),
                        "prompt_path": str(variant_prompt_path),
                        "stage": "variant",
                        "source_candidate": candidate.id,
                        "evaluation_order": variant_eval_order,
                    }
                )
                candidate_records.append(variant_record)
                generated_candidate_ids.append(variant_candidate.id)

            record["generated_candidates"] = generated_candidate_ids
            record["generated_prompt_count"] = len(generated_candidate_ids)
            candidate_records.append(record)

        if not candidate_pool:
            raise RuntimeError("No prompt candidates were evaluated in this iteration.")

        utils.save_jsonl(candidate_records, candidate_history_path)

        top_candidates = utils.select_top_candidates(candidate_pool, beam_width)
        best_candidate = top_candidates[0]
        best_eval = best_candidate.evaluation
        if best_eval is None:
            raise RuntimeError("Best candidate is missing an evaluation result.")
        history.append(best_eval)

        logger.info(
            "Iteration %d best candidate %s | MAE=%.3f | Acc@7.5g=%.2f%%",
            iteration,
            best_candidate.id,
            best_candidate.score,
            best_eval.metrics["acc_within_7_5"] * 100,
        )

        iteration_record = {
            "iteration": iteration,
            "best_candidate": utils.candidate_to_record(best_candidate),
            "top_candidates": [utils.candidate_to_record(candidate) for candidate in top_candidates],
            "no_improvement_streak": no_improvement_streak,
        }
        _record_history(history_path, iteration_record)

        if best_overall is None or best_candidate.score < best_overall.score:
            best_overall = best_candidate
            no_improvement_streak = 0
        else:
            no_improvement_streak += 1
            logger.info("No MAE improvement detected (streak=%d)", no_improvement_streak)

        best_prompt_snapshot = prompts_dir / f"best_after_iteration_{iteration:02d}.txt"
        utils.save_prompt(best_candidate.prompt, best_prompt_snapshot)

        if no_improvement_streak >= max_iterations_without_improvement:
            logger.info(
                "Stopping early after %d iterations without improvement.",
                no_improvement_streak,
            )
            break

        current_beam = [
            make_candidate(
                prompt=candidate.prompt,
                iteration_index=iteration + 1,
                parent_id=candidate.id,
                variant_index=rank,
            )
            for rank, candidate in enumerate(top_candidates, start=1)
        ]

    if best_overall is None or best_overall.evaluation is None:
        raise RuntimeError("Optimization failed to evaluate any prompt candidates.")

    best_prompt_path = prompts_dir / "best_prompt.txt"
    utils.save_prompt(best_overall.prompt, best_prompt_path)
    logger.info(
        "Best prompt across all iterations: %s (MAE=%.3f)",
        best_overall.id,
        best_overall.score,
    )

    validation_eval = utils.evaluate_prompt(
        client=client,
        prompt_template=best_overall.prompt,
        df=val_df,
        sample_size=validation_size,
        iteration=iterations_completed + 1,
        temperature=eval_temperature,
        seed_offset=0,
        show_progress=progress,
    )
    validation_path = results_dir / "validation_evaluation.json"
    utils.save_evaluation(validation_eval, validation_path)
    utils.log_evaluation_summary(
        logger,
        validation_eval,
        "validation evaluation",
    )

    _record_history(
        history_path,
        {
            "iteration": "validation",
            "best_candidate_id": best_overall.id,
            "metrics": validation_eval.metrics,
            "evaluation_size": validation_eval.evaluation_size,
            "prompt_path": str(best_prompt_path),
        },
    )

    if history:
        metrics_plot = results_dir / "metrics_progress.png"
        utils.plot_metric_progress(history, metrics_plot)
        logger.info("Metric progression plot saved to %s", metrics_plot)

    summary = {
        "iterations_requested": iterations,
        "iterations_completed": iterations_completed,
        "train_sample_size": sample_size,
        "validation_sample_size": validation_size,
        "provider": client.provider,
        "model": client.model,
        "beam_width": beam_width,
        "beam_expansion": beam_expansion,
        "eval_temperature": eval_temperature,
        "generation_temperature": generation_temperature,
        "dry_run": provider == "offline",
        "best_candidate": utils.candidate_to_record(best_overall),
        "validation_metrics": validation_eval.metrics,
        "best_prompt_path": str(best_prompt_path),
        "history_path": str(history_path),
        "candidate_history_path": str(candidate_history_path),
    }

    summary_path = results_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Optimization summary written to %s", summary_path)

    typer.secho(
        "Optimization complete! Review summary.json and validation_evaluation.json for results.",
        fg=typer.colors.GREEN,
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
