# NutriBench Prompt Optimization

Automatic prompt optimization for the NutriBench carbohydrate estimation benchmark, inspired by ProTeGi textual gradient descent. This guide assumes a MacBook Pro running macOS Monterey or later and walks through every step required to reproduce the workflow—from installing dependencies to running the optimization loop, analysing results, and preparing presentation assets.

## 📋 Table of Contents

0. [Key Deliverables](#key-deliverables)
1. [Initial Mac Setup](#1-initial-mac-setup)
2. [Create Project Workspace](#2-create-project-workspace)
3. [Install Python Dependencies](#3-install-python-dependencies)
4. [Configure API Keys](#4-configure-api-keys)
5. [Download & Split NutriBench](#5-download--split-nutribench)
6. [Project Layout](#6-project-layout)
7. [ProTeGi Optimization Pipeline](#7-protegi-optimization-pipeline)
8. [Evaluation & Reporting](#8-evaluation--reporting)
9. [API Cost Tracking](#9-api-cost-tracking)
10. [Smoke Tests & Troubleshooting](#10-smoke-tests--troubleshooting)
11. [Next Steps & Slide Outline](#11-next-steps--slide-outline)

The repository is organised to prioritise reproducibility, logging, and deterministic data splits. All commands are intended for the macOS **zsh** shell.

---

## Key Deliverables

- **Comprehensive PDF report** (`docs/nutribench_prompt_optimization_summary.pdf`) summarising methodology, iteration results, validation analysis, and recommendations. Regenerate via `python docs/generate_summary_pdf.py`.
- **Validation metrics chart** (`slides/validation_metrics.png`) for slide decks, produced with `python docs/generate_validation_metrics_plot.py`.
- **Prompt snapshots & artifacts** under `prompts/` and `results/` (ignored by default but regenerable through the optimisation CLI) to trace the textual gradient descent process.
- **Presentation outline** (`slides/outline.md`) aligning with project specs for quick deck assembly.

## 1. Initial Mac Setup

1. Upgrade macOS: Apple  → **About This Mac** → **Software Update**.
2. Install command line tools:

	 ```bash
	 xcode-select --install
	 ```

3. Install Homebrew (copy-paste into Terminal):

	 ```bash
	 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
	 ```

4. Update Homebrew:

	 ```bash
	 brew update
	 ```

---

## 2. Create Project Workspace

```bash
mkdir -p ~/nutribench_optimization
cd ~/nutribench_optimization
python3 -m venv venv
source venv/bin/activate
```

> Once activated, your shell prompt will start with `(venv)`.

---

## 3. Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

The `requirements.txt` file includes:

- `pandas`, `numpy`, `scikit-learn` for data preparation
- `datasets` for loading NutriBench from Hugging Face
- `openai`, `google-generativeai`, and `python-dotenv` for API interactions
- `tenacity`, `tqdm`, `typer` for resilience, progress bars, and CLI
- `matplotlib`, `seaborn` for reporting visuals

---

## 4. Configure API Keys

1. **Google Gemini API key**: Generate via [Google AI Studio](https://aistudio.google.com/app/apikey) after creating a project in Google Cloud Console.
2. **OpenAI API key**: Generate via the [OpenAI API dashboard](https://platform.openai.com/account/api-keys).
3. Store both keys in `.env` (already present in the repo structure):

	 ```env
	 GOOGLE_API_KEY=your_google_api_key_here
	 OPENAI_API_KEY=your_openai_api_key_here
	 ```

4. Optional tuning environment variables:

	 ```env
	 GEMINI_MAX_ATTEMPTS=6           # Number of retry attempts for transient Gemini errors
	 GEMINI_REQUEST_TIMEOUT=180      # Seconds before timing out a streaming response
	 ```

> Keys are consumed via `python-dotenv`. Never commit real keys to source control.

---

## 5. Download & Split NutriBench

Run the scripted workflow to fetch the dataset and create deterministic splits:

```bash
python split_data.py --output-dir data --val-size 1000 --seed 42 --overwrite
```

What happens:

1. Downloads `dongx1997/NutriBench` (CoT version) via Hugging Face `datasets`.
2. Saves the raw CSV to `data/nutribench_v2_cot.csv`.
3. Generates `data/train.csv` and `data/val.csv` with 1,000 validation samples.
4. Records metadata (sizes, seed, dataset info) in `data/metadata.json`.

Quick sanity check once the split completes:

```bash
python - <<'PY'
import pandas as pd
train = pd.read_csv("data/train.csv")
val = pd.read_csv("data/val.csv")
print(f"train rows: {len(train)}")  # expected: 14617
print(f"val rows: {len(val)}")      # expected: 1000
PY
```

These counts match the official NutriBench guidance (≥1,000 validation rows) and are logged in `data/metadata.json` for reproducibility.

> The script exposes CLI flags for cache settings, force re-downloads, verbose logging, etc. Run `python split_data.py --help` for details.

---

## 6. Project Layout

```text
nutribench_optimization/
├── .env                     # API keys (keep private)
├── .gitignore               # Excludes venv, data artifacts, logs
├── README.md                # This guide
├── requirements.txt         # Python dependencies
├── split_data.py            # Dataset download & splitting CLI
├── data/
│   ├── train.csv            # Training split (generated)
│   ├── val.csv              # Validation split (generated)
│   └── metadata.json        # Dataset provenance info
├── prompts/
│   ├── baseline_prompt.txt  # Seed prompt template
│   └── best_prompt.txt      # Populated after optimization
├── results/                 # Iteration metrics, plots, summaries
├── logs/                    # Centralised log files
├── slides/outline.md        # Presentation outline for reporting
└── src/
		├── __init__.py
		├── optimize.py          # ProTeGi optimization CLI (Typer)
		├── utils.py             # Dataset, metrics, and LLM helpers
		└── test_smoke.py        # Simple import smoke test
```

---

## 7. ProTeGi Optimization Pipeline

### 7.1 Dry-Run Validation

Before invoking the full optimization loop, compile sources to ensure syntax correctness:

```bash
python3 -m compileall src split_data.py
```

Run the smoke test (optional but recommended if `pytest` is available):

```bash
python -m pytest src/test_smoke.py
```

### 7.2 Launch Optimization

The optimization CLI is implemented with Typer. Common usage patterns:

```bash
# Display help
python -m src.optimize --help

# Run with defaults (Gemini, 3 iterations, 100-sample batches)
python -m src.optimize

# Explicit run command with extra options
python -m src.optimize run \
	--iterations 5 \
	--sample-size 150 \
	--val-sample-size 1000 \
	--provider gemini \
	--model gemini-1.5-pro
```

Key behaviours:

- Evaluates the current prompt on sampled training meals (default 100) with **temperature 0.0**.
- Computes MAE, RMSE, correlation, and accuracy within 7.5 grams.
- Generates textual gradients from the worst-performing samples (default top 5).
- Synthesises improved prompts via the same provider (temperature 0.7).
- Logs every iteration: raw critiques, improved prompts, metrics, and structured JSON.
- Gemini calls automatically retry transient 5xx/timeout errors with exponential backoff, escalate `max_output_tokens` when `finish_reason=MAX_TOKENS`, and record finish-reason counts; tune retries via `GEMINI_MAX_ATTEMPTS` (default 6).
- Persists per-iteration prompts under `prompts/` and the best prompt as `prompts/best_prompt.txt`.
- Produces a metrics progression plot at `results/metrics_progress.png`.
- Evaluates the best prompt on the validation set (default 1,000 samples).

---

## 8. Evaluation & Reporting

- **Iteration artifacts:**
	- `results/iteration_XX/evaluation.json` — metrics + sample-level detail.
	- `logs/optimization.log` — chronological log messages.
- **Summary files:**
	- `results/summary.json` — overall configuration, best iteration, validation metrics.
	- `results/validation_evaluation.json` — final validation performance snapshot.
- **Visuals:**
	- `results/metrics_progress.png` — MAE trend across iterations.
- **Prompt history:**
	- `prompts/prompt_iteration_XX.txt` — plain-text snapshots.

When preparing presentation materials, leverage `slides/outline.md` as a backbone and incorporate:

- Initial vs. optimized prompt comparison
- Metric tables (MAE, accuracy within 7.5g, RMSE, correlation)
- Error distribution plots (e.g., seaborn histograms)
- Insight bullets covering improvements and limitations

---

## 9. API Cost Tracking

Monitor usage in parallel with experimentation:

- **Google Gemini:** Google Cloud Console → Billing → Cost Table (filter by your project).
- **OpenAI:** [OpenAI Usage Dashboard](https://platform.openai.com/usage).

Record costs per iteration and include them in your final presentation deck.

---

## 10. Smoke Tests & Troubleshooting

| Issue | Suggested Fix |
| ----- | ------------- |
| `ImportError: No module named src` | Ensure you launch commands from the project root and use `python -m src.optimize` so Python treats `src/` as a package. |
| `LLMError` regarding missing API keys | Confirm the `.env` file contains both keys and that you activated your virtual environment before running scripts. |
| Rate limiting or flaky responses | Re-run the optimization with smaller batch sizes or insert delays using Typer options. Tenacity already retries transient failures. |
| Dataset splits already exist | Either delete `data/train.csv`/`data/val.csv` or rerun `split_data.py` with `--overwrite`. |
| Model outputs non-numeric values | Utilities coerce invalid responses to `0.0` and log the event. Inspect worst-case samples to refine prompts. |

---

## 11. Next Steps & Slide Outline

- **Slides:** Update `slides/outline.md` with real metrics, plots, and API cost screenshots.
- **Further automation:**
	- Batch multiple prompt candidates per iteration (beam search / bandits).
	- Experiment with provider/model combinations by adjusting CLI flags.
	- Add CI workflows (GitHub Actions) to run smoke tests on push.
- **Reproducibility diary:** Maintain a lab-notebook style log under `results/` capturing configuration, costs, and observations per experiment.

For convenience, a starter deck outline is already provided in `slides/outline.md` with required sections listed in the project specification.

---

Happy optimizing! If you encounter issues not covered here, capture the exact error message, check `logs/optimization.log`, and search with relevant keywords—or open an issue with reproduction steps.
