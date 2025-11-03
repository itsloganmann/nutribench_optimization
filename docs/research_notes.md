# Research Notes

> Summaries compiled from prior knowledge and publicly available descriptions. Direct access to external papers or websites was not performed during this session; references should be verified against the original sources.

## Automatic Prompt Optimization (ProTeGi-style)

- **Core idea**: Iteratively refine prompts by leveraging a "textual gradient"—the language model critiques the current prompt using failure cases, then proposes an improved prompt.
- **Workflow**:
  1. Evaluate the prompt on a batch of examples (temperature 0 for stability).
  2. Identify worst-performing examples and request the model to diagnose errors.
  3. Use the critique to generate a new prompt candidate.
  4. Repeat for multiple iterations, selecting the best-performing prompt.
- **Key insights**:
  - Maintaining a memory of past prompts prevents regression and encourages diversity.
  - Gradient signals are textual; well-structured error descriptions lead to better improvements.
  - Beam search or maintaining multiple candidates can mitigate local minima.

## NutriBench Dataset & Benchmark

- **Objective**: Evaluate LLM ability to estimate nutrition facts—particularly carbohydrate grams—from free-form meal descriptions.
- **Structure**:
  - Meal descriptions with ground-truth macronutrient annotations (carb-focused in this project).
  - Variants may include chain-of-thought (CoT) rationales to guide reasoning.
- **Challenges**:
  - Meal descriptions vary in granularity (ingredients, portions, preparation methods).
  - Requires reasoning about serving sizes, conversions, and implicit assumptions.
  - Sensitive to prompt wording that enforces concise numeric outputs.

## NutriBench GitHub Repository Highlights

- Provides preprocessing scripts, baseline prompts, and evaluation utilities.
- Includes documentation for dataset splits (train/validation/test) and evaluation metrics (MAE, RMSE, accuracy within tolerance).
- Often paired with Hugging Face dataset `dongx1997/NutriBench` which exposes the raw annotations.

## Implementation Implications

- **Prompt Design**: Emphasize table-structured reasoning or step-by-step breakdown to improve accuracy.
- **Evaluation Metrics**: Primary metrics—MAE, RMSE, accuracy within ±7.5g—as mandated by the benchmark.
- **Logging**: Persist critiques, revised prompts, and quantitative metrics for reproducibility.
- **Beam Search**: Track top-K prompts per iteration and explore them in parallel to locate stronger optima.

> Re-validate these summaries against the latest official publications for academic or production usage.
