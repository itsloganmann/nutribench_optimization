# References and Literature Notes

## Automatic Prompt Optimization with "Gradient Descent" and Beam Search (ProTeGi)
- Citation: Zhou, J., Mishra, S. et al. (2023). "Automatic Prompt Optimization with 'Gradient Descent' and Beam Search." *NeurIPS 2023 Workshop on Instruction Tuning*. [OpenReview PDF](https://openreview.net/pdf?id=FQPN4HR7uR)
- Key Ideas:
  - Treats prompt tokens as latent variables optimized via iterative edits derived from language model critiques.
  - Alternates between evaluation on task data, gradient-like textual critiques, and prompt editing using beam search over candidate prompts.
  - Demonstrates improvements across multiple NLP benchmarks and emphasizes reproducible logging of prompt versions.
- Relevance to project:
  - Provides the blueprint for our optimization loop (`evaluate → critique → edit`) implemented in `src/optimize.py`.
  - Motivates tracking mean absolute error and selecting top-k worst samples to form critiques.

## NutriBench: A Dataset for Evaluating Large Language Models on Nutrition Estimation from Meal Descriptions
- Citation: Dong, X. et al. (2024). "NutriBench: A Dataset for Evaluating Large Language Models on Nutrition Estimation from Meal Descriptions." *arXiv preprint arXiv:2407.02450*. [arXiv abstract](https://arxiv.org/abs/2407.02450)
- Key Ideas:
  - Introduces NutriBench, pairing free-form meal descriptions with per-nutrient ground truth, enabling quantitative evaluation of LLM-based nutrition estimators.
  - Reports that vanilla LLM prompts underperform compared to dietitian baselines, highlighting the need for domain-specific prompt optimization.
  - Defines evaluation metrics (MAE, RMSE, accuracy within 7.5 grams) that we adopt directly.
- Relevance to project:
  - Validates our choice of metrics and the requirement to evaluate on ≥1,000 validation samples.
  - Supplies the `nutribench_v2_cot` split we preprocess and use for optimization.

## NutriBench GitHub Benchmark
- Repository: [DongXzz/NutriBench](https://github.com/DongXzz/NutriBench)
- Notes:
  - Contains dataset download instructions, baseline prompting scripts, and evaluation utilities.
  - Provides CSV schema documentation (`meal_description`, `carb`, etc.) that align with our loader in `src/utils.py`.
  - Reference for recommended train/validation splits and reproducibility practices.
- Relevance to project:
  - Ensures our data preprocessing matches the official benchmark format.
  - Serves as an external baseline for comparing optimized prompts.
