---
applyTo: '**'
---
NutriBench Prompt Optimization Project Copilot Specifications

Project Overview

This repository implements automatic prompt optimization for nutrition estimation using the NutriBench dataset and ProTeGi-style textual gradient descent. The goal is to optimize prompts for Gemini or GPT models to achieve the best possible performance on carbohydrate estimation from meal descriptions.

Environment Setup

System Requirements

Device: MacBook Pro, Apple Silicon or Intel
Operating System: macOS Monterey or later
Python: Version 3.9 or higher, use Homebrew to install if needed
Package Manager: Homebrew

Initial Setup Steps

Update macOS and Xcode Command Line Tools. Run xcode-select --install in Terminal.
Install Homebrew. Run the install command from the Homebrew website in Terminal.
Install Python 3. Run brew install python.
Create a project directory. Run mkdir nutribench_optimization and cd nutribench_optimization.
Set up a Python virtual environment. Run python3 -m venv venv and source venv/bin/activate.
Install required Python packages. Run pip install --upgrade pip and pip install pandas numpy scikit-learn matplotlib seaborn datasets openai python-dotenv google-generativeai.

API Key Management

Google Gemini API Key. Create a Google Cloud account, set up a project, and generate an API key via Google AI Studio.
OpenAI API Key. Create an OpenAI account and generate an API key.
Store both keys in a .env file in the project root. The file should contain GOOGLE_API_KEY=your_google_api_key_here and OPENAI_API_KEY=your_openai_api_key_here.

Data Acquisition and Preparation

Download NutriBench v2_cot

Use Hugging Face datasets to download. In Python, run from datasets import load_dataset, then ds = load_dataset("dongx1997/NutriBench", "default"), then ds["train"].to_csv("nutribench_v2_cot.csv").

Split Data

Use scikit-learn to split into train and validation. In Python, import pandas as pd and from sklearn.model_selection import train_test_split. Load the CSV, then run train, val = train_test_split(df, test_size=1000, random_state=42). Save train and val as train.csv and val.csv in a data folder. Ensure the validation set has at least 1000 samples.

Project Structure

nutribench_optimization
data
train.csv
val.csv
prompts
results
src
optimize.py
utils.py
.env
venv
.github
copilot-instructions.md

Optimization Logic

Initial Prompt

Start with a baseline prompt for carbohydrate estimation. For example, "You are a nutrition expert. Estimate the carbohydrate content in grams for the following meal description. Meal: {meal_description} Provide only the number."

ProTeGi Optimization Loop

Evaluate the prompt by running it on a batch of training data and collecting errors, including mean absolute error and accuracy within 7.5 grams.
Generate a gradient by using the LLM to critique the prompt based on the worst errors.
Edit the prompt by using the LLM to generate an improved prompt using the critique.
Repeat the process for several rounds, keeping track of prompt versions and performance.
For final evaluation, run the best prompt on the validation set with at least 1000 samples.

API Usage

Use Gemini or GPT models for all LLM calls.
Use temperature 0.0 for evaluation and 0.7 for generation.
Track API usage and cost via Google Cloud Console and OpenAI dashboard.

Code Requirements

All scripts must be compatible with macOS and Python 3.9 or higher.
Use environment variables for API keys.
Modularize code by separating data loading, prompt evaluation, gradient generation, and editing.
Log all results and prompt versions for reproducibility.
Include error handling for API failures and missing data.

Evaluation Metrics

Mean Absolute Error, which is the average absolute difference between predicted and actual carbohydrates.
Accuracy within 7.5 grams, which is the percentage of predictions within 7.5 grams of the true value.
Additional metrics include root mean squared error, correlation, and error distribution histograms.

Presentation and Reporting

Create slides covering the project overview and goals, dataset and experimental setup, initial versus optimized prompt comparison, performance metrics and error analysis, insights on prompt improvements, API cost analysis, limitations, and future work.
Include visualizations using matplotlib or seaborn for optimization progress, error distributions, and metric comparisons.

Best Practices

Use random seeds for reproducibility.
Document all hyperparameters and experiment settings.
Save all intermediate results and logs.
Write clear docstrings and comments in all code files.
Use version control with git for all changes.

Troubleshooting

If package installation fails, check Homebrew and Python versions.
If API calls fail, verify keys and network connectivity.
If model outputs are inconsistent, adjust temperature or batch size.
For any errors, consult logs and error messages and search online for solutions.

Completion Criteria

The optimized prompt achieves improved mean absolute error and accuracy over the initial prompt on the validation set.
All code, data splits, and results are reproducible.
Presentation slides are complete and include all required sections.
API cost is tracked and reported.
All steps are documented in README.md and this specification file.

Copilot should follow these instructions exactly. If any step is ambiguous, ask for clarification or provide a safe default. Always prioritize reproducibility, clarity, and adherence to the project requirements.

[1](https://markdowntotext.com)
[2](https://picotoolkit.com/text/markdown-to-text-converter)
[3](https://www.mdtotext.com/en)
[4](https://www.reddit.com/r/perl/comments/1f0faxj/perl_script_to_convert_markdown_to_plain_text/)
[5](https://stripmd.vercel.app)
[6](https://www.w3docs.com/nx/marked)
[7](https://support.google.com/docs/answer/12014036?hl=en)
[8](https://www.switchlabs.dev/markdown-to-richtext)
[9](https://stackoverflow.com/questions/761824/python-how-to-convert-markdown-formatted-text-to-text)
[10](https://www.markdownguide.org/getting-started/)