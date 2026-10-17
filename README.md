# FinEduQA

**FinEduQA** is a domain-specific financial literacy question-answering assistant designed to help university students and young adults understand general personal-finance concepts.

The system is built by fine-tuning `google/flan-t5-base` on a cleaned and filtered subset of the FiQA financial question-answering dataset. It also includes a semantic out-of-domain and scope detection layer that prevents the generative model from responding to clearly non-financial questions or advanced financial topics outside the intended educational scope.

> **Important:** FinEduQA is an educational system. Its responses may contain inaccuracies and should not be treated as professional financial, investment, legal, or tax advice.

---

## Live Application

**FinEduQA Web Application:**  
https://huggingface.co/spaces/Gaddiel2005/FinEduQA-API

The application is deployed using **Gradio on Hugging Face Spaces with ZeroGPU**.

Users can enter a financial-literacy question directly through the web interface and receive a generated response.

Example questions:

- How can I improve my credit score?
- How much should I keep in an emergency fund?
- What is an index fund?
- How can I create a monthly budget?

The application also rejects questions that are outside its intended domain.

---

## Demo Video

**Demo video:** [https://drive.google.com/file/d/1Ia5XGhuSXsT613i7ILCNaVQx_b7Jy5Cj/view?usp=sharing](https://drive.google.com/file/d/1Ia5XGhuSXsT613i7ILCNaVQx_b7Jy5Cj/view?usp=sharing)

---

## Project Objective

The project investigates whether domain-specific fine-tuning can improve the ability of a pretrained sequence-to-sequence Transformer to answer personal-finance questions relevant to students and young adults.

The main research question is:

> **To what extent does domain-specific fine-tuning improve the ability of a pretrained sequence-to-sequence Transformer to answer personal-finance questions relevant to university students?**

The project includes:

- data preparation and filtering;
- exploratory analysis;
- a pretrained FLAN-T5 baseline;
- three fine-tuning experiments;
- quantitative and qualitative evaluation;
- manual error analysis;
- out-of-domain evaluation;
- semantic scope detection;
- and deployment through an interactive web application.

---

## Intended Scope

FinEduQA focuses on general financial-literacy topics such as:

- personal budgeting and everyday expenses;
- saving and emergency funds;
- bank accounts and basic banking;
- credit and credit scores;
- debt and loan repayment;
- student loans and education expenses;
- mobile money and digital payments;
- scams, fraud, and financial safety;
- beginner investing concepts;
- basic insurance;
- and personal financial planning.

The deployed system is intentionally not designed for topics such as:

- specific stock recommendations;
- day trading;
- advanced derivatives or options strategies;
- short selling;
- cryptocurrency speculation;
- corporate accounting;
- corporate or jurisdiction-specific taxation;
- leveraged buyouts;
- advanced valuation;
- and complex investment-banking questions.

---

## Dataset

The project uses the **FiQA financial question-answering dataset**, accessed through Hugging Face:

https://huggingface.co/datasets/LLukas22/fiqa

The original dataset contained:

- **17,072 question-answer pairs**
- **6,646 unique questions**

The data was cleaned and filtered to better match the intended financial-literacy domain.

The final dataset contains:

- **6,019 question-answer pairs**
- **1,991 unique questions**

### Final Data Split

| Split | QA Pairs | Unique Questions |
|---|---:|---:|
| Training | 4,852 | 1,592 |
| Validation | 578 | 199 |
| Test | 589 | 200 |
| **Total** | **6,019** | **1,991** |

Splitting was performed by **unique question rather than individual row**. This prevents the same question from appearing in multiple splits when it has several valid answers.

The final split has zero question overlap between training, validation, and test sets.

---

## Data Preparation

The data-preparation pipeline includes:

1. combining the original FiQA splits;
2. basic structural and quality auditing;
3. whitespace cleaning while preserving the meaning, punctuation, and capitalization of the text;
4. filtering very short or incomplete responses;
5. heuristic topic filtering;
6. removal of advanced or unsuitable financial topics;
7. removal of jurisdiction-specific questions;
8. removal of short incomplete responses;
9. grouped train/validation/test splitting by unique question;
10. and verification that there is no question leakage between splits.

Because domain filtering is heuristic, some imperfect or borderline examples may remain in the dataset. This is considered a limitation of the study.

---

## Model

The base language model is:

**Google FLAN-T5 Base**

https://huggingface.co/google/flan-t5-base

FLAN-T5 is used as a sequence-to-sequence question-answering model.

The model receives prompts in the following form:

```text
Answer the following financial literacy question: <question>
```

Important preprocessing parameters:

```text
Maximum input length: 64 tokens
Maximum target length: 512 tokens
```

The final generation configuration uses:

```text
Beam search: 4 beams
Repetition penalty: 1.2
No-repeat n-gram size: 3
Maximum new tokens: 256
```

---

## Experiments

A pretrained FLAN-T5 baseline and three fine-tuning experiments were investigated.

| Model | Main Change | Manual Validation Loss | Selected |
|---|---|---:|---|
| Baseline | No fine-tuning | 3.3664 | No |
| Experiment 1 | LR = 5e-5 | **3.2012** | No |
| Experiment 2 | LR = 1e-5 | 3.2396 | **Yes** |
| Experiment 3 | One representative answer per unique question | 3.2570 | No |

Although Experiment 1 produced the lowest validation loss, qualitative inspection showed unstable and repetitive generations.

Experiment 2 produced slightly higher validation loss but more coherent and stable responses and was therefore selected as the final model.

Experiment 3 tested whether reducing multiple answers per question to one representative answer would improve generation quality. It did not outperform the selected model.

---

## Final Test Results

The pretrained baseline and selected fine-tuned model were evaluated on the same **589 held-out test question-answer pairs** using identical decoding settings.

| Metric | Baseline | Fine-tuned Experiment 2 | Absolute Improvement |
|---|---:|---:|---:|
| ROUGE-1 | 0.0476 | **0.2242** | +0.1766 |
| ROUGE-2 | 0.0088 | **0.0459** | +0.0372 |
| ROUGE-L | 0.0379 | **0.1297** | +0.0918 |
| Semantic Similarity | 0.2871 | **0.4587** | +0.1716 |

The fine-tuned model substantially improves both lexical overlap and semantic similarity relative to the pretrained baseline.

However, automatic metrics do not guarantee factual correctness.

Manual inspection identified cases where responses were:

- vague;
- incomplete;
- repetitive;
- factually inaccurate;
- or semantically related to the reference while still failing to answer the question correctly.

This demonstrates the importance of combining automatic metrics with qualitative error analysis.

---

## Out-of-Domain Detection

Initial testing showed that the fine-tuned FLAN-T5 model would attempt to answer questions even when they were unrelated to financial literacy or outside the intended scope.

A separate semantic guardrail was therefore developed.

The detector uses:

**Sentence Transformers MiniLM**

```text
sentence-transformers/all-MiniLM-L6-v2
```

Two checks are applied before generation.

### 1. Financial-domain detection

The question embedding is compared with embeddings of unique training questions.

The domain score is the mean cosine similarity of the top five nearest training questions.

```text
Semantic threshold = 0.40
Top K = 5
```

Questions below this threshold are rejected as non-financial.

### 2. Financial-scope detection

Questions that pass the first stage are compared with semantic descriptions of intended in-scope and out-of-scope financial topics.

The scope margin is:

```text
maximum in-scope similarity - maximum out-of-scope similarity
```

The selected threshold is:

```text
Scope threshold = -0.20
```

Questions below this value are classified as out-of-scope finance.

---

## OOD Evaluation

The final detector was evaluated using:

- **200 unique held-out test questions** as in-domain examples;
- **10 new non-financial questions**;
- **10 new advanced/out-of-scope financial questions**.

Results:

| Metric | Result |
|---|---:|
| In-domain test acceptance | **98%** |
| Overall OOD rejection | **95%** |
| Non-financial rejection | **100%** |
| Advanced-finance OOD rejection | **90%** |

These held-out examples were not used to retune the detector thresholds.

---

## System Architecture

The deployed system follows this pipeline:

```text
User Question
      |
      v
Semantic Domain Detector
      |
      +---- Non-financial ------> Reject
      |
      v
Financial Scope Detector
      |
      +---- Out of scope -------> Reject
      |
      v
Fine-tuned FLAN-T5
      |
      v
Generated Financial-Literacy Response
```

The deployed application uses:

- **Gradio** for the web interface;
- **Hugging Face Spaces** for hosting;
- **ZeroGPU** for GPU inference;
- **FLAN-T5 Base** for answer generation;
- **MiniLM embeddings** for semantic scope detection;
- and Hugging Face Hub for model storage.

---

## Repository Structure

```text
FinEduQA/
├── README.md
├── requirements.txt
├── app.py
│
├── notebooks/
│   ├── 01_fiqa_data_preparation.ipynb
│   ├── 02_flan_t5_modeling_final.ipynb
│   └── 03_huggingface_deployment.ipynb
│
├── data/
│   ├── fiqa_train.csv
│   ├── fiqa_validation.csv
│   └── fiqa_test.csv
│
├── artifacts/
│   ├── ood_detector_artifacts.pt
│   └── ood_detector_config.json
│
└── results/
    ├── final_baseline_test_predictions.csv
    ├── final_exp2_test_predictions.csv
    ├── final_model_metrics.csv
    ├── final_ood_metrics.csv
    ├── ood_heldout_results.csv
    └── ood_in_domain_test_results.csv
```

---

## Reproducing the Project

Clone the repository:

```bash
git clone https://github.com/Gaddy01/FinEduQA.git
cd FinEduQA
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it.

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The main workflow is documented in the notebooks.

### Data preparation

Run:

```text
notebooks/01_fiqa_data_preparation.ipynb
```

This notebook loads the original FiQA dataset, performs filtering and preprocessing, and creates the train, validation, and test datasets.

### Modeling and evaluation

Run:

```text
notebooks/02_flan_t5_modeling_final.ipynb
```

This notebook contains:

- pretrained baseline evaluation;
- tokenization;
- FLAN-T5 fine-tuning;
- controlled experiments;
- validation-loss comparison;
- final test evaluation;
- ROUGE evaluation;
- semantic-similarity evaluation;
- manual error analysis;
- OOD detector development;
- held-out OOD evaluation;
- and artifact generation.

Training was originally carried out in GPU notebook environments, primarily using NVIDIA T4 GPUs.

### Deployment

Deployment is documented in:

```text
notebooks/03_huggingface_deployment.ipynb
```

The production application is hosted on Hugging Face Spaces:

https://huggingface.co/spaces/Gaddiel2005/FinEduQA-API

The trained model weights are hosted separately on Hugging Face rather than committed to GitHub because of their size.

---

## Results Files

The `results/` directory contains the outputs used for final evaluation.

### `final_model_metrics.csv`

Comparison between the pretrained baseline and selected fine-tuned model.

### `final_baseline_test_predictions.csv`

Predictions produced by the original pretrained FLAN-T5 model on the final test set.

### `final_exp2_test_predictions.csv`

Predictions produced by the selected fine-tuned model on the same test set.

### `final_ood_metrics.csv`

Summary of final out-of-domain detector performance.

### `ood_in_domain_test_results.csv`

Detailed detector results for held-out in-domain questions.

### `ood_heldout_results.csv`

Detailed results for the final held-out OOD evaluation set.

---

## Limitations

FinEduQA has several important limitations.

### Dataset limitations

FiQA contains naturally occurring financial discussions, but the original dataset was not specifically created for students or young adults. Domain adaptation therefore relied partly on heuristic filtering.

The data is also not Rwanda-specific, so the assistant should not be used for jurisdiction-specific financial, legal, or tax guidance.

### Generative-model limitations

The fine-tuned model can still generate:

- inaccurate information;
- generic responses;
- incomplete answers;
- repetition;
- and hallucinated statements.

A high ROUGE or semantic-similarity score does not necessarily imply a factually correct response.

### OOD detector limitations

The semantic guardrail substantially improves scope control but is not perfect.

Some borderline financial questions may be incorrectly rejected, while some sophisticated out-of-scope questions may occasionally pass the detector.

### Deployment limitations

The application relies on Hugging Face Spaces and available ZeroGPU capacity. Startup and inference time can therefore vary depending on resource availability.

---

## Responsible Use

FinEduQA is intended only for **financial education and general financial-literacy questions**.

It should not be used as a substitute for:

- a licensed financial adviser;
- an investment professional;
- a lawyer;
- a tax professional;
- or another qualified financial-services provider.

Users should independently verify important financial information before making financial decisions.

---

## Technologies

The project uses:

- Python
- PyTorch
- Hugging Face Transformers
- FLAN-T5
- Hugging Face Datasets
- MiniLM
- pandas
- NumPy
- ROUGE
- cosine semantic similarity
- Gradio
- Hugging Face Hub
- Hugging Face Spaces / ZeroGPU

---

## Acknowledgements

This project uses:

- the **FiQA** financial question-answering dataset;
- **Google FLAN-T5 Base** as the pretrained sequence-to-sequence model;
- **sentence-transformers/all-MiniLM-L6-v2** for semantic embeddings;
- and open-source libraries from Hugging Face, PyTorch, Gradio, pandas, NumPy, and related projects.

The project-specific data filtering, experimental design, model fine-tuning, evaluation, error analysis, OOD detector, and deployment pipeline were developed as part of the **Machine Learning Techniques I Summative Project**.

---

## Author

**Gaddiel Irakoze**

Bachelor of Software Engineering  
African Leadership University

GitHub: https://github.com/Gaddy01
