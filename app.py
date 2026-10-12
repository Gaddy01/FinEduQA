import spaces
import os
import json
import torch
import torch.nn.functional as F
import gradio as gr

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM


print("PyTorch version:", torch.__version__)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODEL_REPO = "Gaddiel2005/FinEduQA"
HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN secret is missing from the Space settings.")


# ---------------------------------------------------------
# Download OOD detector files
# ---------------------------------------------------------

OOD_ARTIFACT_PATH = hf_hub_download(
    repo_id=MODEL_REPO,
    filename="ood/ood_detector_artifacts.pt",
    token=HF_TOKEN
)

OOD_CONFIG_PATH = hf_hub_download(
    repo_id=MODEL_REPO,
    filename="ood/ood_detector_config.json",
    token=HF_TOKEN
)


# ---------------------------------------------------------
# Load OOD detector configuration
# ---------------------------------------------------------

with open(OOD_CONFIG_PATH, "r") as f:
    ood_config = json.load(f)

SEMANTIC_THRESHOLD = ood_config["semantic_threshold"]
SCOPE_THRESHOLD = ood_config["scope_threshold"]
TOP_K = ood_config["top_k"]
SEMANTIC_MODEL = ood_config["semantic_model"]


# ---------------------------------------------------------
# Load OOD detector artifacts
# ---------------------------------------------------------

ood_artifacts = torch.load(
    OOD_ARTIFACT_PATH,
    map_location="cpu",
    weights_only=False
)

train_embeddings = ood_artifacts["train_embeddings"].float().to("cuda")
in_scope_embeddings = ood_artifacts["in_scope_embeddings"].float().to("cuda")
out_of_scope_embeddings = ood_artifacts["out_of_scope_embeddings"].float().to("cuda")


# ---------------------------------------------------------
# Load MiniLM semantic model
# ---------------------------------------------------------

semantic_tokenizer = AutoTokenizer.from_pretrained(SEMANTIC_MODEL)

semantic_model = AutoModel.from_pretrained(
    SEMANTIC_MODEL
).to("cuda")

semantic_model.eval()


# ---------------------------------------------------------
# Load fine-tuned FLAN-T5
# ---------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_REPO,
    token=HF_TOKEN
)

qa_model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_REPO,
    token=HF_TOKEN
).to("cuda")

qa_model.eval()


# ---------------------------------------------------------
# Mean pooling for MiniLM embeddings
# ---------------------------------------------------------

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

    return torch.sum(
        token_embeddings * mask,
        dim=1
    ) / torch.clamp(mask.sum(dim=1), min=1e-9)


# ---------------------------------------------------------
# Encode user question
# ---------------------------------------------------------

def encode_question(question):
    inputs = semantic_tokenizer(
        question,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    ).to("cuda")

    with torch.no_grad():
        outputs = semantic_model(**inputs)

    embedding = mean_pooling(
        outputs,
        inputs["attention_mask"]
    )

    return F.normalize(embedding, p=2, dim=1)


# ---------------------------------------------------------
# OOD and scope detection
# ---------------------------------------------------------

def classify_question_scope(question):
    question_embedding = encode_question(question)

    similarities = torch.mm(
        question_embedding,
        train_embeddings.T
    )

    top_scores = torch.topk(
        similarities,
        k=TOP_K,
        dim=1
    ).values

    domain_score = top_scores.mean().item()

    if domain_score < SEMANTIC_THRESHOLD:
        return {
            "in_domain": False,
            "reason": "non_financial",
            "domain_score": domain_score,
            "scope_margin": None
        }

    in_scope_score = torch.mm(
        question_embedding,
        in_scope_embeddings.T
    ).max().item()

    out_scope_score = torch.mm(
        question_embedding,
        out_of_scope_embeddings.T
    ).max().item()

    scope_margin = in_scope_score - out_scope_score

    if scope_margin < SCOPE_THRESHOLD:
        return {
            "in_domain": False,
            "reason": "out_of_scope_finance",
            "domain_score": domain_score,
            "scope_margin": scope_margin
        }

    return {
        "in_domain": True,
        "reason": "in_scope",
        "domain_score": domain_score,
        "scope_margin": scope_margin
    }


# ---------------------------------------------------------
# FLAN-T5 answer generation
# ---------------------------------------------------------

def generate_answer(question):
    prompt = "Answer the following financial literacy question: " + question

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=64,
        truncation=True
    ).to("cuda")

    with torch.no_grad():
        outputs = qa_model.generate(
            **inputs,
            num_beams=4,
            early_stopping=True,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            max_new_tokens=256
        )

    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )


# ---------------------------------------------------------
# Complete FinEduQA pipeline
# ---------------------------------------------------------

@spaces.GPU(duration=30)
def financial_assistant(question):
    if question is None:
        return "Please enter a financial-literacy question."

    question = question.strip()

    if not question:
        return "Please enter a financial-literacy question."

    result = classify_question_scope(question)

    if not result["in_domain"]:
        if result["reason"] == "non_financial":
            return (
                "This assistant is designed to answer financial-literacy questions. "
                "Please ask a question related to personal finance, saving, budgeting, "
                "banking, credit, debt, insurance, or beginner investing."
            )

        if result["reason"] == "out_of_scope_finance":
            return (
                "This question is outside the scope of this financial-literacy assistant. "
                "Please ask about personal finance topics such as budgeting, saving, banking, "
                "credit, debt, insurance, or beginner investing."
            )

    return generate_answer(question)


# ---------------------------------------------------------
# Gradio web interface
# ---------------------------------------------------------

demo = gr.Interface(
    fn=financial_assistant,

    inputs=gr.Textbox(
        lines=3,
        label="Financial literacy question",
        placeholder="Example: How can I build an emergency fund?"
    ),

    outputs=gr.Textbox(
        lines=8,
        label="FinEduQA answer"
    ),

    title="FinEduQA",

    description=(
        "FinEduQA is a domain-specific financial-literacy assistant designed for "
        "educational questions about personal finance, including budgeting, saving, "
        "banking, credit, debt, insurance and beginner investing. "
        "Its responses are for educational purposes and should not be treated as "
        "professional financial advice."
    ),

    examples=[
        ["How can I improve my credit score?"],
        ["How much should I keep in an emergency fund?"],
        ["What is an index fund?"]
    ],

    cache_examples=False,
    api_name="ask",
    flagging_mode="never"
)


# ---------------------------------------------------------
# Launch Space
# ---------------------------------------------------------

demo.launch()