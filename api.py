"""
api.py — FastAPI backend for C++ Assignment Evaluator
Run with: uvicorn api:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import json
import re
import os
import time
from pathlib import Path

app = FastAPI(title="C++ Assignment Evaluator API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_PATH   = "data.json"
MODEL_DIR      = "models/cpp_evaluator_lora"
BASE_MODEL     = "unsloth/mistral-7b-instruct-v0.2-bnb-4bit"
MAX_SEQ_LEN    = 1536
MAX_NEW_TOKENS = 128

# ── Global model state ────────────────────────────────────────────────────────
_model      = None
_tokenizer  = None
_model_type = None  # "finetuned" or "base"

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a strict, accurate, and consistent C++ programming instructor evaluating student submissions.

Evaluate the student's submission strictly based on the given rubric and reference solution.

Scoring Guidelines:
- 1: Completely broken — does not compile, or implements almost nothing of what is required.
- 2: Major logical or syntax errors that prevent core requirements from working correctly.
- 3: Partially correct — clearly state what is implemented correctly AND what key parts are missing or broken.
- 4: Mostly correct and runs for standard inputs, but has a minor issue such as a missing edge case, slight inefficiency, or small logic flaw.
- 5: Fully correct — all required functions work correctly for all cases, output is properly formatted, and the implementation follows best practices.

Output exactly these two lines and nothing else:
Score: <1-5>
Rationale: <two to three sentences: name the specific functions or operations that work, identify what is missing or incorrect, and reference the rubric level>"""


def format_rubric(rubric: dict) -> str:
    return "\n".join(f"  {k}: {v}" for k, v in sorted(rubric.items(), key=lambda x: str(x[0])))


def build_user_message(task: str, reference: str, submission: str, rubric: dict) -> str:
    return (
        f"### Task\n{task}\n\n"
        f"### Reference Solution\n```cpp\n{reference}\n```\n\n"
        f"### Student Submission\n```cpp\n{submission}\n```\n\n"
        f"### Rubric\n{format_rubric(rubric)}"
    )


def parse_output(text: str):
    score     = None
    rationale = ""
    m = re.search(r"Score\s*:\s*([1-5])", text, re.IGNORECASE)
    if m:
        score = int(m.group(1))
    m = re.search(r"Rationale\s*:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if m:
        sentences = re.split(r"(?<=[.!?])\s+", m.group(1).strip())
        rationale = " ".join(sentences[:3])
    if score is None:
        m = re.search(r"\b([1-5])\b", text)
        if m:
            score = int(m.group(1))
    return score, rationale


def load_dataset():
    if not Path(DATASET_PATH).exists():
        return []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_model_internal(model_type: str = "finetuned"):
    """Internal helper — loads model into globals. Called at startup."""
    global _model, _tokenizer, _model_type
    from unsloth import FastLanguageModel

    print(f"[startup] Loading base model: {BASE_MODEL}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=True,
    )

    if model_type == "finetuned" and Path(MODEL_DIR).exists():
        from peft import PeftModel
        print(f"[startup] Merging LoRA adapter from: {MODEL_DIR}")
        model = PeftModel.from_pretrained(model, MODEL_DIR)
        model = model.merge_and_unload()
        _model_type = "finetuned"
        print("[startup] Fine-tuned model ready ✓")
    else:
        if model_type == "finetuned":
            print(f"[startup] WARNING: adapter not found at {MODEL_DIR}, using base model")
        _model_type = "base"
        print("[startup] Base model ready ✓")

    FastLanguageModel.for_inference(model)
    _model     = model
    _tokenizer = tokenizer


# ── Auto-load model on startup ────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """Load the fine-tuned model automatically when the API starts."""
    try:
        _load_model_internal("finetuned")
    except Exception as e:
        print(f"[startup] ERROR loading model: {e}")
        print("[startup] API will run without a model — /evaluate will return an error.")


# ── Pydantic models ───────────────────────────────────────────────────────────
class EvaluateRequest(BaseModel):
    task:       str
    submission: str
    reference:  Optional[str] = None
    rubric:     Optional[dict] = None


class EvaluateResponse(BaseModel):
    score:           Optional[int]
    rationale:       str
    raw_output:      str
    model_used:      str
    elapsed_seconds: float


class TaskInfo(BaseModel):
    task:      str
    reference: str
    rubric:    dict


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "status":  "ok",
        "message": "C++ Evaluator API running",
        "model":   _model_type or "not loaded",
    }


@app.get("/health")
def health():
    return {
        "status":       "ok",
        "model_loaded": _model is not None,
        "model_type":   _model_type,
    }


@app.get("/tasks", response_model=list[TaskInfo])
def get_tasks():
    data = load_dataset()
    seen = {}
    for entry in data:
        task = entry["task"]
        if task not in seen:
            seen[task] = {"task": task, "reference": entry["reference"], "rubric": entry["rubric"]}
    return list(seen.values())


@app.get("/model/status")
def model_status():
    return {
        "loaded":              _model is not None,
        "type":                _model_type,
        "finetuned_available": Path(MODEL_DIR).exists(),
    }


@app.post("/model/load")
def load_model(model_type: str = "finetuned"):
    """Manually reload the model (e.g. to switch base ↔ finetuned)."""
    try:
        _load_model_internal(model_type)
        return {"status": "loaded", "type": _model_type}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate(req: EvaluateRequest):
    """Evaluate a student submission using the loaded fine-tuned model."""

    # Reject immediately if model not loaded — no mock fallback
    if _model is None or _tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model is not loaded. The API is still starting up, or the model "
                "failed to load. Check server logs and try again in a moment."
            ),
        )

    # Fetch reference + rubric from dataset if not provided
    reference = req.reference
    rubric    = req.rubric

    if reference is None or rubric is None:
        data = load_dataset()
        for entry in data:
            if entry["task"] == req.task:
                reference = reference or entry["reference"]
                rubric    = rubric    or entry["rubric"]
                break

    if reference is None:
        raise HTTPException(status_code=400, detail="Reference solution not found for this task.")
    if rubric is None:
        raise HTTPException(status_code=400, detail="Rubric not found for this task.")

    # ── Run inference ─────────────────────────────────────────────────────────
    import torch

    user_msg = build_user_message(req.task, reference, req.submission, rubric)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg},
    ]

    try:
        prompt = _tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt = f"<s>[INST] {SYSTEM_PROMPT}\n\n{user_msg} [/INST]"

    t0     = time.time()
    inputs = _tokenizer(prompt, return_tensors="pt").to(_model.device)

    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=_tokenizer.eos_token_id,
        )

    new_ids    = output_ids[0][inputs["input_ids"].shape[1]:]
    raw_output = _tokenizer.decode(new_ids, skip_special_tokens=True)
    elapsed    = round(time.time() - t0, 2)

    score, rationale = parse_output(raw_output)

    return EvaluateResponse(
        score           = score,
        rationale       = rationale,
        raw_output      = raw_output,
        model_used      = _model_type,
        elapsed_seconds = elapsed,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)