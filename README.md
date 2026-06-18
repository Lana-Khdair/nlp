# 💻 C++ Assignment Evaluator

An AI-powered tool that automatically evaluates C++ student submissions using a fine-tuned Mistral-7B model (LoRA adapter via Unsloth). Built with a FastAPI backend and Streamlit frontend.

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| Exact Accuracy | 77.91% |
| Off-by-One Accuracy | 95.71% |
| MAE | 26.38% |
| QWK | 91.78% |

**Base model:** `unsloth/mistral-7b-instruct-v0.2-bnb-4bit`  
**LoRA adapter:** [`lana-4/cpp-evaluator-lora`](https://huggingface.co/lana-4/cpp-evaluator-lora)

---

## 🗂️ Project Structure

```
nlp/
├── api.py          # FastAPI backend — loads model, handles /evaluate
├── ui.py           # Streamlit frontend
├── pipeline.py     # Training pipeline (prepare → baseline → finetune → evaluate → analyze)
├── data.json       # Dataset (task, reference, submission, rubric, score, rationale)
├── cpp-evaluator-lora.ipynb 
├── data/
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl
└── results/
    ├── baseline_predictions.json
    ├── finetuned_predictions.json
    ├── training_log.json
    └── final_report.txt
```

---

## ⚙️ Requirements

- Python 3.10+
- CUDA GPU (tested on T4 and A100)
- See dependencies: `unsloth`, `fastapi`, `uvicorn`, `streamlit`, `peft`, `transformers`, `trl`

Install:
```bash
pip install unsloth fastapi uvicorn streamlit peft transformers trl datasets scikit-learn
```

## ☁️ Running on Google Colab

### 1. Run the Colab notebook with all cells and open the interface link

### 2. Start the backend
```python
    uvicorn api:app --port 8000
```

