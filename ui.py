import streamlit as st
import requests
import json
import time
from collections import defaultdict

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="C++ Evaluator",
    page_icon="💻",
    layout="wide",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark background */
    .stApp { background-color: #0e1117; }

    /* Metric cards */
    .metric-card {
        background: #1a1d27;
        border-radius: 12px;
        padding: 18px 22px;
        border: 1px solid #2a2d3a;
        text-align: center;
    }
    .metric-label {
        color: #8b92a5;
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        line-height: 1;
    }
    .metric-sub {
        color: #8b92a5;
        font-size: 12px;
        margin-top: 4px;
    }

    /* Score bar */
    .score-bar-wrap {
        background: #1a1d27;
        border-radius: 12px;
        padding: 20px 24px;
        border: 1px solid #2a2d3a;
        margin-bottom: 16px;
    }
    .score-row {
        display: flex;
        align-items: center;
        margin-bottom: 8px;
        gap: 10px;
    }
    .score-label { color: #ccc; font-size: 13px; width: 18px; text-align: right; flex-shrink: 0; }
    .bar-bg { flex: 1; background: #2a2d3a; border-radius: 4px; height: 14px; overflow: hidden; }
    .bar-fill { height: 100%; border-radius: 4px; }
    .bar-count { color: #8b92a5; font-size: 12px; width: 28px; text-align: right; flex-shrink: 0; }

    /* Result card */
    .result-card {
        padding: 20px 24px;
        border-radius: 12px;
        background: #1a1d27;
        border: 1px solid #2a2d3a;
    }
    .result-score {
        font-size: 42px;
        font-weight: 800;
        line-height: 1;
    }
    .result-label {
        font-size: 16px;
        font-weight: 500;
        margin-top: 4px;
    }
    .result-rationale {
        color: #c5c9d5;
        font-size: 14px;
        line-height: 1.6;
        margin-top: 14px;
        padding-top: 14px;
        border-top: 1px solid #2a2d3a;
    }

    /* Rubric table */
    .rubric-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
    }
    .rubric-table td {
        padding: 8px 12px;
        border-bottom: 1px solid #2a2d3a;
        color: #c5c9d5;
        vertical-align: top;
    }
    .rubric-table td:first-child {
        font-weight: 700;
        width: 30px;
        text-align: center;
    }

    /* Sidebar */
    [data-testid="stSidebar"] { background: #13151f; border-right: 1px solid #2a2d3a; }

    /* Divider */
    hr { border-color: #2a2d3a !important; }

    /* Code area */
    .stTextArea textarea {
        font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace;
        font-size: 13px;
        background: #1a1d27;
        color: #e2e8f0;
        border: 1px solid #2a2d3a;
        border-radius: 8px;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #4f6ef7, #7c3aed);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        font-size: 15px;
        padding: 10px 28px;
        width: 100%;
    }
    .stButton > button:hover { opacity: 0.88; }

    /* Hide default streamlit header */
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SCORE CONFIG
# ─────────────────────────────────────────────
SCORE_LABELS = {
    1: "Completely Broken",
    2: "Major Errors",
    3: "Partially Correct",
    4: "Mostly Correct",
    5: "Fully Correct",
}
SCORE_COLORS = {
    1: "#e24b4a",
    2: "#ef9f27",
    3: "#378add",
    4: "#1d9e75",
    5: "#22c55e",
}
SCORE_BAR_COLORS = {
    1: "#e24b4a",
    2: "#ef9f27",
    3: "#378add",
    4: "#1d9e75",
    5: "#22c55e",
}

# ─────────────────────────────────────────────
# LOAD DATASET
# ─────────────────────────────────────────────
@st.cache_data
def load_dataset():
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, "data.json")
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Dataset not found at: {path}")
        return []

data = load_dataset()

# Build unique task list (preserving first occurrence order)
seen = set()
unique_tasks = []
for item in data:
    if item["task"] not in seen:
        seen.add(item["task"])
        unique_tasks.append(item)

# Per-task stats
task_score_map = defaultdict(list)
for item in data:
    task_score_map[item["task"]].append(item["score"])

# ─────────────────────────────────────────────
# OVERALL DATASET METRICS
# ─────────────────────────────────────────────
all_scores = [item["score"] for item in data]
total_submissions = len(all_scores)
avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
accuracy_pct = (sum(1 for s in all_scores if s >= 4) / total_submissions * 100) if total_submissions else 0
score_dist = {i: all_scores.count(i) for i in range(1, 6)}
max_dist = max(score_dist.values()) if score_dist else 1

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 Task Selector")

    task_titles = [t["task"] for t in unique_tasks]

    selected_index = st.selectbox(
        "Choose a task",
        range(len(task_titles)),
        format_func=lambda i: task_titles[i],
        index=0,
        label_visibility="collapsed",
    )
    selected_task = unique_tasks[selected_index]
    selected_title = selected_task["task"]

    st.markdown("---")

    # Per-task mini stats in sidebar
    task_scores = task_score_map[selected_title]
    task_avg = sum(task_scores) / len(task_scores) if task_scores else 0
    task_accuracy = sum(1 for s in task_scores if s >= 4) / len(task_scores) * 100 if task_scores else 0

    st.markdown("---")
    st.markdown("**Score Legend**")
    for score, label in SCORE_LABELS.items():
        color = SCORE_COLORS[score]
        st.markdown(
            f'<span style="color:{color}; font-weight:700">●</span> '
            f'<span style="color:#ccc; font-size:13px">{score} — {label}</span>',
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────
# MAIN — HEADER
# ─────────────────────────────────────────────
st.markdown("# 💻 C++ Code Evaluator")
st.markdown(
    '<p style="color:#8b92a5; margin-top:-10px; font-size:15px;">'
    "Submit your C++ solution and get instant AI-powered evaluation.</p>",
    unsafe_allow_html=True,
)

st.markdown("---")

# ─────────────────────────────────────────────
# EVALUATION METRICS DASHBOARD (top)
# ─────────────────────────────────────────────
st.markdown("### 📊 Dataset Evaluation Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""<div class="metric-card">
            <div class="metric-label">Total Submissions</div>
            <div class="metric-value" style="color:#4f6ef7">{total_submissions}</div>
            <div class="metric-sub">{len(unique_tasks)} unique tasks</div>
        </div>""",
        unsafe_allow_html=True,
    )

with col2:
    avg_color = SCORE_COLORS.get(round(avg_score), "#8b92a5")
    st.markdown(
        f"""<div class="metric-card">
            <div class="metric-label">Average Score</div>
            <div class="metric-value" style="color:{avg_color}">{avg_score:.2f}<span style="font-size:18px;color:#8b92a5"> / 5</span></div>
            <div class="metric-sub">{SCORE_LABELS.get(round(avg_score), '')}</div>
        </div>""",
        unsafe_allow_html=True,
    )

with col3:
    acc_color = "#22c55e" if accuracy_pct >= 50 else "#ef9f27" if accuracy_pct >= 30 else "#e24b4a"
    st.markdown(
        f"""<div class="metric-card">
            <div class="metric-label">Accuracy (Score ≥ 4)</div>
            <div class="metric-value" style="color:{acc_color}">{accuracy_pct:.1f}<span style="font-size:18px;color:#8b92a5">%</span></div>
            <div class="metric-sub">{sum(1 for s in all_scores if s >= 4)} passing submissions</div>
        </div>""",
        unsafe_allow_html=True,
    )

with col4:
    perfect = score_dist.get(5, 0)
    st.markdown(
        f"""<div class="metric-card">
            <div class="metric-label">Perfect Scores (5/5)</div>
            <div class="metric-value" style="color:#22c55e">{perfect}</div>
            <div class="metric-sub">{perfect/total_submissions*100:.1f}% of submissions</div>
        </div>""",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TASK DETAIL
# ─────────────────────────────────────────────
st.markdown(f"### 📝 Task")
st.markdown(
    f'<div style="background:#1a1d27;border-radius:10px;padding:16px 20px;'
    f'border-left:4px solid #4f6ef7;color:#e2e8f0;font-size:15px;line-height:1.6;">'
    f'{selected_task["task"]}</div>',
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)

# Two columns: rubric + code editor
left_col, right_col = st.columns([1, 2])

with left_col:
    st.markdown("#### 📋 Scoring Rubric")
    rubric = selected_task.get("rubric", {})
    rubric_rows = ""
    for score_key in ["5", "4", "3", "2", "1"]:
        desc = rubric.get(score_key, "—")
        color = SCORE_COLORS.get(int(score_key), "#888")
        rubric_rows += (
            f'<tr><td style="color:{color}">{score_key}</td>'
            f'<td>{desc}</td></tr>'
        )
    st.markdown(
        f'<div style="background:#1a1d27;border-radius:10px;padding:16px;border:1px solid #2a2d3a;">'
        f'<table class="rubric-table"><tbody>{rubric_rows}</tbody></table></div>',
        unsafe_allow_html=True,
    )

    # Reference solution toggle
    with st.expander("🔍 View Reference Solution"):
        st.code(selected_task.get("reference", "No reference available."), language="cpp")

with right_col:
    st.markdown("#### 🖊️ Code Editor")
    code = st.text_area(
        "Write your C++ solution here",
        height=380,
        value="#include <iostream>\nusing namespace std;\n\nint main() {\n    // Your solution here\n    \n    return 0;\n}\n",
        label_visibility="collapsed",
    )

    submitted = st.button("🚀 Submit & Evaluate")

# ─────────────────────────────────────────────
# EVALUATION RESULT
# ─────────────────────────────────────────────

# Check API health before submitting
def api_is_up():
    for base in ["http://127.0.0.1:8000", "http://localhost:8000"]:
        for endpoint in ["/health", "/"]:
            try:
                r = requests.get(base + endpoint, timeout=2)
                if r.ok:
                    return True
            except Exception:
                continue
    return False

if submitted:
    if not api_is_up():
        st.markdown("---")
        st.error(
            "🔌 **Model API is offline.**  "
            "Your code cannot be evaluated right now.  \n\n"
            "Start your backend server (`uvicorn main:app --reload`) and try again.",
        )
    else:
        with st.spinner("🤖 Evaluating your submission…"):
            start = time.time()
            result = None
            error_msg = None

            try:
                res = requests.post(
                    "http://127.0.0.1:8000/evaluate",
                    json={
                        "task": selected_task["task"],
                        "submission": code,
                        "reference": selected_task.get("reference"),
                    },
                    timeout=30,
                )
                if res.ok:
                    result = res.json()
                else:
                    error_msg = f"Server returned {res.status_code}: {res.text[:200]}"
            except Exception as e:
                error_msg = str(e)

            elapsed = round(time.time() - start, 2)

        st.markdown("---")
        st.markdown("### 📊 Evaluation Result")

        if error_msg:
            st.error(f"❌ Evaluation failed: {error_msg}")
        elif result:
            score = result.get("score", 0)
            color = SCORE_COLORS.get(score, "#888")
            label = SCORE_LABELS.get(score, "")
            rationale = result.get("rationale", "")

            st.markdown(
                f"""<div class="result-card" style="border-left: 5px solid {color};">
                    <div style="display:flex;align-items:center;gap:16px;">
                        <div>
                            <div class="result-score" style="color:{color}">{score}<span style="font-size:22px;color:#8b92a5">/5</span></div>
                            <div class="result-label" style="color:{color}">{label}</div>
                        </div>
                        <div style="flex:1;">
                            <div style="background:#2a2d3a;border-radius:6px;height:10px;overflow:hidden;">
                                <div style="width:{score/5*100}%;height:100%;background:{color};border-radius:6px;"></div>
                            </div>
                            <div style="color:#8b92a5;font-size:12px;margin-top:5px;">{score/5*100:.0f}% score</div>
                        </div>
                    </div>
                    <div class="result-rationale">{rationale}</div>
                </div>""",
                unsafe_allow_html=True,
            )