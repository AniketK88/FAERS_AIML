# AET-SE: Live Demo Script

**Total Demo Time:** ~3 Minutes  
**Goal:** Showcase the end-to-end multi-agent pharmacovigilance pipeline and the Streamlit signal dashboard.

---

### Step 1: Architecture Overview (0:00 - 0:30)
**Action:** Open `README.md` and display the Mermaid architecture diagram.
**Talking Points:**
- "AET-SE is a local-first signal detection engine for pharmacovigilance."
- "We ingested over 770,000 FDA FAERS cases using Polars and computed robust PRR (Proportional Reporting Ratio) baselines in DuckDB."
- "For new, unstructured drug reviews, we built a LangGraph AI pipeline."
- "The local Llama 3.1 model extracts drugs and reactions, a confidence scorer evaluates the extraction, and a ChromaDB vector database maps lay terminology to standardized MedDRA terms."

### Step 2: Live Pipeline Execution (0:30 - 1:15)
**Action:** Open a terminal and run a single adversarial review live through the pipeline.
**Command:**
```bash
python -c "
from aetse.pipeline.runner import run_pipeline
text = 'I have been taking ibuprofen for my back pain. Last week I developed gastrointestinal haemorrhage and was hospitalized.'
result = run_pipeline('demo-001', text, source='review')
print(f'Extracted Drugs: {result[\"extracted_drugs\"]}')
print(f'Signal Flag: {result[\"signal_flag\"]}')
"
```
**Talking Points:**
- "Let's push a raw patient review through our pipeline in real-time."
- *Execute command*
- "Notice how the agent correctly isolates 'ibuprofen', maps 'gastrointestinal haemorrhage', checks the DuckDB statistical baseline, and triggers a `medium` or `high` alert flag instantly."

### Step 3: Streamlit Dashboard — Validation (1:15 - 2:00)
**Action:** Launch the dashboard (`make run-app`) and navigate to **Tab 3: System Metrics**.
**Talking Points:**
- "Let's switch to the UI used by pharmacovigilance teams."
- "Scroll down to our Positive Control validation table."
- "We tested the system against 5 historically proven drug-reaction pairs (e.g., Rofecoxib & Myocardial infarction). The system successfully caught 5 out of 5, assigning them robust PRR scores."

### Step 4: Agent Trace Transparency (2:00 - 2:30)
**Action:** Navigate to **Tab 2: Signal Analysis** and select the ibuprofen review from the dropdown.
**Talking Points:**
- "When our system flags a signal, we want total transparency into *why*."
- "Looking at the Agent Trace timeline, you can see the exact millisecond the extraction completed, how the confidence threshold was evaluated, and how the similarity router mapped the reaction to a MedDRA PT."

### Step 5: Signal Heatmap (2:30 - 3:00)
**Action:** Navigate to **Tab 1: Overview** and filter the heatmap down to `ibuprofen` and `rofecoxib`.
**Talking Points:**
- "Finally, teams can monitor aggregate trends using the live PRR Signal Heatmap."
- "The darker the red, the stronger the statistical association compared to background rates."
- "This tool bridges the gap between massive historical FDA databases and incoming, unstructured real-world patient data."
