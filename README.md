# VeriSight AI
## Multimodal Evidence Review & Claim Intelligence

VeriSight AI is an AI-powered evidence review dashboard for insurance and damage claim verification involving cars, laptops, and packages. The platform inspects damage descriptions against photographic evidence, providing adjusters with an auditable, explainable verdict: **SUPPORTED**, **CONTRADICTED**, or **INSUFFICIENT_EVIDENCE**.

---

## 📌 Project Overview & Features

- **Multimodal Evidence Intake**: Submit claim descriptions alongside multiple evidence photographs (up to 8 images per claim).
- **Automated Claim Tracking**: Generates unique, auditable claim IDs (`CLM-MMDDHHMM-XXXX`).
- **Interactive Evidence Gallery**: Inspect uploaded photographs with stable identifiers (`IMG_01`, `IMG_02`, etc.), displaying per-image findings and highlighting images cited by the AI.
- **Explainable Assessments**:
  - **Verdict Badge**: Clearly differentiates between `SUPPORTED`, `CONTRADICTED`, and `INSUFFICIENT_EVIDENCE`.
  - **Damage & Severity Level**: Assesses damage type, affected component, and severity (`LOW`, `MEDIUM`, `HIGH`, `UNKNOWN`).
  - **Confidence Indicator**: Real-time confidence gauge (clearly labelled as an AI estimate, not a calibrated probability).
  - **Risk Indicators**: Highlights anomalies, potential inconsistencies, or items requiring human adjuster attention without prematurely alleging fraud.
  - **Missing Evidence Recommendations**: Actionable suggestions for supplementary photographs or diagnostic documents.
- **Audit & Export**: Download full analysis records as structured JSON reports.
- **Session Dashboard**: Live tracking of claims reviewed, supported claims, and items requiring human review across the active session.
- **Fail-Safe Demo Mode**: Built-in mock analysis engine that operates seamlessly without external API keys for demonstrations and local UI development.
- **Decoupled Architecture**: Independent UI (`app.py`) and AI service (`analysis_service.py`), allowing seamless plug-and-play AI model integration.

---

## ⚙️ Prerequisites & Environment

- **Python Version**: Python 3.10 or higher (compatible with Python 3.10, 3.11, 3.12).
- **Operating System**: macOS, Linux, or Windows.

---

## 🚀 Installation & Launch

### 1. Clone & Navigate to Project

```bash
cd "Module 1"
```

### 2. Set Up a Virtual Environment (Recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

Install the core dependencies:

```bash
pip install -r requirements.txt
```

### 4. Launch the Application

```bash
streamlit run app.py
```

The application will start immediately at **http://localhost:8501**.

---

## 🔑 Environment Configuration & API Keys

The application works out of the box in **Demo Mode** without any API keys.

To connect a live multimodal vision model, configure environment variables:

1. Copy the example configuration file:
   ```bash
   cp .env.example .env
   ```

2. Open `.env` and set ONE of the following API keys:

| Provider | Environment Variable | Recommended SDK Package |
|---|---|---|
| **Google Gemini 1.5 Flash** | `GEMINI_API_KEY` | `pip install google-generativeai` |
| **OpenAI GPT-4o** | `OPENAI_API_KEY` | `pip install openai` |
| **Anthropic Claude 3.5** | `ANTHROPIC_API_KEY` | `pip install anthropic` |

3. Export the environment variables before starting Streamlit:
   ```bash
   source .env
   streamlit run app.py
   ```

*Alternatively, you may place keys in `.streamlit/secrets.toml` (which is already excluded in `.gitignore`).*

---

## 🎭 Demo / Mock Mode Behavior

- When no API credentials are configured, VeriSight AI automatically activates **Demo Mode**.
- The UI status indicator in the sidebar turns amber: `Demo Mode`.
- Demo assessments are clearly flagged with a purple banner and `[DEMO]` tags, ensuring simulated data is never confused with genuine model outputs.
- To facilitate realistic workflow demonstrations, the demo engine returns context-sensitive mock findings (e.g., detecting keywords like "crack", "dent", or "scratch" will showcase the `SUPPORTED` flow; words like "water" or "stolen" showcase the `INSUFFICIENT_EVIDENCE` flow).

---

## 🤝 AI Integration Contract (For AI Teammate)

The UI interacts with the AI engine solely through `analysis_service.py`. The frontend remains completely independent of model vendors or SDK versions.

### Function Signature

```python
def analyze_claim(claim_data: dict, uploaded_images: list[dict]) -> dict:
    ...
```

#### Input Parameters

1. `claim_data` (dict):
   ```python
   {
       "claim_id": "CLM-10091325-A1B2",
       "object_type": "Car",           # "Car" | "Laptop" | "Package"
       "description": "Front bumper cracked after hitting parking barrier",
       "damage_type": "Cracked bumper", # Optional
       "incident_date": "2026-10-09",   # Optional
       "history": "No previous damage"  # Optional
   }
   ```

2. `uploaded_images` (list of dicts):
   ```python
   [
       {
           "image_id": "IMG_01",          # Stable unique image ID
           "filename": "bumper_crack.jpg",
           "pil_image": <PIL.Image.Image> # Opened PIL image object (RGB)
       },
       ...
   ]
   ```

### Expected Output Schema

The function must return a dictionary conforming to the following structure:

```json
{
  "claim_id": "CLM-10091325-A1B2",
  "decision": "SUPPORTED",
  "object_type": "Car",
  "damage_type": "Cracked front bumper",
  "object_part": "Front Bumper",
  "severity": "MEDIUM",
  "supporting_image_ids": ["IMG_01"],
  "evidence_findings": [
    {
      "image_id": "IMG_01",
      "finding": "Hairline impact fracture along lower bumper cover",
      "relevance": "Consistent with low-speed barrier collision"
    }
  ],
  "risk_flags": [],
  "image_quality": "GOOD",
  "confidence": 0.88,
  "justification": "Clear visual fractures match the reported impact location and damage type.",
  "missing_evidence": ["Photo showing full front view of vehicle"]
}
```

#### Allowed Enum Values
- **decision**: `"SUPPORTED"` | `"CONTRADICTED"` | `"INSUFFICIENT_EVIDENCE"`
- **severity**: `"LOW"` | `"MEDIUM"` | `"HIGH"` | `"UNKNOWN"`
- **confidence**: `float` between `0.0` and `1.0`

### Where to Insert Your Code

In [`analysis_service.py`](analysis_service.py), navigate to:
```python
# ── TEAMMATE INTEGRATION HOOK ──
def _call_teammate_ai(claim_data: dict, uploaded_images: list[dict]) -> Optional[dict]:
    ...
```
Drop your model pipeline, API call, or custom agent logic directly inside this hook. Any output returned here will be automatically validated and normalized by `_normalize_response()`.

---

## 🗂️ Project Structure

```
Module 1/
├── app.py                  # Streamlit user interface & dashboard views
├── analysis_service.py     # AI integration contract, adapters & mock engine
├── requirements.txt        # Production dependencies (streamlit, Pillow)
├── .env.example            # Environment variable template
├── .gitignore              # Protects secrets, cache, and virtual environments
└── README.md               # Project documentation & collaboration guide
```

---

## 🔄 Safe GitHub Collaboration Workflow

Follow these steps to synchronize the repository with your team:

### 1. Initial Commit and Push (Main Repository Owner)

```bash
# Verify status (ensure no secrets or cache files are staged)
git status

# Stage project files
git add .

# Create initial integration commit
git commit -m "Prepare UI for AI integration"

# Link your GitHub remote repository (replace with your actual GitHub URL)
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git

# Push to the main branch
git push -u origin main
```

### 2. Teammate Workflow (Developing AI Module)

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd "Module 1"

# Create a dedicated feature branch
git checkout -b ai-integration

# Install dependencies and test UI
pip install -r requirements.txt
streamlit run app.py
```

### 3. Submitting the AI Implementation

Once the teammate has implemented `_call_teammate_ai` or their provider in `analysis_service.py`:

```bash
# Stage the modified analysis files
git add .

# Commit with a descriptive message
git commit -m "Add AI evidence analysis"

# Push the feature branch to GitHub
git push -u origin ai-integration
```

### 4. Integration & Merge

1. Open a **Pull Request** on GitHub from `ai-integration` into `main`.
2. Review the diff to confirm that UI code in `app.py` was preserved.
3. Merge the Pull Request.
4. Pull the updated code on the primary machine:
   ```bash
   git checkout main
   git pull origin main
   streamlit run app.py
   ```

---

> ⚖️ **Compliance Notice**: VeriSight AI provides evidence review and decision support for claims professionals. It does not automatically approve or deny insurance claims. Final determinations remain subject to human adjuster verification.
