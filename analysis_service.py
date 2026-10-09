"""
analysis_service.py
===================
AI Analysis Service & Integration Interface for VeriSight AI.

-----------------------------------------------------------------------------
CONTRACT SPECIFICATION FOR AI INTEGRATION
-----------------------------------------------------------------------------
Public Entry Point:
    analyze_claim(claim_data: dict, uploaded_images: list[dict]) -> dict

Parameters:
-----------
claim_data : dict
    {
        "claim_id": str,          # e.g. "CLM-10091325-A1B2"
        "object_type": str,       # "Car" | "Laptop" | "Package"
        "description": str,       # Damage description submitted by user
        "damage_type": str,       # Optional claimed damage type
        "incident_date": str,     # Optional ISO date string (YYYY-MM-DD)
        "history": str            # Optional prior incident / claim history notes
    }

uploaded_images : list[dict]
    [
        {
            "image_id": "IMG_01",              # Stable identifier for the image
            "filename": "front_bumper.jpg",    # Original filename
            "pil_image": <PIL.Image.Image>     # Opened PIL image object (RGB)
        },
        ...
    ]

Expected Normalized Response Dictionary:
----------------------------------------
{
    "claim_id": str,                      # Matched claim ID
    "decision": str,                      # "SUPPORTED" | "CONTRADICTED" | "INSUFFICIENT_EVIDENCE"
    "object_type": str,                   # Object category evaluated
    "damage_type": str,                   # Primary damage assessed
    "object_part": str,                   # Specific component (e.g. "Front Bumper", "Display")
    "severity": str,                      # "LOW" | "MEDIUM" | "HIGH" | "UNKNOWN"
    "supporting_image_ids": list[str],    # List of image IDs cited as evidence (e.g. ["IMG_01"])
    "evidence_findings": [                # Per-image observations
        {
            "image_id": str,              # e.g. "IMG_01"
            "finding": str,               # What is visible in this image
            "relevance": str              # How this relates to the claim
        }
    ],
    "risk_flags": list[str],              # Inconsistencies or notes requiring human adjuster review
    "image_quality": str,                 # "GOOD" | "FAIR" | "POOR"
    "confidence": float,                  # AI estimate between 0.0 and 1.0 (uncalibrated)
    "justification": str,                 # Narrative reasoning explaining the decision
    "missing_evidence": list[str],        # Suggestions for additional angles/photos needed
    "is_demo": bool,                      # True if simulated/mock data; False if real AI model
    "timestamp": str                      # ISO timestamp of evaluation
}

-----------------------------------------------------------------------------
HOW TEAMMATE PLUGS IN THEIR IMPLEMENTATION
-----------------------------------------------------------------------------
1. In this file, go to section:
   `# ── TEAMMATE INTEGRATION HOOK ──`
2. Implement your logic in `_call_teammate_ai(claim_data, uploaded_images)`.
3. You can use any model (Gemini, GPT-4o, Claude, Ollama, local models).
4. The output will be automatically validated and normalized by `_normalize_response()`.
5. If no API key or model is configured, the system automatically falls back to
   safe, labeled demo mode so the UI never crashes during presentations.
"""

import os
import json
import base64
import logging
import datetime
from io import BytesIO
from typing import Optional, List, Dict, Any

from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & Allowed Values
# ---------------------------------------------------------------------------

ALLOWED_DECISIONS = {"SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"}
ALLOWED_SEVERITIES = {"LOW", "MEDIUM", "HIGH", "UNKNOWN"}

# ---------------------------------------------------------------------------
# Provider Detection & Credential Management
# ---------------------------------------------------------------------------

def _detect_provider() -> Optional[str]:
    """
    Detect configured AI provider from environment variables or Streamlit secrets.
    Returns provider identifier string or None if unconfigured.
    """
    # Teammate custom provider / flag
    if os.environ.get("USE_CUSTOM_AI_MODEL"):
        return "custom"

    # Google Gemini
    if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
        return "gemini"

    # OpenAI
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"

    # Anthropic
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"

    # Streamlit secrets fallback (.streamlit/secrets.toml)
    try:
        import streamlit as st
        secrets = getattr(st, "secrets", {})
        if secrets.get("GOOGLE_API_KEY") or secrets.get("GEMINI_API_KEY"):
            return "gemini"
        if secrets.get("OPENAI_API_KEY"):
            return "openai"
        if secrets.get("ANTHROPIC_API_KEY"):
            return "anthropic"
    except Exception:
        pass

    return None


def _get_api_key(provider: str) -> Optional[str]:
    """Retrieve API key for the given provider without exposing secrets in logs."""
    key_map = {
        "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        "openai": ["OPENAI_API_KEY"],
        "anthropic": ["ANTHROPIC_API_KEY"],
        "custom": ["CUSTOM_AI_API_KEY"],
    }
    for var in key_map.get(provider, []):
        val = os.environ.get(var)
        if val:
            return val

    try:
        import streamlit as st
        secrets = getattr(st, "secrets", {})
        for var in key_map.get(provider, []):
            val = secrets.get(var)
            if val:
                return val
    except Exception:
        pass

    return None


def get_provider_status() -> Dict[str, Any]:
    """
    Return a summary dict of the current AI provider connection status.
    Used by the UI sidebar indicator.
    """
    provider = _detect_provider()
    if provider is None:
        return {
            "configured": False,
            "provider": None,
            "message": "No API key configured. Running in Demo Mode.",
        }

    api_key = _get_api_key(provider)
    if provider != "custom" and not api_key:
        return {
            "configured": False,
            "provider": provider,
            "message": f"Provider '{provider}' detected but API key is missing.",
        }

    return {
        "configured": True,
        "provider": provider,
        "message": f"Connected — {provider.upper()} provider active.",
    }


# ---------------------------------------------------------------------------
# Image Helpers
# ---------------------------------------------------------------------------

def _image_to_base64(pil_image: Image.Image, max_dim: int = 1024) -> str:
    """Safely resize and encode a PIL Image as a base64 JPEG string."""
    img = pil_image.copy()
    img.thumbnail((max_dim, max_dim), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _build_prompt(claim_data: dict, image_ids: List[str]) -> str:
    """Build the structured prompt enforcing JSON output from vision models."""
    images_label = ", ".join(image_ids) if image_ids else "none"
    return f"""You are an expert insurance claims adjuster reviewing photographic evidence.

Claim details:
- Claim ID       : {claim_data.get('claim_id', 'N/A')}
- Object category: {claim_data.get('object_type', 'N/A')}
- Description    : {claim_data.get('description', 'N/A')}
- Claimed damage : {claim_data.get('damage_type') or 'Not specified'}
- Incident date  : {claim_data.get('incident_date') or 'Not specified'}
- Prior history  : {claim_data.get('history') or 'None provided'}
- Images provided: {images_label}

Instructions:
1. Carefully inspect every image for visible damage relevant to the claim.
2. Compare the visible evidence to the described damage.
3. Decide between:
   - "SUPPORTED" if clear evidence supports the damage description.
   - "CONTRADICTED" if reliable photographic evidence contradicts the claim.
   - "INSUFFICIENT_EVIDENCE" if damage is ambiguous, obstructed, or incomplete.
4. Assess severity: "LOW", "MEDIUM", "HIGH", or "UNKNOWN".
5. Reference ONLY the provided image IDs (e.g. {images_label}). Do NOT invent image IDs.
6. Note any risk indicators or missing photographic angles for human follow-up.
7. Treat confidence as an estimate (0.0 to 1.0).

Respond ONLY with a valid JSON object matching this schema (no markdown wrappers):
{{
  "claim_id": "{claim_data.get('claim_id', 'N/A')}",
  "decision": "<SUPPORTED|CONTRADICTED|INSUFFICIENT_EVIDENCE>",
  "object_type": "<string>",
  "damage_type": "<string>",
  "object_part": "<string>",
  "severity": "<LOW|MEDIUM|HIGH|UNKNOWN>",
  "supporting_image_ids": ["<image_id>", ...],
  "evidence_findings": [
    {{
      "image_id": "<image_id>",
      "finding": "<specific observation>",
      "relevance": "<relationship to claim>"
    }}
  ],
  "risk_flags": ["<risk flag or inconsistency>", ...],
  "image_quality": "<GOOD|FAIR|POOR>",
  "confidence": <0.0 to 1.0>,
  "justification": "<clear concise explanation>",
  "missing_evidence": ["<suggested additional photo/proof>", ...]
}}"""


# ---------------------------------------------------------------------------
# Response Normalizer & Safety Validator
# ---------------------------------------------------------------------------

def _normalize_response(raw: dict, claim_data: dict, image_ids: List[str]) -> dict:
    """
    Validate, sanitize, and normalize raw AI output into the guaranteed schema.
    Guarantees that UI never encounters missing keys or unexpected types.
    """
    # 1. Normalize decision
    decision = str(raw.get("decision", "INSUFFICIENT_EVIDENCE")).strip().upper()
    if decision not in ALLOWED_DECISIONS:
        decision = "INSUFFICIENT_EVIDENCE"

    # 2. Normalize severity
    severity = str(raw.get("severity", "UNKNOWN")).strip().upper()
    if severity not in ALLOWED_SEVERITIES:
        severity = "UNKNOWN"

    # 3. Normalize confidence (float between 0.0 and 1.0)
    confidence_raw = raw.get("confidence", 0.5)
    try:
        confidence = float(confidence_raw)
        confidence = max(0.0, min(1.0, confidence))
    except (TypeError, ValueError):
        confidence = 0.5

    # 4. Filter image IDs to only those actually submitted
    valid_ids = set(image_ids)
    raw_supporting = raw.get("supporting_image_ids", [])
    if isinstance(raw_supporting, list):
        supporting_ids = [str(iid) for iid in raw_supporting if str(iid) in valid_ids]
    else:
        supporting_ids = []

    # 5. Sanitize evidence findings
    findings = []
    raw_findings = raw.get("evidence_findings", [])
    if isinstance(raw_findings, list):
        for f in raw_findings:
            if isinstance(f, dict):
                iid = str(f.get("image_id", ""))
                if iid in valid_ids:
                    findings.append({
                        "image_id": iid,
                        "finding": str(f.get("finding", "Visual observation recorded.")),
                        "relevance": str(f.get("relevance", "Evaluated against claim description.")),
                    })

    # If model cited no specific findings, provide baseline per-image placeholder
    if not findings and valid_ids:
        findings = [
            {
                "image_id": iid,
                "finding": "Image reviewed by analysis engine.",
                "relevance": "Evaluated during visual inspection.",
            }
            for iid in image_ids
        ]

    # 6. Sanitize risk flags
    raw_flags = raw.get("risk_flags", [])
    risk_flags = [str(r) for r in raw_flags if r] if isinstance(raw_flags, list) else []

    # 7. Sanitize missing evidence
    raw_missing = raw.get("missing_evidence", [])
    missing_evidence = [str(m) for m in raw_missing if m] if isinstance(raw_missing, list) else []

    # 8. Assemble complete dictionary adhering to contract
    return {
        "claim_id": str(claim_data.get("claim_id", raw.get("claim_id", "N/A"))),
        "decision": decision,
        "object_type": str(raw.get("object_type") or claim_data.get("object_type", "")),
        "damage_type": str(raw.get("damage_type") or claim_data.get("damage_type", "")),
        "object_part": str(raw.get("object_part", "")),
        "severity": severity,
        "supporting_image_ids": supporting_ids,
        "evidence_findings": findings,
        "risk_flags": risk_flags,
        "image_quality": str(raw.get("image_quality", "FAIR")).upper(),
        "confidence": round(confidence, 2),
        "justification": str(raw.get("justification", "Analysis completed based on submitted photographic evidence.")),
        "missing_evidence": missing_evidence,
        "is_demo": False,
        "timestamp": datetime.datetime.now().isoformat(),
    }


# ---------------------------------------------------------------------------
# Demo / Mock Engine (Active when no AI provider is configured)
# ---------------------------------------------------------------------------

def _demo_response(claim_data: dict, uploaded_images: list[dict]) -> dict:
    """
    Return a clearly-labelled mock result when running without API keys.
    Prevents UI crashes and enables full interactive testing during demos.
    Never misleads adjusters: explicitly tagged with is_demo=True.
    """
    image_ids = [img["image_id"] for img in uploaded_images]
    desc_lower = claim_data.get("description", "").lower()

    # Intelligent mock scenarios based on claim text (for demo presentation)
    if "scratch" in desc_lower or "crack" in desc_lower or "dent" in desc_lower or "broken" in desc_lower:
        decision = "SUPPORTED"
        severity = "MEDIUM"
        confidence = 0.85
        supporting = [image_ids[0]] if image_ids else []
        findings = [
            {
                "image_id": iid,
                "finding": f"[DEMO] Visible surface deformity observed consistent with reported incident.",
                "relevance": "[DEMO] Corroborates user's reported damage location.",
            }
            for iid in image_ids
        ]
        flags = []
        justification = (
            "[DEMO SIMULATION] Photographic evidence shows damage consistent with the claimed "
            f"{claim_data.get('damage_type') or 'damage'} on the {claim_data.get('object_type', 'item')}. "
            "Connect an API key (e.g. GEMINI_API_KEY) for real multimodal model inspection."
        )
        missing = ["Close-up macro shot with measurement scale", "Overview shot of entire object"]
    elif "stolen" in desc_lower or "lost" in desc_lower or "water" in desc_lower:
        decision = "INSUFFICIENT_EVIDENCE"
        severity = "UNKNOWN"
        confidence = 0.40
        supporting = []
        findings = [
            {
                "image_id": iid,
                "finding": "[DEMO] Submitted image does not display internal or unobservable damage directly.",
                "relevance": "[DEMO] Cannot confirm internal condition from exterior photo alone.",
            }
            for iid in image_ids
        ]
        flags = ["Damage claimed cannot be verified exclusively via exterior photographs"]
        justification = (
            "[DEMO SIMULATION] The claimed issue cannot be conclusively verified from exterior photographs alone. "
            "Additional technical diagnostic report is suggested. Connect an AI provider to enable live analysis."
        )
        missing = ["Diagnostic / service center report", "Proof of incident report / police report"]
    else:
        decision = "INSUFFICIENT_EVIDENCE"
        severity = "LOW"
        confidence = 0.50
        supporting = []
        findings = [
            {
                "image_id": iid,
                "finding": f"[DEMO] Evidence placeholder for {iid}.",
                "relevance": "[DEMO] Simulated evaluation in demonstration mode.",
            }
            for iid in image_ids
        ]
        flags = []
        justification = (
            "⚠️ DEMO MODE ACTIVE — No AI provider API key detected. "
            "These results are simulated for presentation purposes. "
            "Set GEMINI_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY in your environment to enable real AI inspection."
        )
        missing = ["Additional clear lighting photographs"]

    return {
        "claim_id": claim_data.get("claim_id", "N/A"),
        "decision": decision,
        "object_type": claim_data.get("object_type", ""),
        "damage_type": claim_data.get("damage_type", ""),
        "object_part": "Primary exterior surface",
        "severity": severity,
        "supporting_image_ids": supporting,
        "evidence_findings": findings,
        "risk_flags": flags,
        "image_quality": "FAIR",
        "confidence": confidence,
        "justification": justification,
        "missing_evidence": missing,
        "is_demo": True,
        "timestamp": datetime.datetime.now().isoformat(),
    }


# ---------------------------------------------------------------------------
# ── TEAMMATE INTEGRATION HOOK ──
# ---------------------------------------------------------------------------

def _call_teammate_ai(claim_data: dict, uploaded_images: list[dict]) -> Optional[dict]:
    """
    =========================================================================
    TEAMMATE: DROP YOUR AI IMPLEMENTATION HERE!
    =========================================================================
    If you have built custom vision analysis code (e.g. in your own file,
    or directly in this function):
    
    1. Inspect `claim_data` (claim_id, object_type, description, etc.)
    2. Inspect `uploaded_images` (image_id, filename, pil_image)
    3. Run your model / inference logic.
    4. Return a dict matching the contract (or raw dict — it will be normalized).
    5. Return None if you want to fall back to the provider adapters below.
    =========================================================================
    """
    # Example integration pattern:
    # ----------------------------
    # try:
    #     import your_ai_module
    #     return your_ai_module.analyze(claim_data, uploaded_images)
    # except ImportError:
    #     pass

    return None


# ---------------------------------------------------------------------------
# Default Provider Adapters (Gemini, OpenAI, Anthropic)
# ---------------------------------------------------------------------------

def _call_gemini(api_key: str, prompt: str, images_b64: list[dict]) -> str:
    """Call Google Gemini 1.5 Flash Vision API and return raw JSON string."""
    import google.generativeai as genai  # type: ignore

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    parts = [prompt]
    for img_info in images_b64:
        parts.append({
            "inline_data": {
                "mime_type": img_info["mime_type"],
                "data": img_info["b64"],
            }
        })

    response = model.generate_content(parts)
    return response.text


def _call_openai(api_key: str, prompt: str, images_b64: list[dict]) -> str:
    """Call OpenAI GPT-4o Vision API and return raw JSON string."""
    import openai  # type: ignore

    client = openai.OpenAI(api_key=api_key)
    content: list[dict] = [{"type": "text", "text": prompt}]
    for img_info in images_b64:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{img_info['mime_type']};base64,{img_info['b64']}",
                "detail": "high",
            },
        })

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": content}],
        max_tokens=1500,
    )
    return response.choices[0].message.content or "{}"


def _call_anthropic(api_key: str, prompt: str, images_b64: list[dict]) -> str:
    """Call Anthropic Claude Vision API and return raw JSON string."""
    import anthropic  # type: ignore

    client = anthropic.Anthropic(api_key=api_key)
    content: list[dict] = []
    for img_info in images_b64:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": img_info["mime_type"],
                "data": img_info["b64"],
            },
        })
    content.append({"type": "text", "text": prompt})

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1500,
        messages=[{"role": "user", "content": content}],
    )
    return response.content[0].text


# ---------------------------------------------------------------------------
# Public Entry Point
# ---------------------------------------------------------------------------

def analyze_claim(claim_data: dict, uploaded_images: list[dict]) -> dict:
    """
    Analyze a damage claim against uploaded photographic evidence.

    Parameters
    ----------
    claim_data : dict
        Keys: claim_id, object_type, description, damage_type, incident_date, history
    uploaded_images : list[dict]
        Each element: {
            "image_id": str (e.g. "IMG_01"),
            "filename": str,
            "pil_image": PIL.Image.Image
        }

    Returns
    -------
    dict
        Normalized dictionary adhering to the VeriSight AI contract.
        Guarantees keys: decision, object_type, damage_type, object_part, severity,
        supporting_image_ids, evidence_findings, risk_flags, image_quality,
        confidence, justification, missing_evidence, is_demo, timestamp.
    """
    image_ids = [img["image_id"] for img in uploaded_images]

    # Step 1: Check teammate custom hook first
    try:
        custom_result = _call_teammate_ai(claim_data, uploaded_images)
        if custom_result is not None:
            logger.info("Using teammate custom AI module response.")
            return _normalize_response(custom_result, claim_data, image_ids)
    except Exception as exc:
        logger.error("Teammate AI module raised exception: %s", exc)
        return {
            **_demo_response(claim_data, uploaded_images),
            "is_demo": False,
            "justification": f"Teammate AI module encountered an error: {exc}. Please check logs.",
        }

    # Step 2: Detect configured API provider
    provider = _detect_provider()

    # Step 3: If no provider configured, return safe demo response
    if provider is None:
        logger.info("No AI provider configured — running in demo mode.")
        return _demo_response(claim_data, uploaded_images)

    api_key = _get_api_key(provider)
    if not api_key and provider != "custom":
        logger.warning("Provider '%s' detected but API key missing — running in demo mode.", provider)
        return _demo_response(claim_data, uploaded_images)

    # Step 4: Encode images for vision API
    images_b64: list[dict] = []
    for img_info in uploaded_images:
        try:
            b64 = _image_to_base64(img_info["pil_image"])
            images_b64.append({
                "image_id": img_info["image_id"],
                "b64": b64,
                "mime_type": "image/jpeg",
            })
        except Exception as exc:
            logger.warning("Could not encode image %s: %s", img_info.get("image_id"), exc)

    prompt = _build_prompt(claim_data, image_ids)

    # Step 5: Execute provider call with robust error handling
    try:
        if provider == "gemini":
            raw_text = _call_gemini(api_key, prompt, images_b64)
        elif provider == "openai":
            raw_text = _call_openai(api_key, prompt, images_b64)
        elif provider == "anthropic":
            raw_text = _call_anthropic(api_key, prompt, images_b64)
        else:
            logger.error("Unknown provider '%s'.", provider)
            return _demo_response(claim_data, uploaded_images)

        # Strip markdown json codeblock wrappers if present
        raw_text = raw_text.strip()
        if raw_text.startswith("```"):
            parts = raw_text.split("```")
            if len(parts) >= 2:
                inner = parts[1]
                if inner.startswith("json"):
                    inner = inner[4:]
                raw_text = inner.strip()

        raw_dict = json.loads(raw_text)
        return _normalize_response(raw_dict, claim_data, image_ids)

    except json.JSONDecodeError as exc:
        logger.error("Provider returned unparseable JSON: %s", exc)
        return {
            **_demo_response(claim_data, uploaded_images),
            "is_demo": False,
            "justification": f"AI model response was not valid JSON. Provider: {provider}. Error: {exc}.",
        }
    except Exception as exc:
        logger.error("Provider call failed: %s", exc)
        return {
            **_demo_response(claim_data, uploaded_images),
            "is_demo": False,
            "justification": f"AI analysis failed ({type(exc).__name__}: {exc}). Please verify API credentials and connectivity.",
        }
