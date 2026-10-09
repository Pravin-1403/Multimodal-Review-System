"""
analysis_service.py
===================
AI Analysis Service for VeriSight AI — Multimodal Evidence Review.

-----------------------------------------------------------------------------
PUBLIC CONTRACT (DO NOT CHANGE SIGNATURE — UI DEPENDS ON IT)
-----------------------------------------------------------------------------
    analyze_claim(claim_data: dict, uploaded_images: list[dict]) -> dict
    get_provider_status() -> dict

claim_data keys:
    claim_id        str
    object_type     str   "Car" | "Laptop" | "Package"
    description     str
    damage_type     str   (optional)
    incident_date   str   (optional ISO date)
    history         str   (optional)

uploaded_images items:
    image_id        str   "IMG_01", "IMG_02", …
    filename        str
    pil_image       PIL.Image.Image

Response schema (always all keys present):
    claim_id, decision, object_type, damage_type, object_part, severity,
    supporting_image_ids, evidence_findings, risk_flags, image_quality,
    confidence, justification, missing_evidence, is_demo, timestamp

-----------------------------------------------------------------------------
PROVIDER SELECTION (auto-detected, priority order):
    1. GEMINI_API_KEY or GOOGLE_API_KEY  → google-genai SDK v2+
    2. OPENAI_API_KEY                    → openai SDK v1+
    3. No key found                      → demo/mock mode (clearly labelled)

TEAMMATE HOOK: implement _call_teammate_ai() to add a custom model.
-----------------------------------------------------------------------------
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
# Constants
# ---------------------------------------------------------------------------

ALLOWED_DECISIONS  = {"SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"}
ALLOWED_SEVERITIES = {"LOW", "MEDIUM", "HIGH", "UNKNOWN"}

# Gemini model to use — gemini-2.5-flash is fast, cheap, and vision-capable in google-genai SDK
GEMINI_MODEL          = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_FALLBACK_MODELS = ["gemini-2.5-flash", "gemini-flash-latest", "gemini-2.5-flash-lite"]
# OpenAI model to use — gpt-4o supports vision
OPENAI_MODEL   = "gpt-4o"
# Max image dimension (pixels) before encoding — keeps payload small
MAX_IMAGE_DIM  = 1024


# ---------------------------------------------------------------------------
# Provider detection & credential management
# ---------------------------------------------------------------------------

def _detect_provider() -> Optional[str]:
    """
    Auto-detect which AI provider is configured.
    Checks environment variables first, then Streamlit secrets.
    Returns 'gemini', 'openai', 'custom', or None (demo mode).
    """
    # Custom teammate hook always wins if flagged
    if os.environ.get("USE_CUSTOM_AI_MODEL"):
        return "custom"

    # Env vars
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return "gemini"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"

    # Streamlit secrets fallback
    try:
        import streamlit as st
        secrets = getattr(st, "secrets", {})
        if secrets.get("GEMINI_API_KEY") or secrets.get("GOOGLE_API_KEY"):
            return "gemini"
        if secrets.get("OPENAI_API_KEY"):
            return "openai"
    except Exception:
        pass

    return None


def _get_api_key(provider: str) -> Optional[str]:
    """Retrieve the API key for the given provider — never log the value."""
    key_map = {
        "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        "openai": ["OPENAI_API_KEY"],
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
    """Return a status dict consumed by the sidebar status indicator."""
    provider = _detect_provider()
    if provider is None:
        return {
            "configured": False,
            "provider": None,
            "message": "No API key configured. Running in Demo Mode.",
        }
    if provider == "custom":
        return {
            "configured": True,
            "provider": "custom",
            "message": "Connected — Custom AI module active.",
        }
    api_key = _get_api_key(provider)
    if not api_key:
        return {
            "configured": False,
            "provider": provider,
            "message": f"Provider '{provider}' detected but key is missing.",
        }
    return {
        "configured": True,
        "provider": provider,
        "message": f"Connected — {provider.upper()} provider active.",
    }


# ---------------------------------------------------------------------------
# Image encoding helpers
# ---------------------------------------------------------------------------

def _pil_to_bytes(pil_image: Image.Image, max_dim: int = MAX_IMAGE_DIM) -> bytes:
    """Resize and encode a PIL Image to JPEG bytes."""
    img = pil_image.copy().convert("RGB")
    img.thumbnail((max_dim, max_dim), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _pil_to_b64(pil_image: Image.Image, max_dim: int = MAX_IMAGE_DIM) -> str:
    """Resize and encode a PIL Image as a base64 string."""
    return base64.b64encode(_pil_to_bytes(pil_image, max_dim)).decode("utf-8")


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(claim_data: dict, image_ids: List[str]) -> str:
    """
    Build the structured vision prompt that instructs the model to return JSON.
    The prompt enforces rigorous multimodal evidence assessment:
    1. Evaluates whether each image shows the specific claimed component.
    2. Prohibits concluding damage is absent when surface is hidden, distant, blurry, or wrong angle.
    3. Distinguishes between SUPPORTED, CONTRADICTED, and INSUFFICIENT_EVIDENCE.
    4. Explains visible parts vs unassessed parts.
    5. Prohibits misrepresentation risk flags solely because damage is not visible.
    6. Separates observed facts from interpretations.
    7. Cites only valid image IDs.
    """
    images_label = ", ".join(image_ids) if image_ids else "none"
    return f"""You are a senior forensic insurance claims assessor and multimodal evidence review expert.

== CLAIM DETAILS ==
Claim ID        : {claim_data.get('claim_id', 'N/A')}
Object category : {claim_data.get('object_type', 'N/A')}
Claimant's desc : {claim_data.get('description', 'N/A')}
Claimed damage  : {claim_data.get('damage_type') or 'Not specified'}
Incident date   : {claim_data.get('incident_date') or 'Not specified'}
Prior history   : {claim_data.get('history') or 'None provided'}
Images provided : {images_label}

== MULTIMODAL REVIEW PROTOCOL ==

STEP 1: IDENTIFY THE CLAIMED COMPONENT
Determine the specific physical part or surface claimed to be damaged from the claimant's description (e.g., "Rear Bumper", "Driver-Side Door", "Laptop Display Panel", "Package Exterior").

STEP 2: PER-IMAGE PERSPECTIVE & PART INSPECTION
For EVERY provided image ({images_label}):
a. Camera Perspective & Angle: What view does this image present? (e.g., Front direct view, Rear 3/4 angle, Overhead, Close-up).
b. Visible Parts: Explicitly identify which components/surfaces of the object are clearly visible.
c. Target Part Coverage:
   - Is the SPECIFIC claimed part/surface shown in this image?
   - If the photo shows a DIFFERENT side or angle of the object (e.g. front view when rear bumper damage is claimed), explicitly document that the claimed area is NOT SHOWN.
   - If the claimed surface is partially occluded, in shadow, too distant, blurry, or viewed from an acute angle, document that detail is INADEQUATE.
d. Factual Physical Observations: Record ONLY physical visual facts observed in this image. Keep findings strictly factual (what surfaces are seen and their physical condition). Do NOT accuse the claimant or assume fraud.

STEP 3: RIGOROUS DECISION RULES
Choose EXACTLY ONE decision:

- "SUPPORTED"
  * Criteria: Clear, sharp visual evidence directly captures physical damage on the claimed part that corroborates the claimant's description.
  * Supporting IDs: Include ONLY the image IDs ({images_label}) that clearly show this corroborating damage.

- "CONTRADICTED"
  * Criteria: The SPECIFIC claimed part/surface is CLEARLY, DIRECTLY, and COMPREHENSIVELY visible in sharp focus and good lighting, AND is conclusively proven to be pristine, intact, or in direct physical conflict with the claim.
  * ⛔ CRITICAL RULE: NEVER conclude "CONTRADICTED" if the claimed surface is NOT visible, partially hidden, too distant, blurry, or shown from an inadequate angle!
  * ⛔ Absence of visible damage in an image showing the wrong angle (e.g., front photo for a rear bumper claim) is NEVER contradiction. That is INSUFFICIENT_EVIDENCE.
  * Supporting IDs: Include ONLY image IDs ({images_label}) that directly and indisputably prove the claimed part is intact.

- "INSUFFICIENT_EVIDENCE"
  * Criteria: The available images cannot reliably establish either confirmation or contradiction.
  * Select this whenever:
    - The claimed part is not visible in the provided photos (e.g., rear damage claimed, but only front photo provided).
    - The photo angle is inadequate or does not cover the damaged area.
    - The photo is too distant, low-resolution, blurry, dark, or obstructed.
    - Any doubt exists whether the affected surface was captured.
  * Supporting IDs: [] (must be empty).

STEP 4: UNCALIBRATED CONFIDENCE ESTIMATE
- If decision is "INSUFFICIENT_EVIDENCE": Confidence MUST be low-to-moderate (0.20 to 0.45). Do NOT assign high confidence when the relevant evidence is incomplete.
- If decision is "SUPPORTED" or "CONTRADICTED": Assign 0.70 to 0.95 ONLY when the specific claimed surface is directly, clearly, and completely visible.

STEP 5: RISK INDICATORS & MISREPRESENTATION GUARDRAIL
- ⛔ DO NOT generate a risk flag for potential misrepresentation or fraud solely because damage is not visible in a photograph, especially when the relevant surface was not shown or had inadequate detail.
- Only flag genuine risk indicators with a defensible visual basis (e.g., mismatched VIN/make/model across images, evidence of image tampering/photoshop, or pre-existing aged rust on a claim of fresh damage). If none exist, return an empty array [].

STEP 6: EXPLANATION IN JUSTIFICATION
Your justification MUST explicitly state:
1. Which parts/surfaces of the object are visible in the provided evidence.
2. Which claimed parts/surfaces are NOT adequately assessed or missing from the photos.
3. The factual rationale linking the evidence to the final decision.

== OUTPUT FORMAT ==
Respond ONLY with the following JSON object — no markdown code fences, no surrounding text:
{{
  "claim_id": "{claim_data.get('claim_id', 'N/A')}",
  "decision": "<SUPPORTED|CONTRADICTED|INSUFFICIENT_EVIDENCE>",
  "object_type": "<identified object category>",
  "damage_type": "<primary damage observed or claimed>",
  "object_part": "<specific affected component>",
  "visible_parts": ["<components clearly visible in photographs>"],
  "unassessed_parts": ["<claimed parts or relevant areas not adequately shown>"],
  "severity": "<LOW|MEDIUM|HIGH|UNKNOWN>",
  "supporting_image_ids": ["<image_id from {images_label}>"],
  "evidence_findings": [
    {{
      "image_id": "<image_id from {images_label}>",
      "finding": "<strictly objective visual observation of visible parts and surface condition>",
      "relevance": "<factual relationship of this observation to the claim>"
    }}
  ],
  "risk_flags": ["<defensible risk indicator, if any — do NOT flag merely because damage is not visible>"],
  "image_quality": "<GOOD|FAIR|POOR>",
  "confidence": <float between 0.0 and 1.0; MUST be <= 0.45 if INSUFFICIENT_EVIDENCE>,
  "justification": "<detailed explanation specifying visible parts, unassessed parts, and decision rationale>",
  "missing_evidence": ["<specific photo views or documentation needed to complete assessment>"]
}}"""


# ---------------------------------------------------------------------------
# Provider adapters  (provider-specific SDK code ONLY inside these functions)
# ---------------------------------------------------------------------------

def _call_gemini(api_key: str, prompt: str, uploaded_images: List[dict]) -> str:
    """
    Call Google Gemini vision model using the NEW google-genai SDK (v2+).

    SDK: pip install google-genai>=2.0.0
    Model: gemini-2.5-flash  (vision-capable, cost-effective, high quality)
    """
    import google.genai as genai          # type: ignore
    from google.genai import types        # type: ignore

    client = genai.Client(api_key=api_key)

    # Build content parts: one image per uploaded file + the text prompt last
    parts: List[types.Part] = []
    for img_info in uploaded_images:
        try:
            img_bytes = _pil_to_bytes(img_info["pil_image"])
            parts.append(
                types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")
            )
        except Exception as exc:
            logger.warning("Could not encode image %s for Gemini: %s",
                           img_info.get("image_id"), exc)

    # Append the structured text prompt
    parts.append(types.Part(text=prompt))

    # Candidate models to try in order of preference
    candidate_models = [GEMINI_MODEL]
    for fallback in GEMINI_FALLBACK_MODELS:
        if fallback not in candidate_models:
            candidate_models.append(fallback)

    last_exc = None
    for model_name in candidate_models:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=parts,
                config=types.GenerateContentConfig(
                    temperature=0.1,            # low temperature for factual JSON output
                    max_output_tokens=2048,
                ),
            )
            return response.text
        except Exception as exc:
            err_str = str(exc)
            if "404" in err_str or "not found" in err_str.lower():
                logger.warning("Gemini model '%s' not found (%s), trying next fallback...", model_name, exc)
                last_exc = exc
                continue
            raise

    if last_exc:
        raise last_exc


def _call_openai(api_key: str, prompt: str, uploaded_images: List[dict]) -> str:
    """
    Call OpenAI GPT-4o vision model using the NEW openai SDK (v1+).

    SDK: pip install openai>=1.0.0
    Model: gpt-4o  (vision-capable)
    """
    from openai import OpenAI             # type: ignore

    client = OpenAI(api_key=api_key)

    # Build message content: images first, then text prompt
    content: List[dict] = []
    for img_info in uploaded_images:
        try:
            b64 = _pil_to_b64(img_info["pil_image"])
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                    "detail": "high",
                },
            })
        except Exception as exc:
            logger.warning("Could not encode image %s for OpenAI: %s",
                           img_info.get("image_id"), exc)

    content.append({"type": "text", "text": prompt})

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": content}],
        max_tokens=2048,
        temperature=0.1,
    )
    return response.choices[0].message.content or "{}"


# ---------------------------------------------------------------------------
# ── TEAMMATE INTEGRATION HOOK ──────────────────────────────────────────────
# ---------------------------------------------------------------------------

def _call_teammate_ai(
    claim_data: dict,
    uploaded_images: List[dict],
) -> Optional[dict]:
    """
    Drop your custom AI implementation here.

    Parameters
    ----------
    claim_data      : dict with keys claim_id, object_type, description,
                      damage_type, incident_date, history
    uploaded_images : list of dicts, each with image_id, filename, pil_image

    Return
    ------
    dict  — raw model output that will be auto-normalized (keys may be missing,
            they'll get safe defaults)
    None  — fall through to default provider adapters (Gemini / OpenAI)

    Example
    -------
    try:
        import my_vision_module
        raw = my_vision_module.run(claim_data, uploaded_images)
        return raw   # normalizer handles the rest
    except ImportError:
        return None  # fall back to default provider
    """
    return None   # <-- teammate: replace this with your implementation


# ---------------------------------------------------------------------------
# Response normalizer & safety validator
# ---------------------------------------------------------------------------

def _normalize_response(
    raw: dict,
    claim_data: dict,
    image_ids: List[str],
) -> dict:
    """
    Convert raw model output into the guaranteed VeriSight response schema.
    Handles missing fields, type coercions, and invalid enum values safely.
    Enforces forensic assessment rules:
    - Overrides CONTRADICTED to INSUFFICIENT_EVIDENCE if claimed part was unassessed or from inadequate angle.
    - Clamps confidence for INSUFFICIENT_EVIDENCE (<= 0.45) to avoid misleading high confidence.
    - Filters out spurious misrepresentation risk flags based merely on damage absence.
    - Ensures supporting_image_ids only contain valid image IDs and clears them for INSUFFICIENT_EVIDENCE.
    - Guarantees explanation of visible parts vs unassessed parts.
    """
    valid_ids = set(image_ids)

    # 1. Decision
    decision = str(raw.get("decision", "INSUFFICIENT_EVIDENCE")).strip().upper()
    if decision not in ALLOWED_DECISIONS:
        decision = "INSUFFICIENT_EVIDENCE"

    # 2. Severity
    severity = str(raw.get("severity", "UNKNOWN")).strip().upper()
    if severity not in ALLOWED_SEVERITIES:
        severity = "UNKNOWN"

    # 3. Image quality
    image_quality = str(raw.get("image_quality") or "FAIR").upper()
    if image_quality not in {"GOOD", "FAIR", "POOR"}:
        image_quality = "FAIR"

    # 4. Visible & Unassessed parts (Requirement 4)
    raw_visible = raw.get("visible_parts") or []
    visible_parts = [str(p).strip() for p in raw_visible if str(p).strip()] if isinstance(raw_visible, list) else []

    raw_unassessed = raw.get("unassessed_parts") or []
    unassessed_parts = [str(p).strip() for p in raw_unassessed if str(p).strip()] if isinstance(raw_unassessed, list) else []

    # 5. Supporting image IDs — only IDs from the actual upload are allowed (Requirement 8)
    raw_supporting = raw.get("supporting_image_ids") or []
    if isinstance(raw_supporting, list):
        supporting_ids = [str(iid) for iid in raw_supporting if str(iid) in valid_ids]
    else:
        supporting_ids = []

    # 6. Safety guardrail: Do not allow CONTRADICTED if claimed surface was not properly inspected (Requirement 1, 2, 3)
    justification = str(raw.get("justification") or "").strip()
    just_lower = justification.lower()
    claimed_text_lower = f"{claim_data.get('object_part', '')} {claim_data.get('damage_type', '')} {claim_data.get('description', '')}".lower()

    wrong_angle_signals = [
        "wrong angle", "inadequate angle", "opposite angle", "opposite side",
        "not visible in this image", "not visible from this angle", "not shown in the photo",
        "not shown in this image", "rear is not shown", "front is not shown",
        "not adequately assessed", "cannot be assessed from",
        "too distant to assess", "claimed part is not visible", "claimed area is not visible"
    ]

    claimed_part_is_unassessed = False
    if unassessed_parts:
        for u in unassessed_parts:
            u_low = u.lower()
            if any(term in u_low and term in claimed_text_lower for term in ["rear", "interior", "undercarriage", "passenger", "driver", "back"]):
                claimed_part_is_unassessed = True
                break

    indicates_inadequate_view = (
        any(sig in just_lower for sig in wrong_angle_signals)
        or claimed_part_is_unassessed
        or (image_quality == "POOR" and decision == "CONTRADICTED")
    )

    if decision == "CONTRADICTED" and indicates_inadequate_view:
        logger.warning("Overriding CONTRADICTED -> INSUFFICIENT_EVIDENCE: claimed area was not adequately captured.")
        decision = "INSUFFICIENT_EVIDENCE"

    # If INSUFFICIENT_EVIDENCE, no images should be cited as supporting a confirmed verdict
    if decision == "INSUFFICIENT_EVIDENCE":
        supporting_ids = []

    # 7. Confidence calibration (Requirement 7)
    try:
        confidence = float(raw.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
    except (TypeError, ValueError):
        confidence = 0.5

    if decision == "INSUFFICIENT_EVIDENCE":
        # Incomplete or inconclusive evidence must never display high confidence
        if confidence > 0.45:
            confidence = min(0.45, max(0.20, round(confidence * 0.45, 2)))
        elif confidence == 0.0:
            confidence = 0.30
    else:
        confidence = round(confidence, 2)

    # 8. Evidence findings — discard any finding referencing a non-existent image
    findings: List[dict] = []
    raw_findings = raw.get("evidence_findings") or []
    if isinstance(raw_findings, list):
        for f in raw_findings:
            if not isinstance(f, dict):
                continue
            iid = str(f.get("image_id", ""))
            if iid in valid_ids:
                findings.append({
                    "image_id": iid,
                    "finding": str(f.get("finding") or "Observation recorded."),
                    "relevance": str(f.get("relevance") or "Evaluated against claim."),
                })

    if not findings and image_ids:
        findings = [
            {
                "image_id": iid,
                "finding": "Image was reviewed by the analysis engine.",
                "relevance": "Evaluated in context of the submitted claim.",
            }
            for iid in image_ids
        ]

    # 9. Risk flags sanitization (Requirement 5 & 6)
    raw_flags = raw.get("risk_flags") or []
    risk_flags = []
    if isinstance(raw_flags, list):
        for r in raw_flags:
            flag_text = str(r).strip()
            if not flag_text:
                continue
            fl_lower = flag_text.lower()
            # Requirement 5: Do not generate risk flag for misrepresentation solely because damage is not visible
            is_spurious_fraud_flag = (
                any(term in fl_lower for term in ["misrepresentation", "fraud", "dishonest", "deceptive", "fabricated"])
                and any(w in fl_lower for w in ["not visible", "not observed", "not shown", "no damage", "missing from photo", "unseen"])
            )
            is_missing_evidence_as_risk = (
                decision == "INSUFFICIENT_EVIDENCE"
                and any(p in fl_lower for p in [
                    "damage not visible", "damage is not visible", "no visible damage",
                    "damage claimed is not visible", "damage not shown", "claimed damage not visible",
                    "damage not observed", "no damage observed"
                ])
            )
            if not is_spurious_fraud_flag and not is_missing_evidence_as_risk:
                risk_flags.append(flag_text)

    # 10. Missing evidence suggestions
    raw_missing = raw.get("missing_evidence") or []
    missing_evidence = [str(m).strip() for m in raw_missing if str(m).strip()] if isinstance(raw_missing, list) else []
    if decision == "INSUFFICIENT_EVIDENCE" and unassessed_parts and not missing_evidence:
        missing_evidence = [f"Clear, close-up photograph of the {part} from direct angle" for part in unassessed_parts]

    # 11. Ensure justification explains visible vs unassessed parts (Requirement 4)
    if not justification:
        justification = "Analysis completed."
    if visible_parts and not any(k in just_lower for k in ["visible part", "parts visible", "visible:"]):
        justification += f" Visible parts: {', '.join(visible_parts)}."
    if unassessed_parts and not any(k in just_lower for k in ["unassessed", "not assessed", "missing view", "not shown"]):
        justification += f" Unassessed parts: {', '.join(unassessed_parts)}."

    return {
        "claim_id":            str(claim_data.get("claim_id") or raw.get("claim_id") or "N/A"),
        "decision":            decision,
        "object_type":         str(raw.get("object_type") or claim_data.get("object_type") or ""),
        "damage_type":         str(raw.get("damage_type") or claim_data.get("damage_type") or ""),
        "object_part":         str(raw.get("object_part") or ""),
        "visible_parts":       visible_parts,
        "unassessed_parts":     unassessed_parts,
        "severity":            severity,
        "supporting_image_ids": supporting_ids,
        "evidence_findings":   findings,
        "risk_flags":          risk_flags,
        "image_quality":       image_quality,
        "confidence":          round(confidence, 2),
        "justification":       justification,
        "missing_evidence":    missing_evidence,
        "is_demo":             False,
        "timestamp":           datetime.datetime.now().isoformat(),
    }


# ---------------------------------------------------------------------------
# Demo / mock engine  (active when no API key is configured)
# ---------------------------------------------------------------------------

def _demo_response(claim_data: dict, uploaded_images: List[dict]) -> dict:
    """
    Return a clearly-labelled DEMO result when no provider is configured.
    Keyword-matching makes the demo slightly realistic for presentations.
    is_demo=True ensures the UI banner is always shown.
    """
    image_ids = [img["image_id"] for img in uploaded_images]
    desc_lower = (claim_data.get("description") or "").lower()
    dmg_lower  = (claim_data.get("damage_type") or "").lower()
    combined   = desc_lower + " " + dmg_lower

    # Pick a scenario based on keywords in the description
    if any(w in combined for w in ("crack", "scratch", "dent", "broken", "shatter", "chip", "fracture")):
        decision, severity, confidence = "SUPPORTED", "MEDIUM", 0.82
        supporting = [image_ids[0]] if image_ids else []
        visible_parts = [claim_data.get("object_part") or "Primary exterior surface"]
        unassessed_parts = ["Remaining perimeter surfaces"]
        findings = [
            {
                "image_id": iid,
                "finding": "[DEMO] Surface deformity consistent with reported impact observed.",
                "relevance": "[DEMO] Aligns with the claimant's described damage location.",
            }
            for iid in image_ids
        ]
        flags = []
        justification = (
            "[⚠️ DEMO SIMULATION — not real AI analysis] "
            f"Photographic evidence appears consistent with the claimed "
            f"{claim_data.get('damage_type') or 'damage'} to the "
            f"{claim_data.get('object_type', 'item')}. "
            f"Visible parts: {', '.join(visible_parts)}. Unassessed parts: {', '.join(unassessed_parts)}. "
            "Set GEMINI_API_KEY or OPENAI_API_KEY to enable live multimodal analysis."
        )
        missing = [
            "Close-up photograph with a measurement scale",
            "Full overview of the entire object",
        ]
    elif any(w in combined for w in ("stolen", "lost", "missing", "water", "flood", "fire")):
        decision, severity, confidence = "INSUFFICIENT_EVIDENCE", "UNKNOWN", 0.35
        supporting = []
        visible_parts = ["Exterior overview"]
        unassessed_parts = ["Internal mechanical / electronic components"]
        findings = [
            {
                "image_id": iid,
                "finding": "[DEMO] Image does not display direct physical evidence of the claimed incident type.",
                "relevance": "[DEMO] Internal or invisible damage cannot be confirmed from exterior photographs alone.",
            }
            for iid in image_ids
        ]
        flags = ["Claimed damage type may require non-visual documentation (e.g. police/fire report)"]
        justification = (
            "[⚠️ DEMO SIMULATION — not real AI analysis] "
            "The claimed incident type cannot be reliably verified from visual photographs alone. "
            f"Visible parts: {', '.join(visible_parts)}. Unassessed parts: {', '.join(unassessed_parts)}. "
            "A supporting document (incident report, diagnostic certificate) is recommended."
        )
        missing = [
            "Official incident report (police / fire department)",
            "Third-party diagnostic or repair estimate",
        ]
    else:
        decision, severity, confidence = "INSUFFICIENT_EVIDENCE", "LOW", 0.35
        supporting = []
        visible_parts = ["General exterior"]
        unassessed_parts = ["Specific claimed damage area"]
        findings = [
            {
                "image_id": iid,
                "finding": f"[DEMO] Evidence placeholder for {iid}.",
                "relevance": "[DEMO] Simulated in demo mode — connect an API key for real analysis.",
            }
            for iid in image_ids
        ]
        flags = []
        justification = (
            "⚠️ DEMO MODE ACTIVE — No AI provider API key is configured. "
            "These results are entirely simulated. "
            f"Visible parts: {', '.join(visible_parts)}. Unassessed parts: {', '.join(unassessed_parts)}. "
            "Set GEMINI_API_KEY (recommended) or OPENAI_API_KEY to enable real analysis."
        )
        missing = ["Additional clear photographs from multiple angles"]

    return {
        "claim_id":             claim_data.get("claim_id", "N/A"),
        "decision":             decision,
        "object_type":          claim_data.get("object_type", ""),
        "damage_type":          claim_data.get("damage_type", ""),
        "object_part":          claim_data.get("object_part") or "Primary exterior surface",
        "visible_parts":        visible_parts,
        "unassessed_parts":      unassessed_parts,
        "severity":             severity,
        "supporting_image_ids": supporting,
        "evidence_findings":    findings,
        "risk_flags":           flags,
        "image_quality":        "FAIR",
        "confidence":           confidence,
        "justification":        justification,
        "missing_evidence":     missing,
        "is_demo":              True,
        "timestamp":            datetime.datetime.now().isoformat(),
    }


# ---------------------------------------------------------------------------
# JSON extraction helper (strips markdown fences if model adds them)
# ---------------------------------------------------------------------------

def _extract_json(raw_text: str) -> dict:
    """
    Parse JSON from model output.
    Handles: plain JSON, ```json ... ```, ``` ... ```, and leading/trailing text.
    Raises json.JSONDecodeError if no valid JSON object is found.
    """
    text = raw_text.strip()

    # Find the outermost JSON object
    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        json_str = text[start : end + 1]
        return json.loads(json_str)

    raise json.JSONDecodeError("No JSON object found", text, 0)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def analyze_claim(claim_data: dict, uploaded_images: List[dict]) -> dict:
    """
    Analyze a damage claim against photographic evidence.

    Parameters
    ----------
    claim_data : dict
        claim_id, object_type, description, damage_type, incident_date, history
    uploaded_images : list[dict]
        Each: {image_id: str, filename: str, pil_image: PIL.Image.Image}

    Returns
    -------
    dict
        Normalized result with all required keys guaranteed.
        is_demo=True  → mock data, provider not used
        is_demo=False → real model call was made (or error path)
    """
    image_ids = [img["image_id"] for img in uploaded_images]

    # ── Step 1: Teammate custom hook (always checked first) ─────────────────
    try:
        custom_result = _call_teammate_ai(claim_data, uploaded_images)
        if custom_result is not None:
            logger.info("Using teammate custom AI module response.")
            return _normalize_response(custom_result, claim_data, image_ids)
    except Exception as exc:
        logger.error("Teammate AI module error: %s", exc)
        return {
            "claim_id":            str(claim_data.get("claim_id") or "N/A"),
            "decision":            "INSUFFICIENT_EVIDENCE",
            "object_type":         str(claim_data.get("object_type") or ""),
            "damage_type":         str(claim_data.get("damage_type") or ""),
            "object_part":         "N/A",
            "visible_parts":       [],
            "unassessed_parts":     [],
            "severity":            "UNKNOWN",
            "supporting_image_ids": [],
            "evidence_findings":   [],
            "risk_flags":          [f"Teammate AI module error: {type(exc).__name__}: {exc}"],
            "image_quality":       "UNKNOWN",
            "confidence":          0.0,
            "justification":       f"Teammate AI module raised an error: {type(exc).__name__}: {exc}.",
            "missing_evidence":    [],
            "is_demo":             False,
            "timestamp":           datetime.datetime.now().isoformat(),
        }

    # ── Step 2: Detect configured provider ──────────────────────────────────
    provider = _detect_provider()

    if provider is None:
        logger.info("No provider configured — returning demo response.")
        return _demo_response(claim_data, uploaded_images)

    api_key = _get_api_key(provider)
    if not api_key and provider != "custom":
        logger.warning("Provider '%s' key missing — returning demo response.", provider)
        return _demo_response(claim_data, uploaded_images)

    # ── Step 3: Build prompt ─────────────────────────────────────────────────
    prompt = _build_prompt(claim_data, image_ids)
    logger.info("Calling provider '%s' for claim %s with %d image(s).",
                provider, claim_data.get("claim_id"), len(uploaded_images))

    # ── Step 4: Call provider and parse response ─────────────────────────────
    try:
        if provider == "gemini":
            raw_text = _call_gemini(api_key, prompt, uploaded_images)
        elif provider == "openai":
            raw_text = _call_openai(api_key, prompt, uploaded_images)
        else:
            logger.error("Unknown provider '%s'.", provider)
            return _demo_response(claim_data, uploaded_images)

        logger.debug("Raw model output (first 300 chars): %s", raw_text[:300])

        raw_dict = _extract_json(raw_text)
        result = _normalize_response(raw_dict, claim_data, image_ids)
        logger.info("Analysis complete: decision=%s confidence=%.2f",
                    result["decision"], result["confidence"])
        return result

    except json.JSONDecodeError as exc:
        logger.error("Model returned unparseable JSON for claim %s: %s",
                     claim_data.get("claim_id"), exc)
        return {
            "claim_id":            str(claim_data.get("claim_id") or "N/A"),
            "decision":            "INSUFFICIENT_EVIDENCE",
            "object_type":         str(claim_data.get("object_type") or ""),
            "damage_type":         str(claim_data.get("damage_type") or ""),
            "object_part":         "N/A",
            "visible_parts":       [],
            "unassessed_parts":     [],
            "severity":            "UNKNOWN",
            "supporting_image_ids": [],
            "evidence_findings":   [
                {
                    "image_id": iid,
                    "finding": "The AI model response could not be parsed as structured JSON.",
                    "relevance": "Unable to extract visual findings.",
                }
                for iid in image_ids
            ],
            "risk_flags":          ["Model returned unparseable response"],
            "image_quality":       "UNKNOWN",
            "confidence":          0.0,
            "justification":       (
                f"The AI model's response could not be parsed as JSON. "
                f"Provider: {provider}. "
                "This may be a temporary model issue — please try again."
            ),
            "missing_evidence":    ["Please resubmit the claim for re-analysis."],
            "is_demo":             False,
            "timestamp":           datetime.datetime.now().isoformat(),
        }

    except Exception as exc:
        # Covers: network errors, rate limits, auth failures, timeouts
        err_type = type(exc).__name__
        err_msg  = str(exc)
        logger.error("Provider call failed [%s]: %s", err_type, err_msg)

        lowered = err_msg.lower()
        if "api_key" in lowered or "authentication" in lowered or "401" in err_msg or "permission_denied" in lowered:
            user_msg = (
                f"API key authentication failed for provider '{provider}'. "
                "Please verify your key is correct and has the required permissions."
            )
        elif "quota" in lowered or "rate limit" in lowered or "ratelimit" in lowered or "429" in err_msg or "resource_exhausted" in lowered:
            user_msg = (
                f"Rate limit or quota exceeded for provider '{provider}'. "
                "Please wait a moment and try again."
            )
        elif "timeout" in lowered or "timed out" in lowered:
            user_msg = (
                "The analysis request timed out. "
                "This may be due to large images or network issues. "
                "Please try again with fewer or smaller images."
            )
        elif "404" in err_msg or "not found" in lowered:
            user_msg = (
                f"Configured model for provider '{provider}' was not found. "
                "Please check the model name or API version."
            )
        else:
            user_msg = (
                f"Analysis failed ({err_type}): {err_msg[:120]}. "
                "Please check your API configuration and network connectivity."
            )

        return {
            "claim_id":            str(claim_data.get("claim_id") or "N/A"),
            "decision":            "INSUFFICIENT_EVIDENCE",
            "object_type":         str(claim_data.get("object_type") or ""),
            "damage_type":         str(claim_data.get("damage_type") or ""),
            "object_part":         "N/A",
            "visible_parts":       [],
            "unassessed_parts":     [],
            "severity":            "UNKNOWN",
            "supporting_image_ids": [],
            "evidence_findings":   [
                {
                    "image_id": iid,
                    "finding": f"Automated analysis failed: {user_msg}",
                    "relevance": "Unable to verify physical evidence against claim.",
                }
                for iid in image_ids
            ],
            "risk_flags":          [f"Provider call failed: {user_msg}"],
            "image_quality":       "UNKNOWN",
            "confidence":          0.0,
            "justification":       f"Analysis could not be completed: {user_msg}",
            "missing_evidence":    ["Please re-try the claim submission once provider service is available."],
            "is_demo":             False,
            "timestamp":           datetime.datetime.now().isoformat(),
        }
