"""
app.py — VeriSight AI  |  Multimodal Evidence Review & Claim Intelligence
==========================================================================
Run with:
    streamlit run app.py
"""

import json
import uuid
import logging
import datetime
import io

import streamlit as st
from PIL import Image

from analysis_service import analyze_claim, get_provider_status

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config  (MUST be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="VeriSight AI — Evidence Review",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS  —  Premium insurance-dashboard aesthetic
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global reset ───────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

.stApp {
    background: #F0F4F9;
}

/* ── Sidebar ────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(175deg, #0D1B2A 0%, #1B2B45 60%, #1F3460 100%);
    border-right: 1px solid rgba(255,255,255,0.07);
}
section[data-testid="stSidebar"] * {
    color: #E8EEF8 !important;
}
section[data-testid="stSidebar"] .stButton button {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    color: #E8EEF8 !important;
    border-radius: 10px;
    width: 100%;
    text-align: left;
    padding: 10px 16px;
    transition: all 0.2s ease;
    font-weight: 500;
    font-size: 0.9rem;
}
section[data-testid="stSidebar"] .stButton button:hover {
    background: rgba(59,130,246,0.25);
    border-color: rgba(59,130,246,0.5);
}

/* ── Stat cards ─────────────────────────────────────────────── */
.stat-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 22px 24px;
    text-align: center;
    box-shadow: 0 2px 12px rgba(13,27,42,0.06);
    transition: box-shadow 0.2s;
}
.stat-card:hover { box-shadow: 0 6px 24px rgba(13,27,42,0.12); }
.stat-label { font-size: 0.78rem; font-weight: 600; color: #64748B;
              letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 6px; }
.stat-value { font-size: 2.2rem; font-weight: 800; color: #0D1B2A; line-height: 1; }
.stat-value.blue  { color: #2563EB; }
.stat-value.green { color: #059669; }
.stat-value.amber { color: #D97706; }

/* ── Section header ─────────────────────────────────────────── */
.section-header {
    font-size: 1.25rem;
    font-weight: 700;
    color: #0D1B2A;
    margin: 0 0 4px 0;
}
.section-sub {
    font-size: 0.85rem;
    color: #64748B;
    margin: 0 0 18px 0;
}

/* ── Claim card / form card ─────────────────────────────────── */
.claim-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 20px;
    padding: 32px;
    box-shadow: 0 2px 16px rgba(13,27,42,0.07);
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 18px !important;
    padding: 24px !important;
    box-shadow: 0 2px 14px rgba(13,27,42,0.06) !important;
    margin-bottom: 20px !important;
}

/* ── Decision badges ────────────────────────────────────────── */
.badge {
    display: inline-block;
    padding: 8px 20px;
    border-radius: 30px;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.badge-supported    { background: #D1FAE5; color: #065F46; border: 1.5px solid #6EE7B7; }
.badge-contradicted { background: #FEE2E2; color: #991B1B; border: 1.5px solid #FCA5A5; }
.badge-insufficient { background: #FEF3C7; color: #92400E; border: 1.5px solid #FCD34D; }
.badge-demo         { background: #EDE9FE; color: #5B21B6; border: 1.5px solid #C4B5FD; }

/* ── Severity pill ───────────────────────────────────────────── */
.sev-low     { color: #059669; background: #D1FAE5; border-radius: 6px; padding: 2px 10px; font-weight: 600; font-size: 0.8rem; }
.sev-medium  { color: #B45309; background: #FEF3C7; border-radius: 6px; padding: 2px 10px; font-weight: 600; font-size: 0.8rem; }
.sev-high    { color: #991B1B; background: #FEE2E2; border-radius: 6px; padding: 2px 10px; font-weight: 600; font-size: 0.8rem; }
.sev-unknown { color: #4B5563; background: #F3F4F6; border-radius: 6px; padding: 2px 10px; font-weight: 600; font-size: 0.8rem; }

/* ── Risk flag card ─────────────────────────────────────────── */
.risk-flag {
    background: #FFFBEB;
    border: 1px solid #FCD34D;
    border-left: 4px solid #F59E0B;
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 10px;
    font-size: 0.88rem;
    color: #78350F;
}
.no-risk {
    background: #F0FDF4;
    border: 1px solid #BBF7D0;
    border-radius: 10px;
    padding: 12px 16px;
    color: #166534;
    font-size: 0.88rem;
    font-weight: 500;
}

/* ── Evidence image card ─────────────────────────────────────── */
.evidence-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 14px;
    box-shadow: 0 1px 8px rgba(13,27,42,0.05);
}
.evidence-card.highlighted {
    border: 2px solid #2563EB;
    box-shadow: 0 0 0 4px rgba(37,99,235,0.08);
}
.img-id-tag {
    display: inline-block;
    background: #EFF6FF;
    color: #1D4ED8;
    border: 1px solid #BFDBFE;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 700;
    padding: 2px 8px;
    margin-bottom: 6px;
}

/* ── Confidence bar ─────────────────────────────────────────── */
.conf-bar-track {
    background: #E2E8F0;
    border-radius: 99px;
    height: 8px;
    width: 100%;
    overflow: hidden;
    margin-top: 4px;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #2563EB, #38BDF8);
    transition: width 0.5s ease;
}

/* ── Missing evidence ────────────────────────────────────────── */
.missing-item {
    background: #F8FAFC;
    border: 1px dashed #CBD5E1;
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 0.87rem;
    color: #475569;
}

/* ── Divider ────────────────────────────────────────────────── */
.vs-divider { border: none; border-top: 1.5px solid #E2E8F0; margin: 24px 0; }

/* ── Demo banner ─────────────────────────────────────────────── */
.demo-banner {
    background: linear-gradient(90deg, #EDE9FE, #F5F3FF);
    border: 1.5px solid #C4B5FD;
    border-radius: 12px;
    padding: 14px 18px;
    color: #5B21B6;
    font-weight: 600;
    font-size: 0.88rem;
    margin-bottom: 20px;
}

/* ── Thumbnail container ─────────────────────────────────────── */
.thumb-label {
    font-size: 0.75rem;
    color: #64748B;
    text-align: center;
    margin-top: 4px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* ── Main header ─────────────────────────────────────────────── */
.main-header {
    background: linear-gradient(135deg, #0D1B2A 0%, #1E3A5F 100%);
    border-radius: 20px;
    padding: 28px 36px;
    margin-bottom: 28px;
    color: white;
}
.main-header h1 {
    font-size: 1.9rem;
    font-weight: 800;
    color: white;
    margin: 0;
}
.main-header p {
    color: #94A3B8;
    font-size: 0.9rem;
    margin: 6px 0 0 0;
}

/* ── Status dot ──────────────────────────────────────────────── */
.status-dot-green { display:inline-block; width:8px; height:8px;
    background:#10B981; border-radius:50%; margin-right:6px; }
.status-dot-amber { display:inline-block; width:8px; height:8px;
    background:#F59E0B; border-radius:50%; margin-right:6px; }

/* ── General label override ──────────────────────────────────── */
label { font-weight: 500 !important; color: #1E293B !important; }
.stTextArea textarea, .stTextInput input, .stSelectbox select {
    border-radius: 10px !important;
    border: 1.5px solid #CBD5E1 !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1D4ED8, #2563EB) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 14px 32px !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 12px rgba(37,99,235,0.35) !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 20px rgba(37,99,235,0.45) !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------

def _init_session():
    defaults = {
        "page": "new_claim",
        "recent_reviews": [],          # list of result dicts
        "current_result": None,
        "form_reset_key": 0,           # increment to reset form widgets
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_session()

# ---------------------------------------------------------------------------
# Session stats helpers
# ---------------------------------------------------------------------------

def _stats() -> dict:
    reviews = st.session_state.recent_reviews
    total = len(reviews)
    supported = sum(1 for r in reviews if r.get("decision") == "SUPPORTED" and not r.get("is_demo"))
    needs_review = total - supported
    return {"total": total, "supported": supported, "needs_review": needs_review}


# ---------------------------------------------------------------------------
# Unique ID generator
# ---------------------------------------------------------------------------

def _new_claim_id() -> str:
    ts = datetime.datetime.now().strftime("%m%d%H%M")
    short = uuid.uuid4().hex[:4].upper()
    return f"CLM-{ts}-{short}"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="padding: 10px 0 24px 0;">
            <div style="font-size: 1.5rem; font-weight: 800; letter-spacing:-0.5px;">
                🔍 VeriSight AI
            </div>
            <div style="font-size: 0.72rem; color: #94A3B8; margin-top: 2px; letter-spacing:0.04em;">
                EVIDENCE REVIEW PLATFORM
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("<div style='font-size:0.72rem;font-weight:700;letter-spacing:0.08em;color:#64748B;margin-bottom:10px;'>NAVIGATION</div>", unsafe_allow_html=True)

        if st.button("＋  New Claim", key="nav_new"):
            st.session_state.page = "new_claim"
            st.rerun()

        if st.button("📊  Review Dashboard", key="nav_dash"):
            st.session_state.page = "dashboard"
            st.rerun()

        # Recent reviews
        if st.session_state.recent_reviews:
            st.markdown("<div style='margin-top:24px;font-size:0.72rem;font-weight:700;letter-spacing:0.08em;color:#64748B;margin-bottom:10px;'>RECENT REVIEWS</div>", unsafe_allow_html=True)
            for rev in reversed(st.session_state.recent_reviews[-5:]):
                decision = rev.get("decision", "?")
                icon = {"SUPPORTED": "✅", "CONTRADICTED": "❌", "INSUFFICIENT_EVIDENCE": "⚠️"}.get(decision, "📄")
                demo_tag = " [DEMO]" if rev.get("is_demo") else ""
                label = f"{icon} {rev.get('claim_id', '?')}{demo_tag}"
                if st.button(label, key=f"rev_{rev.get('claim_id')}"):
                    st.session_state.current_result = rev
                    st.session_state.page = "results"
                    st.rerun()

        # System status
        st.markdown("---")
        status = get_provider_status()
        if status["configured"]:
            st.markdown(f"""
            <div style="font-size:0.78rem;">
                <span class="status-dot-green"></span>
                <strong>AI Online</strong><br>
                <span style="color:#94A3B8;font-size:0.72rem;">{status['message']}</span>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="font-size:0.78rem;">
                <span class="status-dot-amber"></span>
                <strong>Demo Mode</strong><br>
                <span style="color:#94A3B8;font-size:0.72rem;">{status['message']}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.68rem;color:#475569;border-top:1px solid rgba(255,255,255,0.08);
                    padding-top:14px;line-height:1.6;">
            ⚠️ For decision support only.<br>
            Not a substitute for human review.
        </div>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Stat cards row
# ---------------------------------------------------------------------------

def render_stat_cards():
    s = _stats()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Claims Reviewed</div>
            <div class="stat-value blue">{s['total']}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Supported</div>
            <div class="stat-value green">{s['supported']}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Needs Review</div>
            <div class="stat-value amber">{s['needs_review']}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# New Claim page
# ---------------------------------------------------------------------------

MAX_IMAGES = 8
MAX_IMAGE_MB = 10

def render_new_claim():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>Evidence Review Workspace</h1>
        <p>Submit a damage claim with photographic evidence. Our AI will analyse
           visual evidence and cross-reference it with the claim description to
           produce a structured, auditable assessment.</p>
    </div>""", unsafe_allow_html=True)

    render_stat_cards()

    # ── Claim form ──────────────────────────────────────────────────────────
    rk = st.session_state.form_reset_key   # used in widget keys for reset

    with st.container(border=True):
        st.markdown('<p class="section-header">📋 Submit New Claim</p>', unsafe_allow_html=True)
        st.markdown('<p class="section-sub">All fields marked * are required for analysis.</p>', unsafe_allow_html=True)

        col_a, col_b = st.columns([1, 1])

        with col_a:
            claim_id_display = _new_claim_id()
            st.text_input(
                "Claim ID (auto-generated)",
                value=claim_id_display,
                disabled=True,
                key=f"claim_id_display_{rk}",
            )
            object_type = st.selectbox(
                "Object Category *",
                ["Car", "Laptop", "Package"],
                key=f"object_type_{rk}",
            )
            damage_type = st.text_input(
                "Claimed Damage Type (optional)",
                placeholder="e.g. Cracked screen, Dented door, Water damage",
                key=f"damage_type_{rk}",
            )

        with col_b:
            incident_date = st.date_input(
                "Incident Date (optional)",
                value=None,
                key=f"incident_date_{rk}",
            )
            history = st.text_area(
                "Prior Claim History / Notes (optional)",
                placeholder="Previous incidents, repair records, or additional context…",
                height=108,
                key=f"history_{rk}",
            )

        description = st.text_area(
            "Claim Description *",
            placeholder="Describe the damage, how it occurred, and any relevant circumstances…",
            height=120,
            key=f"description_{rk}",
        )

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown("**Evidence Images *** *(up to 8 images, 10 MB each)*")
        uploaded_files = st.file_uploader(
            "Upload images",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            accept_multiple_files=True,
            key=f"file_uploader_{rk}",
            label_visibility="collapsed",
        )

        # Image validation & thumbnails
        valid_images = []
        if uploaded_files:
            errors = []
            for uf in uploaded_files[:MAX_IMAGES]:
                size_mb = uf.size / (1024 * 1024)
                if size_mb > MAX_IMAGE_MB:
                    errors.append(f"⚠️ {uf.name} is {size_mb:.1f} MB — exceeds {MAX_IMAGE_MB} MB limit.")
                    continue
                try:
                    img = Image.open(uf).convert("RGB")
                    valid_images.append({"file": uf, "pil": img})
                except Exception:
                    errors.append(f"⚠️ {uf.name} could not be opened as an image.")

            if len(uploaded_files) > MAX_IMAGES:
                st.warning(f"Only the first {MAX_IMAGES} images will be used.")
            for e in errors:
                st.error(e)

            if valid_images:
                st.markdown("**Uploaded Evidence Previews:**")
                cols = st.columns(min(len(valid_images), 4))
                for idx, vi in enumerate(valid_images):
                    with cols[idx % 4]:
                        st.image(vi["pil"], use_container_width=True)
                        name = vi["file"].name
                        short = name[:18] + "…" if len(name) > 18 else name
                        st.markdown(f'<div class="thumb-label">IMG_{idx+1:02d} · {short}</div>', unsafe_allow_html=True)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        submit_btn = st.button("🔍  Submit for Analysis", type="primary", key=f"submit_{rk}")

        if submit_btn:
            # Validation
            errs = []
            if not description.strip():
                errs.append("Claim description is required.")
            if not valid_images:
                errs.append("At least one valid evidence image is required.")
            if errs:
                for e in errs:
                    st.error(f"❌ {e}")
            else:
                _run_analysis(
                    claim_id=claim_id_display,
                    object_type=object_type,
                    description=description,
                    damage_type=damage_type,
                    incident_date=str(incident_date) if incident_date else "",
                    history=history,
                    valid_images=valid_images,
                )


def _run_analysis(claim_id, object_type, description, damage_type,
                  incident_date, history, valid_images):
    """Prepare payloads, call the service, store result, navigate."""
    claim_data = {
        "claim_id": claim_id,
        "object_type": object_type,
        "description": description,
        "damage_type": damage_type,
        "incident_date": incident_date,
        "history": history,
    }
    uploaded_images = [
        {
            "image_id": f"IMG_{i+1:02d}",
            "filename": vi["file"].name,
            "pil_image": vi["pil"],
        }
        for i, vi in enumerate(valid_images)
    ]

    with st.spinner("🔍 Analysing evidence — this may take a few seconds…"):
        try:
            result = analyze_claim(claim_data, uploaded_images)
        except Exception as exc:
            logger.error("Unexpected error from analyze_claim: %s", exc)
            st.error(f"Unexpected error: {exc}")
            return

    # Attach filenames to result for display
    result["_filenames"] = {img["image_id"]: img["filename"] for img in uploaded_images}
    result["_pil_images"] = {img["image_id"]: img["pil_image"] for img in uploaded_images}

    # Store in session
    st.session_state.recent_reviews.append(result)
    st.session_state.current_result = result
    st.session_state.page = "results"
    st.rerun()


# ---------------------------------------------------------------------------
# Results page
# ---------------------------------------------------------------------------

DECISION_BADGE = {
    "SUPPORTED": ("badge-supported", "✅ SUPPORTED"),
    "CONTRADICTED": ("badge-contradicted", "❌ CONTRADICTED"),
    "INSUFFICIENT_EVIDENCE": ("badge-insufficient", "⚠️ INSUFFICIENT EVIDENCE"),
}

SEV_CLASS = {
    "LOW": "sev-low",
    "MEDIUM": "sev-medium",
    "HIGH": "sev-high",
    "UNKNOWN": "sev-unknown",
}


def render_results():
    result = st.session_state.current_result
    if result is None:
        st.info("No result to display. Submit a claim first.")
        return

    decision = result.get("decision", "INSUFFICIENT_EVIDENCE")
    badge_cls, badge_text = DECISION_BADGE.get(decision, ("badge-insufficient", decision))
    is_demo = result.get("is_demo", False)
    filenames = result.get("_filenames", {})
    pil_images = result.get("_pil_images", {})
    conf_pct = int(result.get("confidence", 0) * 100)
    sev = result.get("severity", "UNKNOWN")

    # ── Top bar ────────────────────────────────────────────────────────────
    hcol1, hcol2 = st.columns([4, 1])
    with hcol1:
        st.markdown(f"""
        <div class="main-header" style="margin-bottom:16px;">
            <h1>Analysis Results</h1>
            <p>Claim ID: <strong>{result.get('claim_id', 'N/A')}</strong> &nbsp;·&nbsp;
               {result.get('object_type','')} &nbsp;·&nbsp;
               {result.get('timestamp','')[:19].replace('T',' ')}</p>
        </div>""", unsafe_allow_html=True)
    with hcol2:
        st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)
        if st.button("＋  New Claim", key="new_claim_from_results"):
            st.session_state.page = "new_claim"
            st.session_state.current_result = None
            st.session_state.form_reset_key += 1
            st.rerun()

    # Demo banner
    if is_demo:
        st.markdown("""
        <div class="demo-banner">
            🟣 DEMO MODE — These results are entirely simulated and do not reflect
            any real AI analysis. Configure an API key to enable genuine analysis.
        </div>""", unsafe_allow_html=True)

    # ── A. Final assessment ─────────────────────────────────────────────────
    with st.container(border=True):
        st.markdown('<p class="section-header">🏷️ Final Assessment</p>', unsafe_allow_html=True)

        ac1, ac2, ac3 = st.columns(3)
        with ac1:
            st.markdown(f'<span class="badge {badge_cls}">{badge_text}</span>', unsafe_allow_html=True)
            sev_cls = SEV_CLASS.get(sev, "sev-unknown")
            st.markdown(f"<div style='margin-top:10px;font-size:0.85rem;color:#64748B;'>Severity: <span class='{sev_cls}'>{sev}</span></div>", unsafe_allow_html=True)
        with ac2:
            st.markdown(f"""
            <div>
                <div style='font-size:0.75rem;font-weight:600;color:#64748B;text-transform:uppercase;letter-spacing:0.06em;'>AI Confidence Estimate</div>
                <div style='font-size:1.8rem;font-weight:800;color:#2563EB;'>{conf_pct}%</div>
                <div class="conf-bar-track"><div class="conf-bar-fill" style="width:{conf_pct}%;"></div></div>
                <div style='font-size:0.68rem;color:#94A3B8;margin-top:4px;'>Estimate only — not a calibrated probability</div>
            </div>""", unsafe_allow_html=True)
        with ac3:
            st.markdown(f"""
            <div>
                <div style='font-size:0.75rem;font-weight:600;color:#64748B;text-transform:uppercase;letter-spacing:0.06em;'>Object / Damage</div>
                <div style='font-weight:600;color:#0D1B2A;margin-top:4px;'>{result.get('object_type','')}</div>
                <div style='color:#475569;font-size:0.88rem;'>{result.get('damage_type','—')}</div>
                <div style='color:#94A3B8;font-size:0.82rem;margin-top:4px;'>Part: {result.get('object_part','—')}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<hr class='vs-divider'>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:0.9rem;color:#334155;line-height:1.6;'>{result.get('justification','')}</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── B. Evidence gallery ─────────────────────────────────────────────────
    st.markdown('<p class="section-header">🖼️ Evidence Gallery</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Highlighted images were cited by the analysis.</p>', unsafe_allow_html=True)

    supporting_ids = set(result.get("supporting_image_ids", []))
    findings_by_id = {f["image_id"]: f for f in result.get("evidence_findings", [])}

    all_img_ids = list(pil_images.keys())
    if all_img_ids:
        cols_per_row = 3
        for row_start in range(0, len(all_img_ids), cols_per_row):
            row_ids = all_img_ids[row_start: row_start + cols_per_row]
            cols = st.columns(cols_per_row)
            for col, iid in zip(cols, row_ids):
                with col:
                    with st.container(border=True):
                        is_highlighted = iid in supporting_ids
                        tag_extra = " · CITED" if is_highlighted else ""
                        st.markdown(f'<span class="img-id-tag">{iid}{tag_extra}</span>', unsafe_allow_html=True)
                        if iid in pil_images:
                            st.image(pil_images[iid], use_container_width=True)
                        fn = filenames.get(iid, "")
                        st.markdown(f"<div class='thumb-label' style='text-align:left;margin-top:6px;'>{fn}</div>", unsafe_allow_html=True)

                        if iid in findings_by_id:
                            f = findings_by_id[iid]
                            st.markdown(f"""
                            <div style='margin-top:10px;font-size:0.82rem;'>
                                <div style='font-weight:600;color:#1E293B;'>{f['finding']}</div>
                                <div style='color:#64748B;margin-top:3px;'>{f['relevance']}</div>
                            </div>""", unsafe_allow_html=True)
                        else:
                            st.markdown("<div style='font-size:0.8rem;color:#94A3B8;margin-top:8px;'>No specific finding for this image.</div>", unsafe_allow_html=True)
    else:
        st.info("No images available for display.")

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── C. Risk flags ───────────────────────────────────────────────────────
    st.markdown('<p class="section-header">🚩 Risk Indicators</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Flags require human review — they do not confirm fraud.</p>', unsafe_allow_html=True)
    risk_flags = result.get("risk_flags", [])
    if risk_flags:
        for flag in risk_flags:
            st.markdown(f'<div class="risk-flag">⚠️ {flag}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="no-risk">✅ No risk flags reported.</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── D. Missing evidence ─────────────────────────────────────────────────
    missing = result.get("missing_evidence", [])
    if missing:
        st.markdown('<p class="section-header">📎 Missing Evidence</p>', unsafe_allow_html=True)
        st.markdown('<p class="section-sub">The following additional evidence would strengthen the review.</p>', unsafe_allow_html=True)
        for m in missing:
            st.markdown(f'<div class="missing-item">📌 {m}</div>', unsafe_allow_html=True)
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── E. Review summary + download ────────────────────────────────────────
    with st.expander("📄 Full Review Summary & JSON Report", expanded=False):
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("**Claim Details**")
            st.markdown(f"- **Claim ID:** {result.get('claim_id','')}")
            st.markdown(f"- **Object:** {result.get('object_type','')}")
            st.markdown(f"- **Damage Type:** {result.get('damage_type','—')}")
            st.markdown(f"- **Object Part:** {result.get('object_part','—')}")
            st.markdown(f"- **Severity:** {result.get('severity','')}")
            st.markdown(f"- **Image Quality:** {result.get('image_quality','')}")
            st.markdown(f"- **Timestamp:** {result.get('timestamp','')}")
        with col_s2:
            st.markdown("**Evidence Findings**")
            for f in result.get("evidence_findings", []):
                st.markdown(f"- `{f['image_id']}`: {f['finding']}")

        st.markdown("**AI Justification**")
        st.markdown(f"> {result.get('justification','')}")

        # Downloadable JSON (strip PIL objects)
        report = {k: v for k, v in result.items() if not k.startswith("_")}
        report_json = json.dumps(report, indent=2, default=str)
        st.download_button(
            label="⬇️  Download JSON Report",
            data=report_json,
            file_name=f"{result.get('claim_id','report')}.json",
            mime="application/json",
        )


# ---------------------------------------------------------------------------
# Dashboard page
# ---------------------------------------------------------------------------

def render_dashboard():
    st.markdown("""
    <div class="main-header">
        <h1>Review Dashboard</h1>
        <p>Session-level overview of all claims reviewed during this session.</p>
    </div>""", unsafe_allow_html=True)

    render_stat_cards()

    reviews = st.session_state.recent_reviews
    if not reviews:
        st.info("No claims reviewed yet. Submit your first claim to see data here.")
        return

    st.markdown('<p class="section-header">All Reviews This Session</p>', unsafe_allow_html=True)
    for rev in reversed(reviews):
        decision = rev.get("decision", "?")
        badge_cls, badge_text = DECISION_BADGE.get(decision, ("badge-insufficient", decision))
        demo_tag = " &nbsp;<span style='font-size:0.72rem;color:#7C3AED;'>[DEMO]</span>" if rev.get("is_demo") else ""
        ts = rev.get("timestamp", "")[:19].replace("T", " ")
        conf_pct = int(rev.get("confidence", 0) * 100)
        sev = rev.get("severity", "UNKNOWN")

        with st.container():
            st.markdown(f"""
            <div class="claim-card" style="margin-bottom:16px;">
                <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:10px;">
                    <div>
                        <span style='font-weight:700;font-size:1rem;color:#0D1B2A;'>{rev.get('claim_id','?')}</span>{demo_tag}
                        <div style='font-size:0.82rem;color:#64748B;margin-top:2px;'>
                            {rev.get('object_type','')} &nbsp;·&nbsp; {rev.get('damage_type','—')} &nbsp;·&nbsp; {ts}
                        </div>
                    </div>
                    <div style='display:flex;gap:10px;align-items:center;flex-wrap:wrap;'>
                        <span class="badge {badge_cls}">{badge_text}</span>
                        <span class="{SEV_CLASS.get(sev,'sev-unknown')}">{sev}</span>
                        <span style='font-size:0.82rem;color:#2563EB;font-weight:600;'>{conf_pct}% confidence</span>
                    </div>
                </div>
                <div style='margin-top:10px;font-size:0.85rem;color:#475569;border-top:1px solid #F1F5F9;padding-top:10px;'>
                    {rev.get('justification','')[:250]}{"…" if len(rev.get('justification',''))>250 else ""}
                </div>
            </div>""", unsafe_allow_html=True)

            view_key = f"view_{rev.get('claim_id','x')}_{rev.get('timestamp','')}"
            if st.button(f"View Details → {rev.get('claim_id','')}", key=view_key):
                st.session_state.current_result = rev
                st.session_state.page = "results"
                st.rerun()


# ---------------------------------------------------------------------------
# App router
# ---------------------------------------------------------------------------

def main():
    render_sidebar()

    page = st.session_state.page
    if page == "new_claim":
        render_new_claim()
    elif page == "results":
        render_results()
    elif page == "dashboard":
        render_dashboard()
    else:
        render_new_claim()


if __name__ == "__main__":
    main()
