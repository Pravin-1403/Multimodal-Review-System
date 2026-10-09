"""
app.py — VeriSight AI  |  Multimodal Evidence Review & Claim Intelligence
==========================================================================
Next-generation AI forensic evidence review platform for insurance claims.
Supports Cars, Laptops, and Packages with multimodal damage verification,
component coverage matrix, human-in-the-loop adjuster adjudication,
and automated claim audit dossier generation.

Run with:
    streamlit run app.py
"""

import json
import uuid
import logging
import datetime
import io
import os
from typing import List, Dict, Any, Optional

import streamlit as st
from PIL import Image

from analysis_service import analyze_claim, get_provider_status

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="VeriSight AI — Evidence Review & Claims Intelligence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sample Scenarios Definition
# ---------------------------------------------------------------------------
SAMPLE_SCENARIOS = {
    "car_bumper": {
        "title": "Scenario 1: Front Bumper Impact",
        "category": "Car",
        "damage_type": "Front Bumper Collision & Displaced Grille",
        "description": "Front-end collision with another vehicle at low speed. The front bumper is visibly deformed, cracked, and the lower grille clips are shattered.",
        "history": "Single-vehicle collision claim. Policyholder has 5 years clean record.",
        "image_file": "samples/car_bumper_damaged.jpg",
        "image_name": "car_front_bumper_damage.jpg",
        "expected": "SUPPORTED",
        "badge_color": "green",
        "icon": "🚗",
        "summary": "Clear, direct damage to front bumper matches claim description.",
    },
    "laptop_screen": {
        "title": "Scenario 2: Suspected Screen Fraud",
        "category": "Laptop",
        "damage_type": "Completely Shattered Retina Display",
        "description": "Claimant asserts laptop display was dropped and completely shattered into spiderweb cracks with heavy LCD ink bleeding across the glass.",
        "history": "Supplemental accidental coverage filed 2 days prior to policy renewal.",
        "image_file": "samples/laptop_pristine.jpg",
        "image_name": "laptop_display_inspection.jpg",
        "expected": "CONTRADICTED",
        "badge_color": "red",
        "icon": "💻",
        "summary": "Target display is fully visible in flawless, pristine condition.",
    },
    "car_wrong_angle": {
        "title": "Scenario 3: Missing View / Wrong Angle",
        "category": "Car",
        "damage_type": "Rear Tailgate & Rear Bumper Dent",
        "description": "Claimant reports severe rear-end collision damage to rear tailgate and bumper. However, submitted photograph only captures the front grille and headlights.",
        "history": "Mobile first-notice-of-loss upload from roadside.",
        "image_file": "samples/car_front_clean.jpg",
        "image_name": "car_front_angle_view.jpg",
        "expected": "INSUFFICIENT_EVIDENCE",
        "badge_color": "amber",
        "icon": "🚙",
        "summary": "Target rear bumper is completely absent from submitted camera angles.",
    },
}

# ---------------------------------------------------------------------------
# CSS Styling — Enterprise InsurTech Design System
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global Typography & Background ─────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, sans-serif !important;
}

.stApp {
    background: #F4F6FB;
    color: #0F172A;
}

/* ── Sidebar ────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B132B 0%, #111D3E 50%, #1A284F 100%);
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}
section[data-testid="stSidebar"] * {
    color: #E2E8F0 !important;
}
section[data-testid="stSidebar"] .stButton button {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.12);
    color: #F8FAFC !important;
    border-radius: 12px;
    width: 100%;
    text-align: left;
    padding: 10px 16px;
    transition: all 0.25s ease;
    font-weight: 600;
    font-size: 0.9rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
}
section[data-testid="stSidebar"] .stButton button:hover {
    background: rgba(59, 130, 246, 0.25);
    border-color: rgba(96, 165, 250, 0.6);
    transform: translateX(2px);
}

/* ── Stat Cards ─────────────────────────────────────────────── */
.stat-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 20px 22px;
    text-align: center;
    box-shadow: 0 4px 18px rgba(15, 23, 42, 0.04);
    transition: all 0.2s ease;
    position: relative;
    overflow: hidden;
}
.stat-card:hover {
    box-shadow: 0 8px 26px rgba(15, 23, 42, 0.09);
    transform: translateY(-2px);
}
.stat-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 4px;
}
.stat-card.blue::before   { background: linear-gradient(90deg, #2563EB, #60A5FA); }
.stat-card.green::before  { background: linear-gradient(90deg, #059669, #34D399); }
.stat-card.amber::before  { background: linear-gradient(90deg, #D97706, #FBBF24); }
.stat-card.purple::before { background: linear-gradient(90deg, #7C3AED, #A78BFA); }

.stat-label {
    font-size: 0.76rem;
    font-weight: 700;
    color: #64748B;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.stat-value {
    font-size: 2.1rem;
    font-weight: 800;
    color: #0F172A;
    line-height: 1.1;
}
.stat-value.blue   { color: #2563EB; }
.stat-value.green  { color: #059669; }
.stat-value.amber  { color: #D97706; }
.stat-value.purple { color: #7C3AED; }

/* ── Section Titles ─────────────────────────────────────────── */
.section-header {
    font-size: 1.28rem;
    font-weight: 800;
    color: #0F172A;
    margin: 0 0 4px 0;
    letter-spacing: -0.01em;
}
.section-sub {
    font-size: 0.88rem;
    color: #64748B;
    margin: 0 0 18px 0;
}

/* ── Container Overrides ────────────────────────────────────── */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 18px !important;
    padding: 24px !important;
    box-shadow: 0 4px 18px rgba(15, 23, 42, 0.04) !important;
    margin-bottom: 22px !important;
}

/* ── Decision Badges ────────────────────────────────────────── */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 8px 18px;
    border-radius: 30px;
    font-size: 0.86rem;
    font-weight: 800;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}
.badge-supported {
    background: #ECFDF5;
    color: #065F46;
    border: 1.5px solid #6EE7B7;
}
.badge-contradicted {
    background: #FEF2F2;
    color: #991B1B;
    border: 1.5px solid #FCA5A5;
}
.badge-insufficient {
    background: #FFFBEB;
    color: #92400E;
    border: 1.5px solid #FCD34D;
}
.badge-adjudicated {
    background: #F5F3FF;
    color: #5B21B6;
    border: 1.5px solid #C4B5FD;
}

/* ── Severity Badges ────────────────────────────────────────── */
.sev-pill {
    padding: 3px 12px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.8rem;
    display: inline-block;
    letter-spacing: 0.04em;
}
.sev-low     { color: #065F46; background: #D1FAE5; border: 1px solid #A7F3D0; }
.sev-medium  { color: #92400E; background: #FEF3C7; border: 1px solid #FDE68A; }
.sev-high    { color: #991B1B; background: #FEE2E2; border: 1px solid #FECACA; }
.sev-unknown { color: #374151; background: #F3F4F6; border: 1px solid #E5E7EB; }

/* ── Pill Tags for Coverage Matrix ──────────────────────────── */
.pill-part {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 600;
    margin: 3px 4px 3px 0;
}
.pill-visible {
    background: #E0F2FE;
    color: #0369A1;
    border: 1px solid #BAE6FD;
}
.pill-unassessed {
    background: #FEF3C7;
    color: #B45309;
    border: 1px solid #FDE68A;
}
.pill-target {
    background: #EDE9FE;
    color: #6D28D9;
    border: 1px solid #DDD6FE;
    font-weight: 700;
}

/* ── Risk & Alert Banners ───────────────────────────────────── */
.risk-card {
    background: #FFFBEB;
    border: 1px solid #FDE68A;
    border-left: 5px solid #F59E0B;
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 12px;
    font-size: 0.9rem;
    color: #78350F;
    font-weight: 500;
}
.clean-card {
    background: #F0FDF4;
    border: 1px solid #BBF7D0;
    border-left: 5px solid #10B981;
    border-radius: 12px;
    padding: 14px 18px;
    color: #166534;
    font-size: 0.9rem;
    font-weight: 600;
}

/* ── Adjuster Adjudication Seal ─────────────────────────────── */
.adjudication-seal {
    background: linear-gradient(135deg, #1E1B4B 0%, #312E81 100%);
    border: 2px solid #6366F1;
    border-radius: 16px;
    padding: 22px 26px;
    color: #FFFFFF;
    box-shadow: 0 8px 30px rgba(79, 70, 229, 0.25);
    margin-top: 14px;
}
.seal-header {
    font-size: 1.15rem;
    font-weight: 800;
    color: #E0E7FF;
    display: flex;
    align-items: center;
    gap: 8px;
}
.seal-meta {
    font-size: 0.85rem;
    color: #C7D2FE;
    margin-top: 6px;
    line-height: 1.6;
}

/* ── Confidence Bar ─────────────────────────────────────────── */
.conf-track {
    background: #E2E8F0;
    border-radius: 99px;
    height: 10px;
    width: 100%;
    overflow: hidden;
    margin-top: 6px;
}
.conf-fill {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #2563EB 0%, #06B6D4 100%);
    transition: width 0.6s cubic-bezier(0.16, 1, 0.3, 1);
}

/* ── Image ID Label ─────────────────────────────────────────── */
.img-id-tag {
    display: inline-block;
    background: #EFF6FF;
    color: #1D4ED8;
    border: 1px solid #BFDBFE;
    border-radius: 6px;
    font-size: 0.76rem;
    font-weight: 800;
    padding: 3px 8px;
    margin-bottom: 8px;
}

/* ── Main Header ────────────────────────────────────────────── */
.main-header {
    background: linear-gradient(135deg, #0A1128 0%, #162447 60%, #1F3868 100%);
    border-radius: 20px;
    padding: 30px 38px;
    margin-bottom: 26px;
    color: white;
    box-shadow: 0 6px 24px rgba(10, 17, 40, 0.12);
}
.main-header h1 {
    font-size: 2.1rem;
    font-weight: 800;
    color: #FFFFFF;
    margin: 0;
    letter-spacing: -0.02em;
}
.main-header p {
    color: #94A3B8;
    font-size: 0.95rem;
    margin: 8px 0 0 0;
    line-height: 1.5;
}

/* ── Scenario Card ──────────────────────────────────────────── */
.preset-card {
    background: #FFFFFF;
    border: 1.5px solid #E2E8F0;
    border-radius: 14px;
    padding: 16px;
    transition: all 0.2s ease;
    height: 100%;
}
.preset-card:hover {
    border-color: #3B82F6;
    box-shadow: 0 4px 16px rgba(59, 130, 246, 0.15);
}

/* ── Primary Action Button ──────────────────────────────────── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 14px 34px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 16px rgba(37, 99, 235, 0.35) !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(37, 99, 235, 0.45) !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
def _init_session():
    defaults = {
        "page": "new_claim",
        "recent_reviews": [],          # List of result dicts
        "current_result": None,
        "form_reset_key": 0,           # Increment to reset form widgets
        "active_preset": None,         # Selected scenario preset key
        "loaded_preset_image": None,   # PIL image loaded from preset
        "loaded_preset_meta": None,    # dict with name, filename
        "dash_search": "",
        "dash_category": "All",
        "dash_decision": "All",
        "dash_sort": "Newest First",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_session()

# ---------------------------------------------------------------------------
# Session Statistics
# ---------------------------------------------------------------------------
def _stats() -> dict:
    reviews = st.session_state.recent_reviews
    total = len(reviews)
    supported = sum(1 for r in reviews if r.get("decision") == "SUPPORTED" and not r.get("is_demo"))
    contradicted = sum(1 for r in reviews if r.get("decision") == "CONTRADICTED" and not r.get("is_demo"))
    insufficient = sum(1 for r in reviews if r.get("decision") == "INSUFFICIENT_EVIDENCE" and not r.get("is_demo"))
    adjudicated = sum(1 for r in reviews if r.get("adjudication", {}).get("is_finalized", False))
    return {
        "total": total,
        "supported": supported,
        "contradicted": contradicted,
        "insufficient": insufficient,
        "adjudicated": adjudicated,
    }

# ---------------------------------------------------------------------------
# Unique ID Generator
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
        <div style="padding: 10px 0 20px 0;">
            <div style="font-size: 1.55rem; font-weight: 800; letter-spacing:-0.5px; color:#FFFFFF;">
                🔍 VeriSight AI
            </div>
            <div style="font-size: 0.72rem; color: #94A3B8; margin-top: 2px; letter-spacing:0.06em; font-weight: 700;">
                FORENSIC CLAIMS INTELLIGENCE · v2.4
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("<div style='font-size:0.74rem;font-weight:700;letter-spacing:0.08em;color:#94A3B8;margin-bottom:12px;'>WORKSPACE NAVIGATION</div>", unsafe_allow_html=True)

        if st.button("＋  New Claim Workspace", key="nav_new"):
            st.session_state.page = "new_claim"
            st.rerun()

        if st.button("📊  Claims Audit Dashboard", key="nav_dash"):
            st.session_state.page = "dashboard"
            st.rerun()

        # Recent Reviews List
        if st.session_state.recent_reviews:
            st.markdown("<div style='margin-top:24px;font-size:0.74rem;font-weight:700;letter-spacing:0.08em;color:#94A3B8;margin-bottom:10px;'>ACTIVE CLAIMS DOSSIER</div>", unsafe_allow_html=True)
            for rev in reversed(st.session_state.recent_reviews[-6:]):
                decision = rev.get("decision", "?")
                icon = {
                    "SUPPORTED": "✅",
                    "CONTRADICTED": "❌",
                    "INSUFFICIENT_EVIDENCE": "⚠️"
                }.get(decision, "📄")
                adj_tag = " [ADJ]" if rev.get("adjudication", {}).get("is_finalized") else ""
                demo_tag = " [DEMO]" if rev.get("is_demo") else ""
                label = f"{icon} {rev.get('claim_id', '?')}{adj_tag}{demo_tag}"
                if st.button(label, key=f"rev_{rev.get('claim_id')}"):
                    st.session_state.current_result = rev
                    st.session_state.page = "results"
                    st.rerun()

        # System Engine Status
        st.markdown("---")
        status = get_provider_status()
        if status["configured"]:
            st.markdown(f"""
            <div style="font-size:0.8rem; background:rgba(16,185,129,0.12); padding:10px 14px; border-radius:10px; border:1px solid rgba(16,185,129,0.3);">
                <div style="font-weight:700; color:#34D399;">🟢 Multimodal AI Engine Online</div>
                <div style="color:#94A3B8; font-size:0.72rem; margin-top:2px;">{status['message']}</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="font-size:0.8rem; background:rgba(245,158,11,0.12); padding:10px 14px; border-radius:10px; border:1px solid rgba(245,158,11,0.3);">
                <div style="font-weight:700; color:#FBBF24;">🟠 Demo Simulation Mode</div>
                <div style="color:#94A3B8; font-size:0.72rem; margin-top:2px;">{status['message']}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.72rem; color:#64748B; border-top:1px solid rgba(255,255,255,0.08); padding-top:16px; line-height:1.6;">
            ⚖️ <strong>Compliance Notice:</strong><br>
            AI outputs are forensic decision-support aids for certified adjusters. Not a legal final adjudication.
        </div>""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Stat Cards Row
# ---------------------------------------------------------------------------
def render_stat_cards():
    s = _stats()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="stat-card blue">
            <div class="stat-label">Total Claims</div>
            <div class="stat-value blue">{s['total']}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="stat-card green">
            <div class="stat-label">Supported</div>
            <div class="stat-value green">{s['supported']}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="stat-card amber">
            <div class="stat-label">Under Investigation</div>
            <div class="stat-value amber">{s['insufficient'] + s['contradicted']}</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="stat-card purple">
            <div class="stat-label">Adjuster Certified</div>
            <div class="stat-value purple">{s['adjudicated']}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# New Claim Workspace
# ---------------------------------------------------------------------------
MAX_IMAGES = 8
MAX_IMAGE_MB = 10

def render_new_claim():
    # Workspace Hero Banner
    st.markdown("""
    <div class="main-header">
        <h1>Multimodal Claims Evidence Workspace</h1>
        <p>Review damage claims with computer-vision intelligence. VeriSight evaluates photographic evidence,
           analyzes component perspectives, cross-references damage assertions, and guards against misclassification.</p>
    </div>""", unsafe_allow_html=True)

    render_stat_cards()

    # ── Interactive 1-Click Forensic Scenario Presets ─────────────────────
    st.markdown('<p class="section-header">⚡ 1-Click Forensic Prototype Scenarios</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Select any sample scenario to instantly populate claim parameters and load high-resolution evidence images.</p>', unsafe_allow_html=True)

    pcol1, pcol2, pcol3 = st.columns(3)

    for col, (sc_key, sc_info) in zip([pcol1, pcol2, pcol3], SAMPLE_SCENARIOS.items()):
        with col:
            with st.container(border=True):
                st.markdown(f"""
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                    <span style="font-size:1.4rem;">{sc_info['icon']}</span>
                    <span class="sev-pill sev-{ 'low' if sc_info['badge_color']=='green' else ('high' if sc_info['badge_color']=='red' else 'medium') }">
                        {sc_info['expected']}
                    </span>
                </div>
                <div style="font-weight:700; font-size:1rem; color:#0F172A; margin-bottom:4px;">{sc_info['title']}</div>
                <div style="font-size:0.8rem; color:#64748B; min-height:42px; margin-bottom:12px;">{sc_info['summary']}</div>
                """, unsafe_allow_html=True)

                if st.button(f"Load {sc_info['category']} Scenario", key=f"btn_preset_{sc_key}", use_container_width=True):
                    # Load PIL image
                    try:
                        pil_img = Image.open(sc_info["image_file"]).convert("RGB")
                        st.session_state.active_preset = sc_key
                        st.session_state.loaded_preset_image = pil_img
                        st.session_state.loaded_preset_meta = {
                            "name": sc_info["image_name"],
                            "path": sc_info["image_file"],
                        }
                        st.session_state.form_reset_key += 1
                        st.toast(f"Loaded {sc_info['title']}!", icon="✨")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to load preset image: {e}")

    # Active Scenario Banner
    active_key = st.session_state.active_preset
    preset_data = SAMPLE_SCENARIOS.get(active_key) if active_key else None

    if preset_data and st.session_state.loaded_preset_image:
        st.markdown(f"""
        <div style="background:#EFF6FF; border:1.5px solid #93C5FD; border-radius:14px; padding:14px 20px; margin: 18px 0; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
            <div>
                <span style="font-weight:800; color:#1D4ED8; font-size:0.95rem;">⚡ Active Scenario: {preset_data['title']}</span>
                <span style="margin-left:8px; font-size:0.84rem; color:#3B82F6;">({preset_data['image_name']}) pre-loaded</span>
            </div>
            <div>
                <span style="font-size:0.8rem; color:#64748B;">Ready for instant analysis</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    rk = st.session_state.form_reset_key

    # ── Claim Intake Form ──────────────────────────────────────────────────
    with st.container(border=True):
        fcol1, fcol2 = st.columns([4, 1])
        with fcol1:
            st.markdown('<p class="section-header">📋 Claim Evidence Submission</p>', unsafe_allow_html=True)
            st.markdown('<p class="section-sub">Provide damage details and attach high-fidelity photographic evidence.</p>', unsafe_allow_html=True)
        with fcol2:
            if preset_data:
                if st.button("🔄 Clear Preset", key="clear_preset_btn"):
                    st.session_state.active_preset = None
                    st.session_state.loaded_preset_image = None
                    st.session_state.loaded_preset_meta = None
                    st.session_state.form_reset_key += 1
                    st.rerun()

        col_a, col_b = st.columns([1, 1])

        # Pre-fill defaults if preset active
        default_obj = preset_data["category"] if preset_data else "Car"
        default_dmg = preset_data["damage_type"] if preset_data else ""
        default_desc = preset_data["description"] if preset_data else ""
        default_hist = preset_data["history"] if preset_data else ""

        with col_a:
            claim_id_display = _new_claim_id()
            st.text_input(
                "Claim Dossier ID (Auto-Generated)",
                value=claim_id_display,
                disabled=True,
                key=f"claim_id_{rk}",
            )
            cat_options = ["Car", "Laptop", "Package"]
            obj_idx = cat_options.index(default_obj) if default_obj in cat_options else 0
            object_type = st.selectbox(
                "Object Category *",
                cat_options,
                index=obj_idx,
                key=f"object_type_{rk}",
            )
            damage_type = st.text_input(
                "Claimed Damage Description *",
                value=default_dmg,
                placeholder="e.g. Cracked front bumper, Shattered LCD screen, Crushed parcel",
                key=f"damage_type_{rk}",
            )

        with col_b:
            incident_date = st.date_input(
                "Incident Date (Optional)",
                value=datetime.date.today(),
                key=f"incident_date_{rk}",
            )
            history = st.text_area(
                "Prior Claim History / Adjuster Notes",
                value=default_hist,
                placeholder="Policy endorsements, prior incidents, or vehicle telemetry notes…",
                height=108,
                key=f"history_{rk}",
            )

        description = st.text_area(
            "Detailed Incident Narrative *",
            value=default_desc,
            placeholder="Detailed narrative describing how and where the damage occurred, impact conditions, and surface manifestations…",
            height=120,
            key=f"description_{rk}",
        )

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown("**Photographic Evidence Repository** *(Max 8 images, 10MB each)*")

        # Preset image preview
        valid_images = []
        if st.session_state.loaded_preset_image:
            meta = st.session_state.loaded_preset_meta or {"name": "preset_image.jpg"}
            valid_images.append({
                "name": meta["name"],
                "pil": st.session_state.loaded_preset_image,
                "is_preset": True,
            })

        uploaded_files = st.file_uploader(
            "Upload Evidence Images",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            accept_multiple_files=True,
            key=f"file_uploader_{rk}",
            label_visibility="collapsed",
        )

        if uploaded_files:
            for uf in uploaded_files[:MAX_IMAGES]:
                size_mb = uf.size / (1024 * 1024)
                if size_mb > MAX_IMAGE_MB:
                    st.warning(f"File {uf.name} exceeds {MAX_IMAGE_MB}MB limit.")
                    continue
                try:
                    img = Image.open(uf).convert("RGB")
                    valid_images.append({
                        "name": uf.name,
                        "pil": img,
                        "is_preset": False,
                    })
                except Exception as e:
                    st.error(f"Could not open {uf.name}: {e}")

        # Render evidence thumbnail cards
        if valid_images:
            st.markdown(f"**Evidence Images Queued ({len(valid_images)}):**")
            cols = st.columns(min(len(valid_images), 4))
            for idx, vi in enumerate(valid_images):
                with cols[idx % 4]:
                    with st.container(border=True):
                        st.image(vi["pil"], use_container_width=True)
                        badge_tag = " [PRESET]" if vi.get("is_preset") else ""
                        st.markdown(f"<div style='font-size:0.75rem; font-weight:700; color:#475569; margin-top:4px;'>IMG_{idx+1:02d}{badge_tag}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div style='font-size:0.72rem; color:#64748B; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;'>{vi['name']}</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        submit_btn = st.button("🚀  Run Multimodal Forensic Analysis", type="primary", key=f"submit_{rk}")

        if submit_btn:
            errors = []
            if not description.strip():
                errors.append("Claim incident narrative is required.")
            if not valid_images:
                errors.append("At least one photographic evidence image is required. (Upload images or click a sample scenario above!)")

            if errors:
                for err in errors:
                    st.error(f"❌ {err}")
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

# ---------------------------------------------------------------------------
# Analysis Execution Orchestrator
# ---------------------------------------------------------------------------
def _run_analysis(claim_id, object_type, description, damage_type,
                  incident_date, history, valid_images):
    """Package payload, invoke AI service, store in session state."""
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
            "filename": vi["name"],
            "pil_image": vi["pil"],
        }
        for i, vi in enumerate(valid_images)
    ]

    with st.spinner("🔍 Running multimodal forensic analysis — inspecting component coverage, perspective, and damage characteristics…"):
        try:
            result = analyze_claim(claim_data, uploaded_images)
        except Exception as exc:
            logger.error("Analysis execution error: %s", exc)
            st.error(f"Analysis error: {exc}")
            return

    # Attach preview metadata
    result["_filenames"] = {img["image_id"]: img["filename"] for img in uploaded_images}
    result["_pil_images"] = {img["image_id"]: img["pil_image"] for img in uploaded_images}

    # Initialize human adjudication container
    result["adjudication"] = {
        "is_finalized": False,
        "action": "PENDING_REVIEW",
        "adjudicator_id": "ADJ-4482",
        "payout_approved": 0,
        "notes": "",
        "signed_at": None,
    }

    # Persist and route
    st.session_state.recent_reviews.append(result)
    st.session_state.current_result = result
    st.session_state.page = "results"
    st.rerun()

# ---------------------------------------------------------------------------
# Results & Forensic Audit Dossier
# ---------------------------------------------------------------------------
DECISION_BADGE = {
    "SUPPORTED": ("badge-supported", "🛡️ SUPPORTED"),
    "CONTRADICTED": ("badge-contradicted", "⛔ CONTRADICTED"),
    "INSUFFICIENT_EVIDENCE": ("badge-insufficient", "⚠️ INSUFFICIENT EVIDENCE"),
}

SEV_CLASS = {
    "LOW": "sev-low",
    "MEDIUM": "sev-medium",
    "HIGH": "sev-high",
    "UNKNOWN": "sev-unknown",
}

SEV_PAYOUT_BASELINE = {
    "LOW": (250, 650),
    "MEDIUM": (750, 2400),
    "HIGH": (2500, 8500),
    "UNKNOWN": (0, 0),
}

def render_results():
    result = st.session_state.current_result
    if result is None:
        st.info("No claim result selected. Please submit a claim from the workspace.")
        return

    claim_id = result.get("claim_id", "N/A")
    decision = result.get("decision", "INSUFFICIENT_EVIDENCE")
    badge_cls, badge_text = DECISION_BADGE.get(decision, ("badge-insufficient", decision))
    is_demo = result.get("is_demo", False)
    filenames = result.get("_filenames", {})
    pil_images = result.get("_pil_images", {})
    conf_pct = int(result.get("confidence", 0) * 100)
    sev = result.get("severity", "UNKNOWN")
    adj = result.get("adjudication", {})

    # ── Header Bar ────────────────────────────────────────────────────────
    hcol1, hcol2 = st.columns([4, 1])
    with hcol1:
        st.markdown(f"""
        <div class="main-header" style="margin-bottom:18px;">
            <div style="font-size:0.8rem; font-weight:700; color:#60A5FA; letter-spacing:0.06em; text-transform:uppercase;">
                AI FORENSIC DOSSIER
            </div>
            <h1>Claim Assessment: {claim_id}</h1>
            <p>Object: <strong>{result.get('object_type','')}</strong> &nbsp;·&nbsp;
               Claimed Damage: <strong>{result.get('damage_type','—')}</strong> &nbsp;·&nbsp;
               Analyzed: {result.get('timestamp','')[:19].replace('T',' ')}</p>
        </div>""", unsafe_allow_html=True)
    with hcol2:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if st.button("＋  New Claim", key="res_new_claim", use_container_width=True):
            st.session_state.page = "new_claim"
            st.session_state.current_result = None
            st.session_state.active_preset = None
            st.session_state.loaded_preset_image = None
            st.session_state.loaded_preset_meta = None
            st.session_state.form_reset_key += 1
            st.rerun()

    if is_demo:
        st.markdown("""
        <div style="background:#EDE9FE; border:1.5px solid #C4B5FD; border-radius:12px; padding:14px 18px; color:#5B21B6; font-weight:600; font-size:0.88rem; margin-bottom:18px;">
            🟣 DEMO SIMULATION MODE — Analysis generated using fallback deterministic patterns. Provide a valid GEMINI_API_KEY in .streamlit/secrets.toml for live multimodal inference.
        </div>""", unsafe_allow_html=True)

    # ── Section 1: Executive Forensic Verdict ─────────────────────────────
    with st.container(border=True):
        st.markdown('<p class="section-header">⚖️ Forensic Assessment Summary</p>', unsafe_allow_html=True)

        vc1, vc2, vc3 = st.columns([1.2, 1.2, 1.4])
        with vc1:
            st.markdown(f'<span class="badge {badge_cls}">{badge_text}</span>', unsafe_allow_html=True)
            sev_cls = SEV_CLASS.get(sev, "sev-unknown")
            st.markdown(f"""
            <div style="margin-top:14px; font-size:0.86rem; color:#64748B;">
                Damage Severity: <span class="sev-pill {sev_cls}">{sev}</span>
            </div>
            <div style="margin-top:8px; font-size:0.82rem; color:#64748B;">
                Image Quality: <strong style="color:#0F172A;">{result.get('image_quality', 'UNKNOWN')}</strong>
            </div>
            """, unsafe_allow_html=True)

        with vc2:
            st.markdown(f"""
            <div>
                <div style='font-size:0.75rem; font-weight:700; color:#64748B; text-transform:uppercase; letter-spacing:0.06em;'>AI Confidence Rating</div>
                <div style='font-size:2.1rem; font-weight:800; color:#2563EB; line-height:1.1;'>{conf_pct}%</div>
                <div class="conf-track"><div class="conf-fill" style="width:{conf_pct}%;"></div></div>
                <div style='font-size:0.7rem; color:#94A3B8; margin-top:6px;'>
                    { "High evidentiary reliability" if conf_pct >= 80 else ("Moderate evidentiary alignment" if conf_pct >= 50 else "Capped confidence / Insufficient perspective") }
                </div>
            </div>""", unsafe_allow_html=True)

        with vc3:
            st.markdown(f"""
            <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:12px; padding:12px 16px;">
                <div style='font-size:0.75rem; font-weight:700; color:#64748B; text-transform:uppercase;'>Target Component in Claim</div>
                <div style='font-weight:800; color:#0F172A; font-size:1.05rem; margin-top:2px;'>
                    {result.get('object_part') or result.get('damage_type') or 'General Surface'}
                </div>
                <div style='font-size:0.78rem; color:#64748B; margin-top:4px;'>
                    Category: <strong>{result.get('object_type','')}</strong>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<hr style='border:none; border-top:1px solid #E2E8F0; margin:18px 0;'>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-size:0.92rem; color:#1E293B; line-height:1.7;">
            <strong>Forensic Justification:</strong><br>
            {result.get('justification','')}
        </div>""", unsafe_allow_html=True)

    # ── Section 2: Component Coverage & Perspective Matrix ────────────────
    with st.container(border=True):
        st.markdown('<p class="section-header">🎯 Component Coverage & Perspective Matrix</p>', unsafe_allow_html=True)
        st.markdown('<p class="section-sub">Audits whether submitted photographs depict the claimed target part from an adequate angle without occlusion.</p>', unsafe_allow_html=True)

        mat_col1, mat_col2 = st.columns([1, 1])

        with mat_col1:
            st.markdown("**Claim Target vs Photographic Visibility:**")
            target_part = result.get('object_part') or 'Claimed Surface'
            st.markdown(f"<span class='pill-part pill-target'>Target: {target_part}</span>", unsafe_allow_html=True)

            if decision == "SUPPORTED":
                st.markdown("""
                <div style="background:#ECFDF5; border:1px solid #A7F3D0; border-radius:10px; padding:10px 14px; margin-top:10px; font-size:0.85rem; color:#065F46;">
                    ✅ <strong>Verified Coverage:</strong> The target part was clearly visible from a direct angle, displaying physical damage consistent with the incident report.
                </div>""", unsafe_allow_html=True)
            elif decision == "CONTRADICTED":
                st.markdown("""
                <div style="background:#FEF2F2; border:1px solid #FECACA; border-radius:10px; padding:10px 14px; margin-top:10px; font-size:0.85rem; color:#991B1B;">
                    ⛔ <strong>Clear Contradiction:</strong> The target component was clearly visible in the photograph and showed no signs of the claimed defect, being intact/pristine.
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background:#FFFBEB; border:1px solid #FDE68A; border-radius:10px; padding:10px 14px; margin-top:10px; font-size:0.85rem; color:#92400E;">
                    ⚠️ <strong>Coverage Deficiency:</strong> The photographic evidence does not provide an adequate view of the claimed component (misaligned angle, distant shot, or occluded surface).
                </div>""", unsafe_allow_html=True)

        with mat_col2:
            st.markdown("**Detected Surface Breakdown:**")
            visible_parts = result.get("visible_parts", [])
            unassessed_parts = result.get("unassessed_parts", [])

            st.markdown("<div style='font-size:0.8rem; font-weight:600; color:#64748B; margin-bottom:4px;'>VISIBLE SURFACES DETECTED:</div>", unsafe_allow_html=True)
            if visible_parts:
                pills_html = "".join([f"<span class='pill-part pill-visible'>✓ {p}</span>" for p in visible_parts])
                st.markdown(f"<div>{pills_html}</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='font-size:0.82rem; color:#94A3B8;'>No specific surfaces identified.</div>", unsafe_allow_html=True)

            st.markdown("<div style='font-size:0.8rem; font-weight:600; color:#64748B; margin-top:10px; margin-bottom:4px;'>UNASSESSED / MISSING SURFACES:</div>", unsafe_allow_html=True)
            if unassessed_parts:
                pills_un = "".join([f"<span class='pill-part pill-unassessed'>⚠ {p}</span>" for p in unassessed_parts])
                st.markdown(f"<div>{pills_un}</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='font-size:0.82rem; color:#059669;'>All relevant surfaces adequately captured.</div>", unsafe_allow_html=True)

    # ── Section 3: Evidence Gallery & Citations ───────────────────────────
    st.markdown('<p class="section-header">🖼️ Evidence Repository & Visual Citations</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Evidence images cited directly by the AI model during forensic evaluation.</p>', unsafe_allow_html=True)

    supporting_ids = set(result.get("supporting_image_ids", []))
    findings_by_id = {f["image_id"]: f for f in result.get("evidence_findings", [])}
    all_img_ids = list(pil_images.keys())

    if all_img_ids:
        cols_per_row = 3
        for row_start in range(0, len(all_img_ids), cols_per_row):
            row_ids = all_img_ids[row_start : row_start + cols_per_row]
            cols = st.columns(cols_per_row)
            for col, iid in zip(cols, row_ids):
                with col:
                    with st.container(border=True):
                        is_highlighted = iid in supporting_ids
                        tag_extra = " · CITED IN FINDINGS" if is_highlighted else ""
                        st.markdown(f'<span class="img-id-tag">{iid}{tag_extra}</span>', unsafe_allow_html=True)

                        if iid in pil_images:
                            st.image(pil_images[iid], use_container_width=True)

                        fn = filenames.get(iid, "")
                        st.markdown(f"<div style='font-size:0.75rem; color:#64748B; margin-top:6px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;'>{fn}</div>", unsafe_allow_html=True)

                        if iid in findings_by_id:
                            f = findings_by_id[iid]
                            st.markdown(f"""
                            <div style='background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:10px; margin-top:10px; font-size:0.82rem;'>
                                <div style='font-weight:700; color:#0F172A;'>{f['finding']}</div>
                                <div style='color:#64748B; margin-top:4px;'>{f['relevance']}</div>
                            </div>""", unsafe_allow_html=True)
                        else:
                            st.markdown("<div style='font-size:0.78rem; color:#94A3B8; margin-top:8px;'>No specific abnormality isolated.</div>", unsafe_allow_html=True)
    else:
        st.info("No photographic evidence stored with this claim record.")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── Section 4: Risk Flags & Missing Evidence ──────────────────────────
    r_col1, r_col2 = st.columns(2)
    with r_col1:
        with st.container(border=True):
            st.markdown('<p class="section-header">🚩 Risk & Integrity Flags</p>', unsafe_allow_html=True)
            st.markdown('<p class="section-sub">Flags highlight evidentiary anomalies for adjuster scrutiny.</p>', unsafe_allow_html=True)
            risk_flags = result.get("risk_flags", [])
            if risk_flags:
                for flag in risk_flags:
                    st.markdown(f'<div class="risk-card">⚠️ {flag}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="clean-card">✅ No fraud indicators or metadata discrepancies detected.</div>', unsafe_allow_html=True)

    with r_col2:
        with st.container(border=True):
            st.markdown('<p class="section-header">📎 Supplemental Evidence Recommendations</p>', unsafe_allow_html=True)
            st.markdown('<p class="section-sub">Required documentation to strengthen forensic claim file.</p>', unsafe_allow_html=True)
            missing = result.get("missing_evidence", [])
            if missing:
                for m in missing:
                    st.markdown(f"""
                    <div style="background:#F8FAFC; border:1px dashed #CBD5E1; border-radius:10px; padding:10px 14px; margin-bottom:8px; font-size:0.86rem; color:#334155;">
                        📌 {m}
                    </div>""", unsafe_allow_html=True)
            else:
                st.markdown('<div class="clean-card">✅ Evidence file is complete for preliminary adjudication.</div>', unsafe_allow_html=True)

    # ── Section 5: Human-in-the-Loop Adjuster Adjudication Panel ───────────
    with st.container(border=True):
        st.markdown('<p class="section-header">⚖️ Human-in-the-Loop Adjuster Adjudication</p>', unsafe_allow_html=True)
        st.markdown('<p class="section-sub">Empowers certified claims examiners to validate AI findings, override assessments, set settlement payout, and sign off.</p>', unsafe_allow_html=True)

        low_b, high_b = SEV_PAYOUT_BASELINE.get(sev, (0, 0))
        default_payout = int((low_b + high_b) / 2) if decision == "SUPPORTED" else 0

        # Adjudication Status Display
        if adj.get("is_finalized"):
            st.markdown(f"""
            <div class="adjudication-seal">
                <div class="seal-header">
                    <span>🛡️ CERTIFIED CLAIM ADJUDICATION SEAL</span>
                    <span style="background:rgba(255,255,255,0.2); padding:2px 10px; border-radius:12px; font-size:0.75rem;">AUDIT CERTIFICATE</span>
                </div>
                <div class="seal-meta">
                    <strong>Certified Decision:</strong> {adj.get('action')} &nbsp;·&nbsp;
                    <strong>Approved Settlement Payout:</strong> ${adj.get('payout_approved'):,} USD<br>
                    <strong>Adjudicator Sign-Off:</strong> {adj.get('adjudicator_id')} &nbsp;·&nbsp;
                    <strong>Signed At:</strong> {adj.get('signed_at')}
                </div>
                <div style="margin-top:12px; font-size:0.88rem; color:#EEF2FF; border-top:1px solid rgba(255,255,255,0.15); padding-top:8px;">
                    <strong>Examiner Remarks:</strong> {adj.get('notes') or 'No additional notes entered.'}
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

        adj_col1, adj_col2 = st.columns([1, 1])

        with adj_col1:
            st.markdown(f"**Baseline AI Repair Estimate Range:** `${low_b:,} - ${high_b:,} USD`")
            approved_amount = st.slider(
                "Authorized Settlement / Repair Payout ($ USD)",
                min_value=0,
                max_value=10000,
                value=int(adj.get("payout_approved", default_payout)),
                step=50,
                key=f"slider_payout_{claim_id}",
            )
            adj_action = st.selectbox(
                "Examiner Adjudication Determination",
                [
                    "APPROVE_CLAIM (Support AI Assessment)",
                    "APPROVE_MODIFIED_SETTLEMENT",
                    "OVERRIDE_TO_CONTRADICTED (Deny Claim)",
                    "REQUEST_FORENSIC_REINSPECTION",
                ],
                index=0 if decision == "SUPPORTED" else (2 if decision == "CONTRADICTED" else 3),
                key=f"select_action_{claim_id}",
            )

        with adj_col2:
            adj_id = st.text_input(
                "Authorized Examiner / Adjuster ID",
                value=adj.get("adjudicator_id", "ADJ-4482 (Senior Claims Analyst)"),
                key=f"input_adj_id_{claim_id}",
            )
            adj_notes = st.text_area(
                "Official Adjudication Memo & Legal Endorsement",
                value=adj.get("notes", f"Reviewed forensic AI findings for {claim_id}. Photographic evidence correlates with {decision} determination."),
                height=96,
                key=f"input_adj_notes_{claim_id}",
            )

        if st.button("✍️  Sign & Finalize Claim Adjudication", type="primary", key=f"sign_claim_{claim_id}"):
            adj["is_finalized"] = True
            adj["action"] = adj_action.split(" ")[0]
            adj["adjudicator_id"] = adj_id
            adj["payout_approved"] = approved_amount
            adj["notes"] = adj_notes
            adj["signed_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            result["adjudication"] = adj
            st.toast("Claim successfully adjudicated & certified!", icon="🛡️")
            st.rerun()

    # ── Section 6: Official Claims Dossier & Export ────────────────────────
    with st.expander("📄 Full Claim Dossier Export & Audit Serialization", expanded=False):
        dossier_text = _generate_claim_dossier_markdown(result)

        dcol1, dcol2 = st.columns([1, 1])
        with dcol1:
            st.markdown("**Structured Dossier Preview (Markdown / Text):**")
            st.text_area("Audit Dossier Text", value=dossier_text, height=260, disabled=True)
        with dcol2:
            st.markdown("**Export Options:**")
            st.download_button(
                label="📥  Download Certified Dossier (.md)",
                data=dossier_text,
                file_name=f"DOSSIER_{claim_id}.md",
                mime="text/markdown",
                use_container_width=True,
            )

            # Strip PIL objects for clean JSON download
            clean_result = {k: v for k, v in result.items() if not k.startswith("_")}
            st.download_button(
                label="📥  Download JSON Audit Log (.json)",
                data=json.dumps(clean_result, indent=2, default=str),
                file_name=f"AUDIT_{claim_id}.json",
                mime="application/json",
                use_container_width=True,
            )

# ---------------------------------------------------------------------------
# Dossier Formatter Helper
# ---------------------------------------------------------------------------
def _generate_claim_dossier_markdown(res: dict) -> str:
    claim_id = res.get("claim_id", "N/A")
    decision = res.get("decision", "INSUFFICIENT_EVIDENCE")
    obj_type = res.get("object_type", "Unknown")
    dmg_type = res.get("damage_type", "N/A")
    sev = res.get("severity", "UNKNOWN")
    conf = int(res.get("confidence", 0) * 100)
    target_part = res.get("object_part", "N/A")
    vis_parts = ", ".join(res.get("visible_parts", [])) or "None identified"
    unass_parts = ", ".join(res.get("unassessed_parts", [])) or "None identified"
    adj = res.get("adjudication", {})

    lines = [
        f"# VERISIGHT AI — FORENSIC CLAIM AUDIT DOSSIER",
        f"**Dossier Reference:** {claim_id}",
        f"**Audit Timestamp:** {res.get('timestamp', datetime.datetime.now().isoformat())}",
        f"**Object Category:** {obj_type}",
        f"**Claimed Damage:** {dmg_type}",
        "",
        "## 1. FORENSIC AI VERDICT",
        f"- **Primary Determination:** {decision}",
        f"- **Confidence Rating:** {conf}%",
        f"- **Damage Severity:** {sev}",
        f"- **Image Quality:** {res.get('image_quality', 'UNKNOWN')}",
        f"- **Justification:** {res.get('justification', '')}",
        "",
        "## 2. COMPONENT COVERAGE & PERSPECTIVE AUDIT",
        f"- **Target Claimed Component:** {target_part}",
        f"- **Visible Surfaces Identified:** {vis_parts}",
        f"- **Unassessed / Occluded Surfaces:** {unass_parts}",
        "",
        "## 3. EVIDENCE FINDINGS & CITATIONS",
    ]
    for ef in res.get("evidence_findings", []):
        lines.append(f"- **[{ef.get('image_id')}]**: {ef.get('finding')} (Relevance: {ef.get('relevance')})")

    lines.extend([
        "",
        "## 4. INTEGRITY & RISK INDICATORS",
    ])
    flags = res.get("risk_flags", [])
    if flags:
        for fl in flags:
            lines.append(f"- ⚠️ {fl}")
    else:
        lines.append("- ✅ Zero fraud indicators detected.")

    lines.extend([
        "",
        "## 5. ADJUSTER ADJUDICATION & SIGN-OFF",
        f"- **Adjudication Finalized:** {'YES' if adj.get('is_finalized') else 'PENDING'}",
        f"- **Certified Action:** {adj.get('action', 'PENDING')}",
        f"- **Approved Settlement Payout:** ${adj.get('payout_approved', 0):,} USD",
        f"- **Examiner ID:** {adj.get('adjudicator_id', 'N/A')}",
        f"- **Signed At:** {adj.get('signed_at', 'N/A')}",
        f"- **Examiner Notes:** {adj.get('notes', 'None')}",
        "",
        "---",
        "*Confidential InsurTech Audit Record — VeriSight AI Intelligence Engine*",
    ])
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Claims Dashboard & Audit Explorer
# ---------------------------------------------------------------------------
def render_dashboard():
    st.markdown("""
    <div class="main-header">
        <h1>Claims Intelligence & Audit Dashboard</h1>
        <p>Comprehensive repository of all evidence claims analyzed during this session.
           Filter, search, inspect component coverage, and review examiner certifications.</p>
    </div>""", unsafe_allow_html=True)

    render_stat_cards()

    reviews = st.session_state.recent_reviews
    if not reviews:
        st.info("No claims reviewed yet. Submit your first claim in the workspace or try one of the prototype scenarios!")
        return

    # ── Search & Filter Controls ──────────────────────────────────────────
    with st.container(border=True):
        st.markdown('<p class="section-header">🔍 Filter & Search Claims</p>', unsafe_allow_html=True)
        fcol1, fcol2, fcol3, fcol4 = st.columns([2, 1, 1, 1])

        with fcol1:
            search_query = st.text_input(
                "Search Claim ID / Description",
                value=st.session_state.dash_search,
                placeholder="e.g. CLM-1009, bumper, screen…",
                key="dash_input_search",
            )
            st.session_state.dash_search = search_query

        with fcol2:
            cat_choice = st.selectbox(
                "Object Category",
                ["All", "Car", "Laptop", "Package"],
                index=["All", "Car", "Laptop", "Package"].index(st.session_state.dash_category),
                key="dash_select_cat",
            )
            st.session_state.dash_category = cat_choice

        with fcol3:
            dec_choice = st.selectbox(
                "Verdict Status",
                ["All", "SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE", "ADJUDICATED"],
                index=["All", "SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE", "ADJUDICATED"].index(st.session_state.dash_decision),
                key="dash_select_dec",
            )
            st.session_state.dash_decision = dec_choice

        with fcol4:
            sort_choice = st.selectbox(
                "Sort Order",
                ["Newest First", "Oldest First", "Highest Confidence"],
                index=["Newest First", "Oldest First", "Highest Confidence"].index(st.session_state.dash_sort),
                key="dash_select_sort",
            )
            st.session_state.dash_sort = sort_choice

    # Filter logic
    filtered = []
    sq = search_query.strip().lower()
    for r in reviews:
        # Category filter
        if cat_choice != "All" and r.get("object_type") != cat_choice:
            continue
        # Decision filter
        if dec_choice == "ADJUDICATED":
            if not r.get("adjudication", {}).get("is_finalized"):
                continue
        elif dec_choice != "All" and r.get("decision") != dec_choice:
            continue
        # Search query
        if sq:
            cid = str(r.get("claim_id", "")).lower()
            desc = str(r.get("justification", "")).lower()
            dmg = str(r.get("damage_type", "")).lower()
            if sq not in cid and sq not in desc and sq not in dmg:
                continue
        filtered.append(r)

    # Sort logic
    if sort_choice == "Newest First":
        filtered_sorted = list(reversed(filtered))
    elif sort_choice == "Oldest First":
        filtered_sorted = list(filtered)
    elif sort_choice == "Highest Confidence":
        filtered_sorted = sorted(filtered, key=lambda x: x.get("confidence", 0), reverse=True)
    else:
        filtered_sorted = filtered

    st.markdown(f"**Showing {len(filtered_sorted)} of {len(reviews)} Total Claims**")

    if not filtered_sorted:
        st.warning("No claims match your filter criteria. Try broadening your search.")
        return

    # Render Claim Cards
    for rev in filtered_sorted:
        cid = rev.get("claim_id", "N/A")
        dec = rev.get("decision", "INSUFFICIENT_EVIDENCE")
        badge_cls, badge_text = DECISION_BADGE.get(dec, ("badge-insufficient", dec))
        sev = rev.get("severity", "UNKNOWN")
        conf = int(rev.get("confidence", 0) * 100)
        ts = rev.get("timestamp", "")[:19].replace("T", " ")
        adj = rev.get("adjudication", {})
        is_adj = adj.get("is_finalized", False)

        with st.container(border=True):
            r_top1, r_top2 = st.columns([3, 2])

            with r_top1:
                st.markdown(f"""
                <div style="display:flex; align-items:center; gap:8px;">
                    <span style="font-size:1.15rem; font-weight:800; color:#0F172A;">{cid}</span>
                    <span class="badge {badge_cls}">{badge_text}</span>
                    { '<span class="badge badge-adjudicated">⚖️ CERTIFIED</span>' if is_adj else '' }
                </div>
                <div style="font-size:0.82rem; color:#64748B; margin-top:4px;">
                    Category: <strong>{rev.get('object_type','')}</strong> &nbsp;·&nbsp;
                    Damage: <strong>{rev.get('damage_type','—')}</strong> &nbsp;·&nbsp;
                    {ts}
                </div>
                """, unsafe_allow_html=True)

            with r_top2:
                st.markdown(f"""
                <div style="text-align:right;">
                    <span class="sev-pill {SEV_CLASS.get(sev, 'sev-unknown')}">Severity: {sev}</span>
                    <span style="font-weight:700; color:#2563EB; font-size:0.88rem; margin-left:8px;">{conf}% Conf</span>
                    { f"<div style='font-size:0.82rem; color:#059669; font-weight:700; margin-top:4px;'>Approved: ${adj.get('payout_approved',0):,} USD</div>" if is_adj else "" }
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
            just = rev.get("justification", "")
            excerpt = just[:240] + ("…" if len(just) > 240 else "")
            st.markdown(f"<div style='font-size:0.86rem; color:#334155; line-height:1.5;'>{excerpt}</div>", unsafe_allow_html=True)

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            bcol1, bcol2 = st.columns([4, 1])
            with bcol2:
                if st.button(f"View Dossier →", key=f"view_{cid}_{rev.get('timestamp','')}", use_container_width=True):
                    st.session_state.current_result = rev
                    st.session_state.page = "results"
                    st.rerun()

# ---------------------------------------------------------------------------
# App Router
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
