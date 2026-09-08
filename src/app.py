"""
Streamlit Unified Reference-Style Merged Fact Ledger Dashboard for Doc-Glue.
Connects to FastAPI backend (/upload, /facts, /reconciliations, /clear, /health).
"""
import os
import requests
import streamlit as st
from typing import Dict, List, Any

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Doc-Glue | Merged Fact Ledger",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern glassmorphic reference cards
st.markdown("""
<style>
    .main { background-color: #0b0f19; color: #f3f4f6; }
    .stTextInput > div > div > input { background-color: #111827; color: #ffffff; border: 1px solid #374151; }
    
    .fact-card {
        background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 24px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    .fact-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #60a5fa;
        margin-bottom: 8px;
    }
    .meta-pill {
        display: inline-block;
        background-color: #374151;
        color: #d1d5db;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 500;
        margin-right: 8px;
        margin-bottom: 8px;
    }
    .rel-section {
        margin-top: 16px;
        padding-top: 16px;
        border-top: 1px dashed #374151;
    }
    .badge-corrob {
        background-color: #065f46;
        color: #34d399;
        border: 1px solid #059669;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .badge-recon {
        background-color: #78350f;
        color: #fbbf24;
        border: 1px solid #d97706;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .badge-contra {
        background-color: #7f1d1d;
        color: #fca5a5;
        border: 1px solid #dc2626;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .rel-box-corrob {
        background-color: #064e3b22;
        border-left: 4px solid #10b981;
        padding: 10px 14px;
        border-radius: 6px;
        margin-top: 8px;
    }
    .rel-box-recon {
        background-color: #78350f22;
        border-left: 4px solid #f59e0b;
        padding: 10px 14px;
        border-radius: 6px;
        margin-top: 8px;
    }
    .rel-box-contra {
        background-color: #7f1d1d22;
        border-left: 4px solid #ef4444;
        padding: 10px 14px;
        border-radius: 6px;
        margin-top: 8px;
    }
</style>
""", unsafe_allow_html=True)


def fetch_health():
    try:
        res = requests.get(f"{API_URL}/health", timeout=5)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return {}


def fetch_facts():
    try:
        res = requests.get(f"{API_URL}/facts", timeout=10)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return []


def fetch_reconciliations():
    try:
        res = requests.get(f"{API_URL}/reconciliations", timeout=10)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return []


# Sidebar
st.sidebar.title("🧩 Doc-Glue")
st.sidebar.caption("Unified Reference-Style Fact Ledger")

health_info = fetch_health()
active_model = health_info.get("active_model", "meta-llama/llama-3.3-70b-instruct:free")
st.sidebar.info(f"🤖 **Active LLM Engine**:\n`{active_model}`")

st.sidebar.subheader("📄 Ingest Document")
uploaded_file = st.sidebar.file_uploader("Upload PDF Document", type=["pdf"])

if uploaded_file is not None:
    if st.sidebar.button("Process & Merge into Ledger", use_container_width=True):
        with st.spinner("Extracting facts & building unified knowledge relations..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                res = requests.post(f"{API_URL}/upload", files=files, timeout=60)
                if res.status_code == 200:
                    data = res.json()
                    if data.get("cached"):
                        st.sidebar.info(f"⚡ Instant Cache Hit: '{data.get('filename')}' retrieved from database!")
                    else:
                        st.sidebar.success(f"Extracted {data.get('facts_extracted')} facts from '{data.get('filename')}'!")
                    st.rerun()
                else:
                    st.sidebar.error(f"Upload failed: {res.text}")
            except Exception as e:
                st.sidebar.error(f"API Connection error: {e}")

st.sidebar.divider()
col_sb1, col_sb2 = st.sidebar.columns(2)
with col_sb1:
    if st.button("🔄 Refresh", use_container_width=True):
        st.rerun()
with col_sb2:
    if st.button("🗑️ Clear DB", use_container_width=True):
        try:
            res = requests.delete(f"{API_URL}/clear", timeout=10)
            if res.status_code == 200:
                st.sidebar.success("Database cleared!")
                st.rerun()
        except Exception as e:
            st.sidebar.error(f"Clear failed: {e}")


# Fetch data
facts = fetch_facts()
reconciliations = fetch_reconciliations()

# Header Metrics
st.title("Unified Reference Fact Ledger")
st.caption("Centralized reference ledger linking grounded claims with corroborating evidence, contextual reconciliations, and direct contradictions.")

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("Unique Facts Stored", len(facts))

corroborated_pairs = [r for r in reconciliations if r.get("relationship") == "CORROBORATED"]
reconciled_pairs = [r for r in reconciliations if r.get("relationship") == "RECONCILED"]
contradicted_pairs = [r for r in reconciliations if r.get("relationship") == "CONTRADICTED"]

col_m2.metric("Supported Claims (🟢)", len(corroborated_pairs))
col_m3.metric("Contextual Reconciliations (🟡)", len(reconciled_pairs))
col_m4.metric("Direct Contradictions (🔴)", len(contradicted_pairs))

st.divider()

if not facts:
    st.info("💡 The Merged Fact Ledger is currently empty. Upload a PDF using the sidebar to ingest facts and build knowledge relations.")
else:
    # Build indexing map for pairwise relations per fact_id
    # Index: fact_id -> { "corroborated": [ (other_fact, reasoning, details) ], "reconciled": [...], "contradicted": [...] }
    relations_map: Dict[str, Dict[str, List[Any]]] = {}

    for f in facts:
        relations_map[f["fact_id"]] = {"corroborated": [], "reconciled": [], "contradicted": []}

    for r in reconciliations:
        rel = r.get("relationship", "RECONCILED")
        fact_a = r.get("fact_a", {})
        fact_b = r.get("fact_b", {})

        id_a = fact_a.get("fact_id")
        id_b = fact_b.get("fact_id")

        reasoning = r.get("reasoning", "")
        details = r.get("resolution_details", {})

        rel_key = rel.lower()
        if rel_key not in ["corroborated", "reconciled", "contradicted"]:
            rel_key = "reconciled"

        if id_a in relations_map:
            relations_map[id_a][rel_key].append((fact_b, reasoning, details))
        if id_b in relations_map:
            relations_map[id_b][rel_key].append((fact_a, reasoning, details))

    # Controls & Filter Bar
    filter_col1, filter_col2 = st.columns([3, 1])
    with filter_col1:
        search_query = st.text_input("🔍 Search Fact Ledger (by Subject, Metric, or Document):", "")
    with filter_col2:
        relation_filter = st.selectbox(
            "Filter by Relation Status:",
            ["All Claims", "Only Supported (🟢)", "Only Reconciled (🟡)", "Only Contradicted (🔴)"]
        )

    # Render Unified Reference Cards
    displayed_count = 0

    for fact in facts:
        subj = fact.get("subject", "Unknown Entity")
        prop = fact.get("property_name", "Metric")
        val = fact.get("value")
        unit = fact.get("unit") or ""
        temp_ctx = fact.get("temporal_context") or "N/A"
        scope_ctx = fact.get("scope_context") or "N/A"

        ev = fact.get("evidence", {})
        doc_name = ev.get("filename", "document.pdf")
        page_no = ev.get("page_number", 1)
        verbatim = ev.get("verbatim_text", "")

        f_id = fact.get("fact_id")
        rels = relations_map.get(f_id, {"corroborated": [], "reconciled": [], "contradicted": []})

        c_list = rels["corroborated"]
        r_list = rels["reconciled"]
        k_list = rels["contradicted"]

        # Apply relation filter
        if relation_filter == "Only Supported (🟢)" and not c_list:
            continue
        if relation_filter == "Only Reconciled (🟡)" and not r_list:
            continue
        if relation_filter == "Only Contradicted (🔴)" and not k_list:
            continue

        # Apply search query filter
        search_target = f"{subj} {prop} {doc_name}".lower()
        if search_query and search_query.lower() not in search_target:
            continue

        displayed_count += 1

        # Card Rendering
        with st.container():
            st.markdown(f"""
            <div class="fact-card">
                <div class="fact-title">📌 {subj} &nbsp;·&nbsp; {prop} = <span style="color: #10b981;">{val} {unit}</span></div>
                <div>
                    <span class="meta-pill">📅 Period: {temp_ctx}</span>
                    <span class="meta-pill">🎯 Scope: {scope_ctx}</span>
                    <span class="meta-pill">📄 Source: Page {page_no} in {doc_name}</span>
                </div>
                <div style="margin-top: 10px; font-style: italic; color: #9ca3af; background: #00000033; padding: 10px; border-radius: 6px; border-left: 3px solid #60a5fa;">
                    "{verbatim[:250]}"
                </div>
            """, unsafe_allow_html=True)

            # Unified Knowledge Relations Section
            if c_list or r_list or k_list:
                st.markdown('<div class="rel-section">', unsafe_allow_html=True)
                st.markdown("##### 🔗 Unified Knowledge Relations & Reasoning")

                # 🟢 Supported / Corroborated By
                if c_list:
                    st.markdown("**🟢 Supported / Corroborated By:**")
                    for target_fact, reason, det in c_list:
                        t_ev = target_fact.get("evidence", {})
                        st.markdown(f"""
                        <div class="rel-box-corrob">
                            <div><span class="badge-corrob">CORROBORATED</span> <strong>{target_fact.get('subject')} | {target_fact.get('property_name')} = {target_fact.get('value')} {target_fact.get('unit') or ''}</strong></div>
                            <div style="font-size: 0.85rem; color: #9ca3af; margin-top: 4px;">Source: Page {t_ev.get('page_number')} in <em>{t_ev.get('filename')}</em></div>
                            <div style="font-size: 0.88rem; color: #d1d5db; margin-top: 4px;">💡 <em>Reasoning:</em> {reason}</div>
                        </div>
                        """, unsafe_allow_html=True)

                # 🟡 Reconciled With
                if r_list:
                    st.markdown("**🟡 Reconciled With (Contextual Parameter Variant):**")
                    for target_fact, reason, det in r_list:
                        t_ev = target_fact.get("evidence", {})
                        explaining_factor = det.get("explaining_factor") if isinstance(det, dict) else det
                        st.markdown(f"""
                        <div class="rel-box-recon">
                            <div><span class="badge-recon">RECONCILED</span> <strong>{target_fact.get('subject')} | {target_fact.get('property_name')} = {target_fact.get('value')} {target_fact.get('unit') or ''}</strong> (Scope: {target_fact.get('scope_context') or 'N/A'}, Period: {target_fact.get('temporal_context') or 'N/A'})</div>
                            <div style="font-size: 0.85rem; color: #9ca3af; margin-top: 4px;">Source: Page {t_ev.get('page_number')} in <em>{t_ev.get('filename')}</em></div>
                            <div style="font-size: 0.88rem; color: #fbbf24; margin-top: 4px;">💡 <em>Contextual Explanation:</em> {reason}</div>
                        </div>
                        """, unsafe_allow_html=True)

                # 🔴 Contradicted By
                if k_list:
                    st.markdown("**🔴 Contradicted By (Direct Conflict Under Identical Scope):**")
                    for target_fact, reason, det in k_list:
                        t_ev = target_fact.get("evidence", {})
                        st.markdown(f"""
                        <div class="rel-box-contra">
                            <div><span class="badge-contra">CONTRADICTION</span> <strong>{target_fact.get('subject')} | {target_fact.get('property_name')} = {target_fact.get('value')} {target_fact.get('unit') or ''}</strong></div>
                            <div style="font-size: 0.85rem; color: #9ca3af; margin-top: 4px;">Source: Page {t_ev.get('page_number')} in <em>{t_ev.get('filename')}</em></div>
                            <div style="font-size: 0.88rem; color: #fca5a5; margin-top: 4px;">⚠️ <em>Conflict Note:</em> {reason}</div>
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

    if displayed_count == 0:
        st.warning("No facts matched your search/filter criteria.")
