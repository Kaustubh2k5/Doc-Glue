"""
Streamlit Inspection Dashboard for Doc-Glue Fact Knowledge Layer.
Connects to FastAPI backend (/upload, /facts, /reconciliations).
"""
import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Doc-Glue | Fact Knowledge Layer",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics and dark glassmorphic card design
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stTabs [data-baseweb="tab-list"] { gap: 12px; }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 20px;
        background-color: #1f2937;
        border-radius: 8px;
        color: #e5e7eb;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6 !important;
        color: #ffffff !important;
    }
    .badge-corroborated {
        background-color: #059669;
        color: #ffffff;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-contradicted {
        background-color: #dc2626;
        color: #ffffff;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-reconciled {
        background-color: #d97706;
        color: #ffffff;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .card-box {
        background-color: #111827;
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


def fetch_facts():
    try:
        res = requests.get(f"{API_URL}/facts", timeout=10)
        if res.status_code == 200:
            return res.json()
    except Exception as e:
        st.sidebar.error(f"Failed to connect to API ({e})")
    return []


def fetch_reconciliations():
    try:
        res = requests.get(f"{API_URL}/reconciliations", timeout=10)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return []


# Sidebar Controls
st.sidebar.title("🧩 Doc-Glue")
st.sidebar.caption("Fact Knowledge Layer & Reconciliation Engine")

st.sidebar.subheader("📄 Upload Document")
uploaded_file = st.sidebar.file_uploader("Upload PDF Document", type=["pdf"])

if uploaded_file is not None:
    if st.sidebar.button("Process Document", use_container_width=True):
        with st.spinner("Parsing PDF, extracting facts, and processing reconciliations..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                res = requests.post(f"{API_URL}/upload", files=files, timeout=60)
                if res.status_code == 200:
                    data = res.json()
                    st.sidebar.success(f"Extracted {data.get('facts_extracted')} facts from '{data.get('filename')}'!")
                    st.rerun()
                else:
                    st.sidebar.error(f"Upload failed: {res.text}")
            except Exception as e:
                st.sidebar.error(f"API Connection error: {e}")

st.sidebar.divider()
if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
    st.rerun()


# Main Dashboard
st.title("Fact Knowledge Layer & Reconciliation Inspector")
st.caption("Extracted numerical/semantic claims grounded in PDF evidence with cross-document reconciliation.")

facts_data = fetch_facts()
reconciliations_data = fetch_reconciliations()

col1, col2, col3 = st.columns(3)
col1.metric("Total Extracted Facts", len(facts_data))
col2.metric("Total Reconciliations", len(reconciliations_data))

corrob_count = sum(1 for r in reconciliations_data if r.get("relationship") == "CORROBORATED")
contra_count = sum(1 for r in reconciliations_data if r.get("relationship") == "CONTRADICTED")
recon_count = sum(1 for r in reconciliations_data if r.get("relationship") == "RECONCILED")
col3.metric("Relations breakdown", f"🟢 {corrob_count} | 🔴 {contra_count} | 🟡 {recon_count}")

tab1, tab2 = st.tabs(["📊 Fact Ledger", "🔍 Reconciliation Inspector"])

with tab1:
    st.subheader("Extracted Grounded Facts")
    if not facts_data:
        st.info("No facts extracted yet. Upload a PDF using the sidebar to begin.")
    else:
        # Filter input
        search_query = st.text_input("Filter facts by Subject or Property:", "")
        
        table_rows = []
        for f in facts_data:
            subj = f.get("subject", "")
            prop = f.get("property_name", "")
            if search_query.lower() in subj.lower() or search_query.lower() in prop.lower():
                ev = f.get("evidence", {})
                table_rows.append({
                    "Subject": subj,
                    "Property Name": prop,
                    "Value": f.get("value"),
                    "Unit": f.get("unit") or "-",
                    "Temporal Context": f.get("temporal_context") or "-",
                    "Scope Context": f.get("scope_context") or "-",
                    "Document": ev.get("filename", "-"),
                    "Page": ev.get("page_number", "-"),
                    "Verbatim Evidence": ev.get("verbatim_text", "")
                })
        
        st.dataframe(table_rows, use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Cross-Document Pairwise Reconciliations")
    if not reconciliations_data:
        st.info("No reconciliations generated yet. Ingest multiple document facts to generate pairwise reconciliations.")
    else:
        for idx, rec in enumerate(reconciliations_data, 1):
            rel = rec.get("relationship", "RECONCILED")
            reasoning = rec.get("reasoning", "")
            details = rec.get("resolution_details", {})

            fact_a = rec.get("fact_a", {})
            fact_b = rec.get("fact_b", {})

            ev_a = fact_a.get("evidence", {})
            ev_b = fact_b.get("evidence", {})

            # Badge styling
            if rel == "CORROBORATED":
                badge_html = '<span class="badge-corroborated">🟢 CORROBORATED</span>'
            elif rel == "CONTRADICTED":
                badge_html = '<span class="badge-contradicted">🔴 CONTRADICTED</span>'
            else:
                badge_html = '<span class="badge-reconciled">🟡 RECONCILED</span>'

            with st.container():
                st.markdown(f"""
                <div class="card-box">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                        <h4 style="margin: 0; color: #f3f4f6;">Pairwise Comparison #{idx}</h4>
                        {badge_html}
                    </div>
                """, unsafe_allow_html=True)

                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**Fact A**")
                    st.write(f"**Subject:** {fact_a.get('subject')}")
                    st.write(f"**Property:** {fact_a.get('property_name')}")
                    st.write(f"**Value:** `{fact_a.get('value')} {fact_a.get('unit') or ''}`")
                    st.write(f"**Context:** {fact_a.get('temporal_context') or 'N/A'} | {fact_a.get('scope_context') or 'N/A'}")
                    st.caption(f"Source: Page {ev_a.get('page_number')} in {ev_a.get('filename')}")

                with col_b:
                    st.markdown("**Fact B**")
                    st.write(f"**Subject:** {fact_b.get('subject')}")
                    st.write(f"**Property:** {fact_b.get('property_name')}")
                    st.write(f"**Value:** `{fact_b.get('value')} {fact_b.get('unit') or ''}`")
                    st.write(f"**Context:** {fact_b.get('temporal_context') or 'N/A'} | {fact_b.get('scope_context') or 'N/A'}")
                    st.caption(f"Source: Page {ev_b.get('page_number')} in {ev_b.get('filename')}")

                with st.expander("🔍 View LLM Reasoning & Verbatim Evidence"):
                    st.write("**LLM Reasoning Log:**")
                    st.info(reasoning)

                    if details:
                        st.write("**Resolution Details:**")
                        st.json(details)

                    col_ev_a, col_ev_b = st.columns(2)
                    with col_ev_a:
                        st.caption(f"Verbatim Evidence (Fact A - Page {ev_a.get('page_number')}):")
                        st.code(ev_a.get("verbatim_text") or "N/A", language="text")
                    with col_ev_b:
                        st.caption(f"Verbatim Evidence (Fact B - Page {ev_b.get('page_number')}):")
                        st.code(ev_b.get("verbatim_text") or "N/A", language="text")

                st.markdown("</div>", unsafe_allow_html=True)
