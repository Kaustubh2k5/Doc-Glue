"""
Streamlit Unified Reference-Style Merged Fact Ledger Dashboard for Doc-Glue.
Connects to FastAPI backend (/upload, /facts, /reconciliations, /clear, /health, /metrics).
"""
import os
import requests
import streamlit as st
import base64
from typing import Dict, List, Any

API_URL = os.getenv("API_URL", "http://localhost:8000")

# Setup Page
st.set_page_config(
    page_title="Doc-Glue Fact Ledger",
    page_icon="Assets/Frame 2608495 (2).svg",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for compact modern glassmorphic reference cards
st.markdown("""
<style>
    .main { background-color: #0b0f19; color: #f3f4f6; }
    .stTextInput > div > div > input { background-color: #111827; color: #ffffff; border: 1px solid #374151; }
    
    .fact-card {
        background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
        border: 1px solid #374151;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 16px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
    }
    .fact-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #60a5fa;
        margin-bottom: 6px;
    }
    .meta-pill {
        display: inline-block;
        background-color: #374151;
        color: #d1d5db;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 500;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    .rel-section {
        margin-top: 12px;
        padding-top: 12px;
        border-top: 1px dashed #374151;
    }
    .badge-corrob {
        background-color: #065f46;
        color: #34d399;
        border: 1px solid #059669;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: 600;
    }
    .badge-recon {
        background-color: #78350f;
        color: #fbbf24;
        border: 1px solid #d97706;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: 600;
    }
    .badge-contra {
        background-color: #7f1d1d;
        color: #fca5a5;
        border: 1px solid #dc2626;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: 600;
    }
    .rel-box {
        background-color: #1f2937;
        border: 1px solid #4b5563;
        padding: 8px 12px;
        border-radius: 6px;
        margin-top: 6px;
    }
    .rel-box-corrob { border-left: 4px solid #10b981; }
    .rel-box-recon { border-left: 4px solid #f59e0b; }
    .rel-box-contra { border-left: 4px solid #ef4444; }
    
    .context-box {
        font-size: 0.8rem;
        color: #9ca3af;
        background: #00000044;
        padding: 6px;
        border-radius: 4px;
        border-left: 2px solid #6b7280;
        margin-top: 4px;
        font-style: italic;
    }
    .gist-text {
        font-size: 0.85rem;
        font-weight: 500;
        margin-top: 4px;
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

def fetch_metrics():
    try:
        res = requests.get(f"{API_URL}/metrics", timeout=5)
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
try:
    with open("Assets/Frame 2608495 (2).svg", "r") as f:
        svg_content = f.read()
    
    # Remove the hardcoded 22x22 pixel size inside the SVG to allow CSS scaling
    svg_content = svg_content.replace('width="22"', 'width="100%"').replace('height="22"', 'height="100%"')
    
    b64 = base64.b64encode(svg_content.encode("utf-8")).decode("utf-8")
    st.sidebar.markdown(
        f'<div style="display: flex; justify-content: center; margin-bottom: 20px;">'
        f'<img src="data:image/svg+xml;base64,{b64}" style="width: 300px; height: auto; max-width: 100%;" />'
        f'</div>',
        unsafe_allow_html=True
    )
except Exception:
    st.sidebar.title("Doc-Glue")

st.sidebar.caption("Unified Reference-Style Fact Ledger")

health_info = fetch_health()
active_model = health_info.get("active_model", "meta-llama/llama-3.3-70b-instruct:free")
st.sidebar.info(f"**Active LLM Engine**:\n`{active_model}`")

st.sidebar.subheader("Ingest Documents")
uploaded_files = st.sidebar.file_uploader("Upload PDF Documents", type=["pdf"], label_visibility="collapsed", accept_multiple_files=True)

if uploaded_files:
    if st.sidebar.button("Process & Merge", use_container_width=True):
        status_text = st.sidebar.empty()
        progress_bar = st.sidebar.progress(0)
        total_files = len(uploaded_files)
        
        for i, uploaded_file in enumerate(uploaded_files):
            file_base = float(i) / total_files
            file_scale = 1.0 / total_files
            
            # Stage 1: Parsing
            status_text.markdown(f"**Stage 1/3: PyMuPDF Parsing**  \n`{uploaded_file.name}` ({i+1}/{total_files})")
            progress_bar.progress(int((file_base + file_scale * 0.25) * 100))
            
            # Stage 2: Parallel Extraction
            status_text.markdown(f"**Stage 2/3: Parallel LLM Fact Extraction**  \n`{uploaded_file.name}` ({i+1}/{total_files})")
            progress_bar.progress(int((file_base + file_scale * 0.50) * 100))
            
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                res = requests.post(f"{API_URL}/upload", files=files, timeout=600)
                
                # Stage 3: Vector Indexing & Storage
                status_text.markdown(f"**Stage 3/3: Vector Indexing & Reconciling**  \n`{uploaded_file.name}` ({i+1}/{total_files})")
                progress_bar.progress(int((file_base + file_scale * 1.0) * 100))
                
                if res.status_code == 200:
                    data = res.json()
                    if data.get("cached"):
                        st.sidebar.info(f"Cache Hit: '{data.get('filename')}' retrieved from database.")
                    else:
                        st.sidebar.success(f"Extracted {data.get('facts_extracted')} facts from '{data.get('filename')}'.")
                else:
                    st.sidebar.error(f"Upload failed for {uploaded_file.name}: {res.text}")
            except Exception as e:
                st.sidebar.error(f"API Connection error on {uploaded_file.name}: {e}")
        
        status_text.markdown("**Ingestion & Fact Reconciliation Complete!**")
        st.rerun()

st.sidebar.divider()
col_sb1, col_sb2 = st.sidebar.columns(2)
with col_sb1:
    if st.button("Refresh", use_container_width=True):
        st.rerun()
with col_sb2:
    if st.button("Clear DB", use_container_width=True):
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
metrics_data = fetch_metrics()

# Tabs
tab_ledger, tab_metrics = st.tabs(["Fact Ledger", "Pipeline Metrics"])

with tab_metrics:
    st.subheader("Ingestion & Clustering Performance")
    
    if metrics_data and "ingestion" in metrics_data:
        ing = metrics_data["ingestion"]
        clus = metrics_data.get("clustering", {})
        
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        mcol1.metric("Docs Processed", ing.get("documents_processed", 0))
        mcol2.metric("Chunk Retention", f"{ing.get('chunk_keep_rate_pct', 0)}%")
        mcol3.metric("Fact Yield / Chunk", ing.get("fact_yield_per_chunk", 0))
        mcol4.metric("Fact Quality Rate", f"{ing.get('fact_quality_rate_pct', 0)}%")
        
        st.divider()
        
        mcol5, mcol6, mcol7, mcol8 = st.columns(4)
        mcol5.metric("Total Clusters", clus.get("num_clusters", 0))
        mcol6.metric("Singletons", clus.get("num_singletons", 0))
        mcol7.metric("Avg Cluster Size", clus.get("avg_cluster_size", 0))
        mcol8.metric("Struct. Rejection Rate", f"{clus.get('structural_filter_rejection_rate_pct', 0)}%")
        
        st.markdown("#### Document Breakdown")
        for doc in ing.get("per_document", []):
            st.caption(f"**{doc.get('filename')}** - Pages: {doc.get('pages')} | Chunks: {doc.get('chunks')} | Facts: {doc.get('facts')}")
    else:
        st.info("No metrics available. Process a document to generate metrics.")

with tab_ledger:
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Unique Facts", len(facts))

    corroborated_pairs = [r for r in reconciliations if r.get("relationship") == "CORROBORATED"]
    reconciled_pairs = [r for r in reconciliations if r.get("relationship") == "RECONCILED"]
    contradicted_pairs = [r for r in reconciliations if r.get("relationship") == "CONTRADICTED"]

    col_m2.metric("Supported", len(corroborated_pairs))
    col_m3.metric("Reconciled", len(reconciled_pairs))
    col_m4.metric("Contradicted", len(contradicted_pairs))

    if not facts:
        st.info("The Merged Fact Ledger is currently empty. Upload a PDF to begin.")
    else:
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
                relations_map[id_a][rel_key].append((fact_b, reasoning, details, "a"))
            if id_b in relations_map:
                relations_map[id_b][rel_key].append((fact_a, reasoning, details, "b"))

        # Controls
        filter_col1, filter_col2 = st.columns([3, 1])
        with filter_col1:
            search_query = st.text_input("Search (Subject, Metric, Document):", "", label_visibility="collapsed", placeholder="Search Ledger...")
        with filter_col2:
            relation_filter = st.selectbox(
                "Filter",
                ["All", "Supported", "Reconciled", "Contradicted"],
                label_visibility="collapsed"
            )

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

            if relation_filter == "Supported" and not c_list: continue
            if relation_filter == "Reconciled" and not r_list: continue
            if relation_filter == "Contradicted" and not k_list: continue

            search_target = f"{subj} {prop} {doc_name}".lower()
            if search_query and search_query.lower() not in search_target:
                continue

            displayed_count += 1

            with st.container():
                st.markdown(f"""
                <div class="fact-card">
                    <div class="fact-title">{subj} &nbsp;·&nbsp; {prop} = <span style="color: #10b981;">{val} {unit}</span></div>
                    <div>
                        <span class="meta-pill">Period: {temp_ctx}</span>
                        <span class="meta-pill">Scope: {scope_ctx}</span>
                        <span class="meta-pill">Source: {doc_name} (Pg {page_no})</span>
                    </div>
                    <div class="context-box">"{verbatim[:150]}..."</div>
                """, unsafe_allow_html=True)

                if c_list or r_list or k_list:
                    st.markdown('<div class="rel-section">', unsafe_allow_html=True)

                    if c_list:
                        for target_fact, reason, det, side in c_list:
                            t_ev = target_fact.get("evidence", {})
                            gist = det.get("reasoning_gist", reason) if isinstance(det, dict) else reason
                            ctx_self = det.get(f"source_context_{'a' if side=='a' else 'b'}", "") if isinstance(det, dict) else ""
                            ctx_target = det.get(f"source_context_{'b' if side=='a' else 'a'}", "") if isinstance(det, dict) else ""
                            
                            st.markdown(f"""
                            <div class="rel-box rel-box-corrob">
                                <div><span class="badge-corrob">SUPPORTED BY</span> <span style="font-size: 0.85rem; font-weight: 600;">{target_fact.get('subject')} | {target_fact.get('property_name')} = {target_fact.get('value')} {target_fact.get('unit') or ''}</span></div>
                                <div class="gist-text" style="color: #34d399;">{gist}</div>
                                <div style="font-size: 0.75rem; color: #9ca3af; margin-top: 4px;">Ref: {t_ev.get('filename')} (Pg {t_ev.get('page_number')})</div>
                                {f'<div class="context-box">"{ctx_target}"</div>' if ctx_target else ''}
                            </div>
                            """, unsafe_allow_html=True)

                    if r_list:
                        for target_fact, reason, det, side in r_list:
                            t_ev = target_fact.get("evidence", {})
                            gist = det.get("reasoning_gist", reason) if isinstance(det, dict) else reason
                            ctx_target = det.get(f"source_context_{'b' if side=='a' else 'a'}", "") if isinstance(det, dict) else ""
                            
                            st.markdown(f"""
                            <div class="rel-box rel-box-recon">
                                <div><span class="badge-recon">RECONCILED WITH</span> <span style="font-size: 0.85rem; font-weight: 600;">{target_fact.get('subject')} | {target_fact.get('property_name')} = {target_fact.get('value')} {target_fact.get('unit') or ''}</span></div>
                                <div class="gist-text" style="color: #fbbf24;">{gist}</div>
                                <div style="font-size: 0.75rem; color: #9ca3af; margin-top: 4px;">Ref: {t_ev.get('filename')} (Pg {t_ev.get('page_number')})</div>
                                {f'<div class="context-box">"{ctx_target}"</div>' if ctx_target else ''}
                            </div>
                            """, unsafe_allow_html=True)

                    if k_list:
                        for target_fact, reason, det, side in k_list:
                            t_ev = target_fact.get("evidence", {})
                            gist = det.get("reasoning_gist", reason) if isinstance(det, dict) else reason
                            ctx_target = det.get(f"source_context_{'b' if side=='a' else 'a'}", "") if isinstance(det, dict) else ""
                            
                            st.markdown(f"""
                            <div class="rel-box rel-box-contra">
                                <div><span class="badge-contra">CONTRADICTED BY</span> <span style="font-size: 0.85rem; font-weight: 600;">{target_fact.get('subject')} | {target_fact.get('property_name')} = {target_fact.get('value')} {target_fact.get('unit') or ''}</span></div>
                                <div class="gist-text" style="color: #fca5a5;">{gist}</div>
                                <div style="font-size: 0.75rem; color: #9ca3af; margin-top: 4px;">Ref: {t_ev.get('filename')} (Pg {t_ev.get('page_number')})</div>
                                {f'<div class="context-box">"{ctx_target}"</div>' if ctx_target else ''}
                            </div>
                            """, unsafe_allow_html=True)

                    st.markdown('</div>', unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

        if displayed_count == 0:
            st.warning("No facts matched your search.")
