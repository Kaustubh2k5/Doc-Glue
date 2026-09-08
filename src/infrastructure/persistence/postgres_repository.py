"""
PostgreSQL + pgvector Persistence Repository implementation.
Provides PostgresFactRepository and PostgresReconciliationRepository.
Handles automatic schema migration, SHA-256 document deduplication, connection pooling, and HNSW vector search.
"""
import os
import json
import uuid
import logging
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from pgvector.psycopg import register_vector

from src.domain.interfaces import IFactRepository, IReconciliationRepository
from src.domain.models import Fact, SourceEvidence, FactComparison, RelationType

load_dotenv()
logger = logging.getLogger(__name__)

CREATE_TABLES_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY,
    filename TEXT NOT NULL,
    file_hash TEXT UNIQUE,
    uploaded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Add file_hash column if missing in existing installations
ALTER TABLE documents ADD COLUMN IF NOT EXISTS file_hash TEXT UNIQUE;

CREATE TABLE IF NOT EXISTS facts (
    fact_id TEXT PRIMARY KEY,
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    subject TEXT NOT NULL,
    property_name TEXT NOT NULL,
    value JSONB,
    unit TEXT,
    temporal_context TEXT,
    scope_context TEXT,
    verbatim_text TEXT,
    page_number INT,
    vector_embedding vector(1536)
);

CREATE TABLE IF NOT EXISTS reconciliations (
    id UUID PRIMARY KEY,
    fact_a_id TEXT REFERENCES facts(fact_id) ON DELETE CASCADE,
    fact_b_id TEXT REFERENCES facts(fact_id) ON DELETE CASCADE,
    relationship TEXT NOT NULL,
    reasoning TEXT NOT NULL,
    resolution_details JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(subject);
CREATE INDEX IF NOT EXISTS idx_facts_property_name ON facts(property_name);
CREATE INDEX IF NOT EXISTS idx_facts_vector_hnsw ON facts USING hnsw (vector_embedding vector_cosine_ops);
"""


def get_db_connection_string() -> str:
    user = os.getenv("POSTGRES_USER", "docglue")
    password = os.getenv("POSTGRES_PASSWORD", "docglue_secret")
    db = os.getenv("POSTGRES_DB", "docglue_db")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5433")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


class PostgresFactRepository(IFactRepository):
    """PostgreSQL + pgvector repository for Fact entity persistence and similarity search."""

    def __init__(self, conninfo: Optional[str] = None):
        self.conninfo = conninfo or get_db_connection_string()
        self.pool = ConnectionPool(self.conninfo, open=True, min_size=1, max_size=10)
        self._init_db()

    def _init_db(self):
        """Runs auto-migrations on startup."""
        with self.pool.connection() as conn:
            conn.execute(CREATE_TABLES_SQL)
            conn.commit()

    def find_document_by_hash(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """Queries whether a PDF file hash has already been ingested."""
        with self.pool.connection() as conn:
            conn.row_factory = dict_row
            row = conn.execute(
                "SELECT id, filename, uploaded_at FROM documents WHERE file_hash = %s;",
                (file_hash,)
            ).fetchone()
            return dict(row) if row else None

    def save_document_hash(self, doc_id: str, filename: str, file_hash: str) -> None:
        try:
            doc_uuid = uuid.UUID(doc_id.replace("doc-", "").zfill(32))
        except Exception:
            doc_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, doc_id)

        with self.pool.connection() as conn:
            conn.execute(
                """
                INSERT INTO documents (id, filename, file_hash, uploaded_at)
                VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (file_hash) DO NOTHING;
                """,
                (doc_uuid, filename, file_hash)
            )
            conn.commit()

    def save_fact(self, fact: Fact, embedding: Optional[List[float]] = None) -> None:
        with self.pool.connection() as conn:
            register_vector(conn)

            doc_uuid = None
            if fact.evidence.document_id:
                try:
                    doc_uuid = uuid.UUID(fact.evidence.document_id.replace("doc-", "").zfill(32))
                except Exception:
                    doc_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, fact.evidence.document_id)

                conn.execute(
                    """
                    INSERT INTO documents (id, filename, uploaded_at)
                    VALUES (%s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (id) DO NOTHING;
                    """,
                    (doc_uuid, fact.evidence.filename)
                )

            val_json = json.dumps(fact.value)

            conn.execute(
                """
                INSERT INTO facts (
                    fact_id, document_id, subject, property_name, value,
                    unit, temporal_context, scope_context, verbatim_text,
                    page_number, vector_embedding
                ) VALUES (
                    %s, %s, %s, %s, %s::jsonb,
                    %s, %s, %s, %s,
                    %s, %s
                )
                ON CONFLICT (fact_id) DO UPDATE SET
                    subject = EXCLUDED.subject,
                    property_name = EXCLUDED.property_name,
                    value = EXCLUDED.value,
                    unit = EXCLUDED.unit,
                    temporal_context = EXCLUDED.temporal_context,
                    scope_context = EXCLUDED.scope_context,
                    verbatim_text = EXCLUDED.verbatim_text,
                    page_number = EXCLUDED.page_number,
                    vector_embedding = EXCLUDED.vector_embedding;
                """,
                (
                    fact.fact_id,
                    doc_uuid,
                    fact.subject,
                    fact.property_name,
                    val_json,
                    fact.unit,
                    fact.temporal_context,
                    fact.scope_context,
                    fact.evidence.verbatim_text,
                    fact.evidence.page_number,
                    embedding
                )
            )
            conn.commit()

    def get_all_facts(self) -> List[Fact]:
        with self.pool.connection() as conn:
            conn.row_factory = dict_row
            rows = conn.execute(
                """
                SELECT f.fact_id, f.document_id, f.subject, f.property_name, f.value,
                       f.unit, f.temporal_context, f.scope_context, f.verbatim_text,
                       f.page_number, d.filename
                FROM facts f
                LEFT JOIN documents d ON f.document_id = d.id;
                """
            ).fetchall()

            facts = []
            for row in rows:
                evidence = SourceEvidence(
                    document_id=str(row["document_id"]) if row["document_id"] else "doc-unknown",
                    filename=row["filename"] or "unknown.pdf",
                    page_number=row["page_number"] or 1,
                    verbatim_text=row["verbatim_text"] or ""
                )
                val = row["value"]
                if isinstance(val, str):
                    try:
                        val = json.loads(val)
                    except Exception:
                        pass

                fact = Fact(
                    fact_id=row["fact_id"],
                    subject=row["subject"],
                    property_name=row["property_name"],
                    value=val,
                    unit=row["unit"],
                    temporal_context=row["temporal_context"],
                    scope_context=row["scope_context"],
                    evidence=evidence
                )
                facts.append(fact)

            return facts

    def find_candidates(self, query_vector: List[float], top_k: int = 5) -> List[Fact]:
        with self.pool.connection() as conn:
            register_vector(conn)
            conn.row_factory = dict_row
            
            rows = conn.execute(
                """
                SELECT f.fact_id, f.document_id, f.subject, f.property_name, f.value,
                       f.unit, f.temporal_context, f.scope_context, f.verbatim_text,
                       f.page_number, d.filename,
                       (f.vector_embedding <=> %s::vector) AS distance
                FROM facts f
                LEFT JOIN documents d ON f.document_id = d.id
                WHERE f.vector_embedding IS NOT NULL
                ORDER BY distance ASC
                LIMIT %s;
                """,
                (query_vector, top_k)
            ).fetchall()

            facts = []
            for row in rows:
                evidence = SourceEvidence(
                    document_id=str(row["document_id"]) if row["document_id"] else "doc-unknown",
                    filename=row["filename"] or "unknown.pdf",
                    page_number=row["page_number"] or 1,
                    verbatim_text=row["verbatim_text"] or ""
                )
                val = row["value"]
                if isinstance(val, str):
                    try:
                        val = json.loads(val)
                    except Exception:
                        pass

                fact = Fact(
                    fact_id=row["fact_id"],
                    subject=row["subject"],
                    property_name=row["property_name"],
                    value=val,
                    unit=row["unit"],
                    temporal_context=row["temporal_context"],
                    scope_context=row["scope_context"],
                    evidence=evidence
                )
                facts.append(fact)

            return facts

    def clear_all_data(self) -> None:
        """Truncates all documents, facts, and reconciliations for clean state."""
        with self.pool.connection() as conn:
            conn.execute("TRUNCATE TABLE reconciliations, facts, documents CASCADE;")
            conn.commit()

    def close(self):
        self.pool.close()


class PostgresReconciliationRepository(IReconciliationRepository):
    """PostgreSQL repository for FactComparison reconciliation records."""

    def __init__(self, conninfo: Optional[str] = None):
        self.conninfo = conninfo or get_db_connection_string()
        self.pool = ConnectionPool(self.conninfo, open=True, min_size=1, max_size=10)

    def save_comparison(self, comparison: FactComparison) -> None:
        with self.pool.connection() as conn:
            rec_id = uuid.uuid4()
            rel_str = comparison.relationship.value if hasattr(comparison.relationship, "value") else str(comparison.relationship)
            res_json = json.dumps(comparison.resolution_details or {})

            conn.execute(
                """
                INSERT INTO reconciliations (id, fact_a_id, fact_b_id, relationship, reasoning, resolution_details, created_at)
                VALUES (%s, %s, %s, %s, %s, %s::jsonb, CURRENT_TIMESTAMP);
                """,
                (
                    rec_id,
                    comparison.fact_a.fact_id,
                    comparison.fact_b.fact_id,
                    rel_str,
                    comparison.reasoning,
                    res_json
                )
            )
            conn.commit()

    def get_all_comparisons(self) -> List[FactComparison]:
        with self.pool.connection() as conn:
            conn.row_factory = dict_row
            rows = conn.execute(
                """
                SELECT r.id, r.fact_a_id, r.fact_b_id, r.relationship, r.reasoning, r.resolution_details, r.created_at,
                       fa.subject as a_subj, fa.property_name as a_prop, fa.value as a_val, fa.unit as a_unit, fa.temporal_context as a_temp, fa.scope_context as a_scope, fa.verbatim_text as a_verb, fa.page_number as a_page, da.filename as a_file,
                       fb.subject as b_subj, fb.property_name as b_prop, fb.value as b_val, fb.unit as b_unit, fb.temporal_context as b_temp, fb.scope_context as b_scope, fb.verbatim_text as b_verb, fb.page_number as b_page, db.filename as b_file
                FROM reconciliations r
                LEFT JOIN facts fa ON r.fact_a_id = fa.fact_id
                LEFT JOIN documents da ON fa.document_id = da.id
                LEFT JOIN facts fb ON r.fact_b_id = fb.fact_id
                LEFT JOIN documents db ON fb.document_id = db.id
                ORDER BY r.created_at DESC;
                """
            ).fetchall()

            comparisons = []
            for row in rows:
                res_det = row["resolution_details"]
                if isinstance(res_det, str):
                    try:
                        res_det = json.loads(res_det)
                    except Exception:
                        pass

                ev_a = SourceEvidence(
                    document_id="doc-a",
                    filename=row["a_file"] or "doc_a.pdf",
                    page_number=row["a_page"] or 1,
                    verbatim_text=row["a_verb"] or ""
                )
                val_a = row["a_val"]
                if isinstance(val_a, str):
                    try: val_a = json.loads(val_a)
                    except Exception: pass

                fact_a = Fact(
                    fact_id=row["fact_a_id"],
                    subject=row["a_subj"] or "Unknown",
                    property_name=row["a_prop"] or "Metric",
                    value=val_a,
                    unit=row["a_unit"],
                    temporal_context=row["a_temp"],
                    scope_context=row["a_scope"],
                    evidence=ev_a
                )

                ev_b = SourceEvidence(
                    document_id="doc-b",
                    filename=row["b_file"] or "doc_b.pdf",
                    page_number=row["b_page"] or 1,
                    verbatim_text=row["b_verb"] or ""
                )
                val_b = row["b_val"]
                if isinstance(val_b, str):
                    try: val_b = json.loads(val_b)
                    except Exception: pass

                fact_b = Fact(
                    fact_id=row["fact_b_id"],
                    subject=row["b_subj"] or "Unknown",
                    property_name=row["b_prop"] or "Metric",
                    value=val_b,
                    unit=row["b_unit"],
                    temporal_context=row["b_temp"],
                    scope_context=row["b_scope"],
                    evidence=ev_b
                )

                comp = FactComparison(
                    fact_a=fact_a,
                    fact_b=fact_b,
                    relationship=RelationType(row["relationship"]),
                    reasoning=row["reasoning"],
                    resolution_details=res_det or {}
                )
                comparisons.append(comp)

            return comparisons

    def clear_all_data(self) -> None:
        with self.pool.connection() as conn:
            conn.execute("TRUNCATE TABLE reconciliations, facts, documents CASCADE;")
            conn.commit()

    def close(self):
        self.pool.close()
