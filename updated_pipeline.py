"""
pipeline.py — Multimodal Hybrid GraphRAG for Danfoss Assembly Manuals (V2)
==========================================================================

Core module. Two phases:

Phase 1 — offline, run once on Colab GPU:
    index_all() → embeds text/tables/image-captions into Qdrant

Phase 2 — online, per query, importable from app.py / evaluate.py:
    answer_query(question) → dict with answer, sources, confidence, timing

Input file layout (all under /content/KG-RAG-System-V2/data/):
    text/<manual_id>_text.json          (three schema variants supported)
    tables/<manual_id>_tables.json      (optional per manual)
    captions/<manual_id>_captions.json  (images array with structured fields)
    images/<manual_id>/<image_id>.png   (referenced by captions; needed for UI)
    kg/mmkg.graphml                     (Inioluwa's KG)

Author : Gargi Deshmukh
Lab    : Smart Manufacturing Systems Laboratory, FSU
"""

from __future__ import annotations

# =============================================================================
# 0. IMPORTS
# =============================================================================
import io
import json
import os
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch
from FlagEmbedding import BGEM3FlagModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from qdrant_client import QdrantClient, models as qmodels

_HAS_CUDA = torch.cuda.is_available()

try:
    from google import genai
    from google.genai import types as genai_types
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False


# =============================================================================
# 1. CONFIG  --  edit here for paths / hyperparameters
# =============================================================================

BASE_DIR = Path(os.environ.get("GRAPHRAG_BASE_DIR", "/content/KG-RAG-System-V2"))

# Input directories
DATA_DIR      = BASE_DIR / "data"
TEXT_DIR      = DATA_DIR / "text"
TABLES_DIR    = DATA_DIR / "tables"
CAPTIONS_DIR  = DATA_DIR / "captions"
IMAGES_DIR    = DATA_DIR / "images"
KG_DIR        = DATA_DIR / "kg"

# Output directories
QDRANT_DIR    = BASE_DIR / "qdrant_db"
SQLITE_PATH   = BASE_DIR / "query_logs.db"

# Manuals — this is the source of truth for which files to load
MANUAL_IDS = [
    "Filter_drier_shell",
    "Danfoss_React_RA_click",
    "EZ_Clip_to_5400_Series",
]

# Models
BGE_M3_MODEL   = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
DENSE_DIM      = 1024
USE_FP16       = _HAS_CUDA   # fp16 only makes sense on GPU

# Gemini
GEMINI_MODEL   = "gemini-3.5-flash"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Qdrant collections
COLL_TEXT   = "text_chunks"
COLL_TABLES = "table_chunks"
COLL_IMAGES = "image_chunks"

# Retrieval hyperparameters
TEXT_TOP_K   = 15
TABLE_TOP_K  = 10
IMAGE_TOP_K  = 10
RERANK_KEEP  = 8      # after reranking, keep this many per query type
FUSION_TOP_K = 10     # final context passed to LLM
KG_HOPS      = 2
KG_TRIPLE_CAP = 25    # cap KG triples in context

# Confidence thresholds — tuned for bge-reranker-v2-m3 sigmoid scores
CONF_HIGH = 0.60
CONF_MED  = 0.30

QUERY_TYPES = ("image", "table", "procedure", "part_number", "out_of_scope")

OUT_OF_SCOPE_MSG = (
    "This question is outside the scope of the Danfoss assembly manuals "
    "currently loaded into the system. Please ask about the DCR filter drier, "
    "Danfoss React RA click, or EZ Clip to 5400 Series."
)


# =============================================================================
# 2. LAZY MODULE-LEVEL SINGLETONS
# =============================================================================

_bge_m3: Optional[BGEM3FlagModel] = None
_reranker_tokenizer = None
_reranker_model = None
_qdrant: Optional[QdrantClient] = None
_kg: Optional[nx.DiGraph] = None
_genai_client = None


def get_bge_m3() -> BGEM3FlagModel:
    global _bge_m3
    if _bge_m3 is None:
        print(f"[init] loading BGE-M3 (fp16={USE_FP16}) ...")
        _bge_m3 = BGEM3FlagModel(BGE_M3_MODEL, use_fp16=USE_FP16)
    return _bge_m3


def get_reranker():
    """Returns (tokenizer, model). Uses transformers directly to avoid
    FlagReranker's incompatibility with newer transformers versions
    (prepare_for_model was removed from slow tokenizers)."""
    global _reranker_tokenizer, _reranker_model
    if _reranker_model is None:
        print(f"[init] loading reranker ({RERANKER_MODEL}) ...")
        _reranker_tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL, use_fast=True)
        _reranker_model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL)
        _reranker_model.eval()
        if _HAS_CUDA:
            _reranker_model = _reranker_model.cuda()
            if USE_FP16:
                _reranker_model = _reranker_model.half()
    return _reranker_tokenizer, _reranker_model


def get_qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        QDRANT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[init] opening Qdrant at {QDRANT_DIR}")
        _qdrant = QdrantClient(path=str(QDRANT_DIR))
    return _qdrant


def get_kg() -> nx.DiGraph:
    global _kg
    if _kg is None:
        graphml_files = list(KG_DIR.glob("*.graphml"))
        if not graphml_files:
            print(f"[warn] no .graphml file in {KG_DIR}. KG traversal disabled.")
            _kg = nx.DiGraph()
        else:
            path = graphml_files[0]
            print(f"[init] loading KG from {path}")
            _kg = nx.read_graphml(path)
            print(f"[init] KG: {_kg.number_of_nodes()} nodes, "
                  f"{_kg.number_of_edges()} edges")
    return _kg


def get_genai_client():
    global _genai_client
    if _genai_client is None:
        if not _GENAI_AVAILABLE:
            raise RuntimeError("google-genai not installed. `pip install google-genai`")
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY env var is empty.")
        _genai_client = genai.Client(api_key=GEMINI_API_KEY)
    return _genai_client


# =============================================================================
# 3. CHUNK DATACLASS
# =============================================================================

@dataclass
class Chunk:
    chunk_id: str
    manual: str
    page: Optional[int]
    content_type: str      # 'text' | 'table' | 'image'
    text: str              # what gets embedded
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "manual": self.manual,
            "page": self.page,
            "content_type": self.content_type,
            "text": self.text,
            **self.metadata,
        }


# =============================================================================
# 4. DATA LOADERS  --  normalize the 3 text schemas into uniform Chunks
# =============================================================================

def _slugify(s: str) -> str:
    """Turn a section heading into a filename-safe token."""
    s = re.sub(r"[^A-Za-z0-9]+", "_", s or "").strip("_").lower()
    return s or "section"


def _load_text_pre_chunked(manual_id: str, data: Dict[str, Any]) -> List[Chunk]:
    """
    EZ Clip style — already chunked. Each entry in text_chunks becomes a Chunk.
    """
    chunks: List[Chunk] = []
    for entry in data.get("text_chunks", []):
        cid = entry.get("chunk_id") or f"{manual_id}_pg{entry.get('page')}_txt{len(chunks)}"
        chunks.append(Chunk(
            chunk_id=cid,
            manual=manual_id,
            page=entry.get("page"),
            content_type="text",
            text=entry.get("content", ""),
            metadata={
                "section": entry.get("section", ""),
                "part_numbers": entry.get("part_numbers", []),
            },
        ))
    return chunks


def _load_text_pages_sections(manual_id: str, pages: List[Dict[str, Any]],
                              content_key: str) -> List[Chunk]:
    """
    Filter drier / React RA style — pages -> sections -> {heading, text-or-content}.
    Some sections use 'text' (string), some use 'content' (list of strings).
    We handle both.
    """
    chunks: List[Chunk] = []
    for page in pages:
        page_num = page.get("page")
        for sec in page.get("sections", []):
            heading = sec.get("heading", "")
            raw = sec.get(content_key)
            if raw is None:
                # Try the other key as fallback
                raw = sec.get("text" if content_key == "content" else "content")
            if raw is None:
                continue
            if isinstance(raw, list):
                body = "\n".join(str(x) for x in raw if x)
            else:
                body = str(raw)
            if not body.strip():
                continue
            cid = f"{manual_id}_pg{page_num}_{_slugify(heading)}"
            # Prepend heading into embedded text — improves retrieval on section-scoped queries
            text_for_embed = f"[{heading}]\n{body}" if heading else body
            chunks.append(Chunk(
                chunk_id=cid,
                manual=manual_id,
                page=page_num,
                content_type="text",
                text=text_for_embed,
                metadata={
                    "section": heading,
                    "part_numbers": [],
                },
            ))
    return chunks


def load_text_chunks(manual_id: str) -> List[Chunk]:
    """Dispatch by schema shape. Returns [] if the file is missing."""
    path = TEXT_DIR / f"{manual_id}_text.json"
    if not path.exists():
        print(f"[warn] no text file for {manual_id}")
        return []
    with open(path) as f:
        data = json.load(f)

    # React RA style: top-level array
    if isinstance(data, list):
        return _load_text_pages_sections(manual_id, data, content_key="content")

    # EZ Clip style: pre-chunked
    if isinstance(data, dict) and "text_chunks" in data:
        return _load_text_pre_chunked(manual_id, data)

    # Filter drier style: {manual, pages:[{page, sections:[{heading, text}]}]}
    if isinstance(data, dict) and "pages" in data:
        return _load_text_pages_sections(manual_id, data["pages"], content_key="text")

    print(f"[warn] unknown text schema for {manual_id}: {path}")
    return []


# ---- Tables --------------------------------------------------------------

def _serialize_table_rows(table: Dict[str, Any]) -> str:
    """
    Turn a table's rows into embedding-friendly text.
    Format: Title / Section header, then one 'Row: col=v; col=v' line per row.
    """
    title = table.get("title", "") or ""
    section = table.get("section", "") or ""
    header_parts = []
    if section: header_parts.append(f"Section: {section}")
    if title:   header_parts.append(f"Table: {title}")
    lines = [". ".join(header_parts)] if header_parts else []
    for row in table.get("data", []):
        if not isinstance(row, dict): continue
        row_str = "; ".join(f"{k}={v}" for k, v in row.items() if str(v).strip())
        if row_str:
            lines.append(f"Row: {row_str}")
    for note in table.get("notes", []) or []:
        lines.append(f"Note: {note}")
    return "\n".join(lines)


def load_table_chunks(manual_id: str) -> List[Chunk]:
    """Load tables JSON (array of table objects). Returns [] if missing."""
    path = TABLES_DIR / f"{manual_id}_tables.json"
    if not path.exists():
        print(f"[info] no tables file for {manual_id} (that's fine)")
        return []
    with open(path) as f:
        data = json.load(f)
    # Accept either a plain array or a wrapper dict with a "tables" key
    if isinstance(data, dict) and "tables" in data:
        data = data["tables"]
    if not isinstance(data, list):
        print(f"[warn] tables schema unexpected for {manual_id}")
        return []
    chunks: List[Chunk] = []
    for tbl in data:
        table_id = tbl.get("table_id") or f"t{len(chunks) + 1}"
        cid = f"{manual_id}_{table_id}"
        text = _serialize_table_rows(tbl)
        if not text.strip():
            continue
        chunks.append(Chunk(
            chunk_id=cid,
            manual=manual_id,
            page=tbl.get("page"),
            content_type="table",
            text=text,
            metadata={
                "section": tbl.get("section", ""),
                "title": tbl.get("title", ""),
                "raw_rows": tbl.get("data", []),   # keep for display
            },
        ))
    return chunks


# ---- Image captions ------------------------------------------------------

def _image_text_for_embedding(entry: Dict[str, Any]) -> str:
    """
    Combine the caption paragraph with flattened structured fields.
    This is what gets embedded — richer than caption alone, gives BGE-M3
    lexical AND semantic signal on components / specs / warnings.
    """
    parts: List[str] = []
    caption = entry.get("caption", "") or ""
    if caption:
        parts.append(caption)
    if entry.get("title"):
        parts.append(f"Title: {entry['title']}")
    if entry.get("diagram_type"):
        parts.append(f"Diagram type: {entry['diagram_type']}")
    if entry.get("section"):
        parts.append(f"Section: {entry['section']}")
    if entry.get("components"):
        parts.append("Components: " + "; ".join(str(c) for c in entry["components"]))
    if entry.get("specifications"):
        parts.append("Specifications: " + "; ".join(str(s) for s in entry["specifications"]))
    if entry.get("warnings"):
        parts.append("Warnings: " + "; ".join(str(w) for w in entry["warnings"]))
    if entry.get("part_numbers"):
        parts.append("Part numbers: " + "; ".join(str(p) for p in entry["part_numbers"]))
    if entry.get("equipment"):
        parts.append("Equipment: " + "; ".join(str(e) for e in entry["equipment"]))
    return "\n".join(parts)


def load_image_chunks(manual_id: str) -> List[Chunk]:
    """Load captions JSON. Each image entry becomes one Chunk."""
    path = CAPTIONS_DIR / f"{manual_id}_captions.json"
    if not path.exists():
        print(f"[warn] no captions file for {manual_id}")
        return []
    with open(path) as f:
        data = json.load(f)
    images = data.get("images", []) if isinstance(data, dict) else []
    chunks: List[Chunk] = []
    for entry in images:
        image_id = entry.get("image_id")
        if not image_id:
            continue
        cid = f"{manual_id}_{image_id}"
        text = _image_text_for_embedding(entry)
        if not text.strip():
            continue
        # Guess where the PNG lives (may not exist yet at index time)
        img_path = IMAGES_DIR / manual_id / f"{image_id}.png"
        chunks.append(Chunk(
            chunk_id=cid,
            manual=manual_id,
            page=entry.get("page"),
            content_type="image",
            text=text,
            metadata={
                "image_id": image_id,
                "image_path": str(img_path),
                "section": entry.get("section", ""),
                "title": entry.get("title", ""),
                "diagram_type": entry.get("diagram_type", ""),
                "caption": entry.get("caption", ""),
                "components": entry.get("components", []),
                "specifications": entry.get("specifications", []),
                "warnings": entry.get("warnings", []),
                "part_numbers": entry.get("part_numbers", []),
                "equipment": entry.get("equipment", []),
            },
        ))
    return chunks


# =============================================================================
# 5. BGE-M3 EMBEDDING (dense + sparse in one pass)
# =============================================================================

def embed(texts: List[str], batch_size: int = 12) -> Dict[str, Any]:
    """Return {'dense': ndarray[N,1024], 'sparse': list[dict[int,float]]}."""
    model = get_bge_m3()
    out = model.encode(
        texts,
        batch_size=batch_size,
        max_length=8192,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    # lexical_weights: list of {token_id_str: weight}
    sparse = [
        {int(tok): float(w) for tok, w in d.items() if float(w) > 0.0}
        for d in out["lexical_weights"]
    ]
    return {"dense": out["dense_vecs"], "sparse": sparse}


def _to_sparse_vector(sparse_dict: Dict[int, float]) -> qmodels.SparseVector:
    if not sparse_dict:
        return qmodels.SparseVector(indices=[0], values=[0.0])
    return qmodels.SparseVector(
        indices=list(sparse_dict.keys()),
        values=list(sparse_dict.values()),
    )


# =============================================================================
# 6. QDRANT SETUP + INDEXING
# =============================================================================

def _ensure_collection(name: str) -> None:
    client = get_qdrant()
    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        return
    print(f"[qdrant] creating collection '{name}'")
    client.create_collection(
        collection_name=name,
        vectors_config={
            "dense": qmodels.VectorParams(
                size=DENSE_DIM,
                distance=qmodels.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            "sparse": qmodels.SparseVectorParams(
                index=qmodels.SparseIndexParams(on_disk=False)
            )
        },
    )


def _upsert_chunks(collection: str, chunks: List[Chunk], batch: int = 32) -> int:
    if not chunks:
        return 0
    _ensure_collection(collection)
    client = get_qdrant()

    print(f"[qdrant] embedding {len(chunks)} chunks for '{collection}' ...")
    vecs = embed([c.text for c in chunks])

    points = []
    for i, ch in enumerate(chunks):
        points.append(qmodels.PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_OID, ch.chunk_id)),
            vector={
                "dense":  vecs["dense"][i].tolist(),
                "sparse": _to_sparse_vector(vecs["sparse"][i]),
            },
            payload=ch.to_payload(),
        ))
    for i in range(0, len(points), batch):
        client.upsert(collection_name=collection, points=points[i:i + batch])
    print(f"[qdrant] upserted {len(points)} → {collection}")
    return len(points)


# =============================================================================
# 7. INDEX_ALL  --  Phase 1 entry point
# =============================================================================

def index_all() -> Dict[str, int]:
    """Load every manual's text/tables/captions, embed, upsert. Idempotent."""
    text_chunks:  List[Chunk] = []
    table_chunks: List[Chunk] = []
    image_chunks: List[Chunk] = []

    for manual_id in MANUAL_IDS:
        print(f"\n[load] {manual_id}")
        t = load_text_chunks(manual_id)
        tb = load_table_chunks(manual_id)
        im = load_image_chunks(manual_id)
        print(f"       text={len(t)}  tables={len(tb)}  images={len(im)}")
        text_chunks  += t
        table_chunks += tb
        image_chunks += im

    counts = {
        COLL_TEXT:   _upsert_chunks(COLL_TEXT,   text_chunks),
        COLL_TABLES: _upsert_chunks(COLL_TABLES, table_chunks),
        COLL_IMAGES: _upsert_chunks(COLL_IMAGES, image_chunks),
    }
    print(f"\n[index_all] done: {counts}")
    return counts


# =============================================================================
# 8. QUERY TYPE DETECTION
# =============================================================================

QUERY_TYPE_PROMPT = """You are classifying a user question about Danfoss industrial
assembly manuals (DCR filter drier, React RA click, EZ Clip to 5400 Series).

Classify into EXACTLY ONE:
- "image"        : refers to a diagram / figure / "show me" / "what does X look like"
- "table"        : asks for a spec, torque value, dimension, or tabular data
- "procedure"    : asks how to do something, order of steps, install/braze/weld
- "part_number"  : asks for a part number given a description, or vice versa
- "out_of_scope" : not about these Danfoss products at all

Reply with ONLY the category, nothing else.

Question: {q}
Category:"""


def detect_query_type(query: str) -> str:
    try:
        client = get_genai_client()
        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=QUERY_TYPE_PROMPT.format(q=query),
            config=genai_types.GenerateContentConfig(
                temperature=0.0,
                # 1024 leaves room for Gemini 3.5 Flash's reasoning tokens
                # plus the one-word category answer.
                max_output_tokens=1024,
            ),
        )
        raw = (resp.text or "").strip().lower().strip('"').strip("'")
        for t in QUERY_TYPES:
            if t in raw:
                return t
    except Exception as e:
        print(f"[warn] query type detection failed: {e}")
    return "procedure"


# =============================================================================
# 9. QUERY EXPANSION
# =============================================================================

EXPANSION_PROMPT = """Rewrite the following technical question in 3 different ways
that could appear in a Danfoss assembly manual. Then write ONE short hypothetical
answer paragraph (2–3 sentences).

Return valid JSON with this exact shape:
{{
  "paraphrases": ["...", "...", "..."],
  "hypothetical_answer": "..."
}}

Question: {q}
JSON:"""


def _lenient_json_load(text: str) -> Optional[dict]:
    """Try hard to parse JSON that Gemini occasionally malforms
    (unquoted keys, markdown fences, missing commas, trailing commas)."""
    if not text:
        return None
    t = text.strip()
    # Strip markdown fences
    if t.startswith("```"):
        t = t.strip("`")
        if t.lower().startswith("json"):
            t = t[4:]
        t = t.strip()
    # Extract outermost braces
    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1:
        return None
    candidate = t[start:end + 1]
    # Strategy 1: strict parse
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    # Strategy 2: quote unquoted keys (common Gemini slip)
    fixed = re.sub(r'([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:',
                   r'\1"\2":', candidate)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    # Strategy 3: last-ditch regex extraction of paraphrases + hypothetical_answer
    # so at least query expansion can degrade gracefully with partial data
    result = {}
    paraphrases = re.findall(r'"([^"\n]{5,200})"', candidate)
    if paraphrases:
        result["paraphrases"] = paraphrases[:3]
    hyde_match = re.search(
        r'hypothetical[_"\']answer["\']?\s*:\s*"([^"]{5,500})"',
        candidate, re.IGNORECASE)
    if hyde_match:
        result["hypothetical_answer"] = hyde_match.group(1)
    return result if result else None


def expand_query(query: str) -> List[str]:
    expansions = [query]
    try:
        client = get_genai_client()
        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=EXPANSION_PROMPT.format(q=query),
            config=genai_types.GenerateContentConfig(
                temperature=0.2,
                # Bumped for Gemini 3.5 Flash's reasoning-token overhead
                max_output_tokens=2048,
                response_mime_type="application/json",
            ),
        )
        data = _lenient_json_load(resp.text or "")
        if data:
            for p in data.get("paraphrases", []) or []:
                if isinstance(p, str) and p.strip():
                    expansions.append(p.strip())
            hyde = data.get("hypothetical_answer")
            if isinstance(hyde, str) and hyde.strip():
                expansions.append(hyde.strip())
    except Exception as e:
        print(f"[warn] query expansion failed: {e}")
    return expansions[:5]


# =============================================================================
# 10. HYBRID VECTOR SEARCH  (dense + sparse fused via RRF)
# =============================================================================

def _hybrid_one(collection: str, dense_vec: List[float],
                sparse_vec: qmodels.SparseVector, top_k: int
               ) -> List[qmodels.ScoredPoint]:
    client = get_qdrant()
    resp = client.query_points(
        collection_name=collection,
        prefetch=[
            qmodels.Prefetch(query=dense_vec, using="dense", limit=top_k * 2),
            qmodels.Prefetch(query=sparse_vec, using="sparse", limit=top_k * 2),
        ],
        query=qmodels.FusionQuery(fusion=qmodels.Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )
    return resp.points


def hybrid_search(queries: List[str], collection: str, top_k: int) -> List[Dict[str, Any]]:
    """Run hybrid search across query variants, deduplicate on chunk_id."""
    q_vecs = embed(queries)
    dedup: Dict[str, Dict[str, Any]] = {}
    for i in range(len(queries)):
        try:
            hits = _hybrid_one(
                collection,
                q_vecs["dense"][i].tolist(),
                _to_sparse_vector(q_vecs["sparse"][i]),
                top_k,
            )
        except Exception as e:
            print(f"[warn] hybrid_search failed on {collection}: {e}")
            hits = []
        for h in hits:
            payload = h.payload or {}
            cid = payload.get("chunk_id", str(h.id))
            if cid not in dedup or h.score > dedup[cid]["score"]:
                dedup[cid] = {
                    "chunk_id": cid,
                    "score": float(h.score),
                    "text": payload.get("text", ""),
                    "payload": payload,
                }
    return sorted(dedup.values(), key=lambda x: x["score"], reverse=True)[:top_k]


# =============================================================================
# 11. CROSS-ENCODER RERANKING
# =============================================================================

def rerank(query: str, candidates: List[Dict[str, Any]],
           keep: int = RERANK_KEEP) -> List[Dict[str, Any]]:
    if not candidates:
        return []
    tokenizer, model = get_reranker()
    pairs = [[query, c["text"]] for c in candidates]
    with torch.no_grad():
        inputs = tokenizer(
            pairs, padding=True, truncation=True,
            return_tensors="pt", max_length=512,
        )
        if _HAS_CUDA:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        logits = model(**inputs).logits.view(-1).float()
        scores = torch.sigmoid(logits).cpu().tolist()
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:keep]


# =============================================================================
# 12. KG TRAVERSAL  --  MMGraphRAG-style (entity embedding + 1-hop)
# =============================================================================
#
# Follows the retrieval design in MMGraphRAG (Section 3.1):
#   1. Embed query and rank entities by cosine similarity → top-k seeds
#   2. Expand 1 hop from those seeds
#   3. Rank / deduplicate triples and apply a hard cap
#
# Signature of kg_retrieve is unchanged so retrieve() / evaluate.py keep working.
# vector_chunks is accepted for compatibility but ignored (pure entity-centric).
#

# Module-level cache for entity embeddings (built once per process)
_entity_emb_cache: Optional[Dict[str, Any]] = None


def _node_text_for_embedding(g: nx.DiGraph, node: str) -> str:
    """Text used to embed an entity (name + type + description)."""
    a = g.nodes[node]
    parts = [
        str(a.get("entity_name") or a.get("name") or node),
        str(a.get("alias", "")),
        str(a.get("part_number", "")),
        str(a.get("entity_type") or a.get("type") or ""),
        str(a.get("description", ""))[:400],
    ]
    return " ".join(p for p in parts if p and p != "None").strip()


def _node_label(g: nx.DiGraph, node: str) -> str:
    a = g.nodes[node]
    et = a.get("entity_type") or a.get("type") or ""
    name = a.get("entity_name") or a.get("name") or node
    return f"{name} [{et}]" if et else str(name)


def _build_entity_index(g: nx.DiGraph) -> Dict[str, Any]:
    """
    Pre-compute dense embeddings for every entity.
    Called once and cached. Uses the same BGE-M3 model as the rest of the pipeline.
    """
    global _entity_emb_cache
    if _entity_emb_cache is not None:
        return _entity_emb_cache

    nodes = list(g.nodes())
    texts = [_node_text_for_embedding(g, n) for n in nodes]

    # Keep only nodes that have usable text
    valid = [(n, t) for n, t in zip(nodes, texts) if t.strip()]
    if not valid:
        _entity_emb_cache = {"nodes": [], "vecs": np.zeros((0, DENSE_DIM))}
        return _entity_emb_cache

    valid_nodes, valid_texts = zip(*valid)

    model = get_bge_m3()
    out = model.encode(
        list(valid_texts),
        batch_size=32,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    vecs = out["dense_vecs"]
    # L2-normalise so cosine = dot product
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
    vecs = vecs / norms

    _entity_emb_cache = {
        "nodes": list(valid_nodes),
        "vecs": vecs,
    }
    print(f"[kg] entity index built: {len(valid_nodes)} entities")
    return _entity_emb_cache


def kg_retrieve(query: str,
                hops: int = 1,                    # paper default = 1
                mode: str = "hybrid",
                vector_chunks: Optional[List[Dict[str, Any]]] = None
               ) -> List[str]:
    """
    MMGraphRAG-style local subgraph retrieval.

    1. Embed query → rank entities by cosine similarity → top-k seeds
    2. Expand 1 hop (paper default; `hops` kept for API compatibility)
    3. Rank, deduplicate, apply KG_TRIPLE_CAP

    Parameters are kept identical to the original so the rest of the pipeline
    does not need changes. `vector_chunks` is ignored (entity-centric design).
    """
    g = get_kg()
    if g.number_of_nodes() == 0:
        return []

    # ------------------------------------------------------------------
    # 1. Entity vector retrieval → seed nodes
    # ------------------------------------------------------------------
    index = _build_entity_index(g)
    nodes = index["nodes"]
    ent_vecs = index["vecs"]

    if len(nodes) == 0:
        return []

    model = get_bge_m3()
    q_out = model.encode(
        [query],
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    q_vec = q_out["dense_vecs"][0]
    q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-9)

    sims = ent_vecs @ q_vec
    TOP_K_SEEDS = 10
    top_idx = np.argsort(-sims)[:TOP_K_SEEDS]

    # Soft threshold; always keep at least the best entity
    seeds = [nodes[i] for i in top_idx if sims[i] > 0.20]
    if not seeds:
        seeds = [nodes[int(np.argmax(sims))]]

    # ------------------------------------------------------------------
    # 2. One-hop expansion (paper default)
    # ------------------------------------------------------------------
    max_hops = max(1, min(hops, 2))   # safety cap

    visited = set(seeds)
    frontier = set(seeds)
    triples: List[Tuple[float, str]] = []   # (score, triple_string)

    for hop in range(max_hops):
        next_frontier = set()
        for node in frontier:
            for _, tgt, data in g.out_edges(node, data=True):
                rel = data.get("relation") or data.get("label") or "RELATED_TO"
                weight = float(data.get("weight", 1.0))
                triple = f"({_node_label(g, node)}) --[{rel}]--> ({_node_label(g, tgt)})"
                edge_score = weight + (5.0 if hop == 0 else 1.0)
                triples.append((edge_score, triple))
                if tgt not in visited:
                    next_frontier.add(tgt)
                    visited.add(tgt)

            for src, _, data in g.in_edges(node, data=True):
                rel = data.get("relation") or data.get("label") or "RELATED_TO"
                weight = float(data.get("weight", 1.0))
                triple = f"({_node_label(g, src)}) --[{rel}]--> ({_node_label(g, node)})"
                edge_score = weight + (5.0 if hop == 0 else 1.0)
                triples.append((edge_score, triple))
                if src not in visited:
                    next_frontier.add(src)
                    visited.add(src)

        frontier = next_frontier
        if not frontier:
            break

    # ------------------------------------------------------------------
    # 3. Rank, deduplicate, cap
    # ------------------------------------------------------------------
    triples.sort(key=lambda x: x[0], reverse=True)

    seen = set()
    out: List[str] = []
    for _, t in triples:
        if t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) >= KG_TRIPLE_CAP:
            break

    return out


# =============================================================================
# 13. RETRIEVAL ORCHESTRATION
# =============================================================================

def retrieve(query: str, query_type: str, mode: str = "hybrid") -> Dict[str, Any]:
    """
    Route to collections by query type, then behave per mode:
      "baseline" — vector + rerank + expansion. No KG.
      "kg_only"  — KG only. No vector search at all.
      "hybrid"   — everything.
    """
    if mode == "kg_only":
        triples = kg_retrieve(query, mode="kg_only")
        return {
            "chunks": [],
            "kg_triples": triples,
            "reranker_scores": [],
        }

    # baseline & hybrid both do vector retrieval
    expansions = expand_query(query)
    if query_type == "image":
        colls = [(COLL_IMAGES, IMAGE_TOP_K), (COLL_TEXT, TEXT_TOP_K)]
    elif query_type in ("table", "part_number"):
        colls = [(COLL_TABLES, TABLE_TOP_K), (COLL_TEXT, TEXT_TOP_K), (COLL_IMAGES, IMAGE_TOP_K)]
    elif query_type == "procedure":
        colls = [(COLL_TEXT, TEXT_TOP_K), (COLL_IMAGES, IMAGE_TOP_K), (COLL_TABLES, TABLE_TOP_K)]
    else:
        colls = [(COLL_TEXT, TEXT_TOP_K)]

    pool: List[Dict[str, Any]] = []
    for coll, k in colls:
        pool.extend(hybrid_search(expansions, coll, k))

    seen: Dict[str, Dict[str, Any]] = {}
    for c in pool:
        cid = c["chunk_id"]
        if cid not in seen or c["score"] > seen[cid]["score"]:
            seen[cid] = c
    candidates = list(seen.values())

    reranked = rerank(query, candidates, keep=FUSION_TOP_K)

    # KG only fires for hybrid mode
    triples: List[str] = []
    if mode == "hybrid":
        triples = kg_retrieve(query, mode="hybrid", vector_chunks=reranked)

    return {
        "chunks": reranked,
        "kg_triples": triples,
        "reranker_scores": [c.get("rerank_score", 0.0) for c in reranked],
    }


# =============================================================================
# 14. PROMPT TEMPLATES
# =============================================================================

BASE_INSTRUCTIONS = """You are a technical assistant for Danfoss industrial assembly manuals.
Answer ONLY using the provided context. Rules:

1. Every specific claim MUST cite its source using [source: <chunk_id>] inline.
2. If the context does not contain the answer, say so — do not invent facts.
3. Prefer exact values verbatim (torques with units, dimensions, part numbers).
4. Image captions describe diagrams — quote details from them and cite the image chunk_id.
5. Keep the answer concise and directly targeted."""

PROMPTS_BY_TYPE = {
    "image": BASE_INSTRUCTIONS + """

The user is asking about a visual/diagram. Describe what the relevant figure shows,
based on its caption. Cite the image chunk_id.""",

    "table": BASE_INSTRUCTIONS + """

The user is asking for a spec or tabular value. Quote the exact value with units.
Cite the table chunk_id.""",

    "procedure": BASE_INSTRUCTIONS + """

The user is asking about a procedure. Present the steps in order using manual wording.
Cite the source chunk for each step.""",

    "part_number": BASE_INSTRUCTIONS + """

The user is asking about a part number. Return the exact part number string and
describe what it refers to. Cite the source chunk_id.""",
}


def _format_context(chunks: List[Dict[str, Any]], triples: List[str]) -> str:
    lines = ["=== RETRIEVED PASSAGES ==="]
    for c in chunks:
        p = c["payload"]
        lines.append(
            f"[chunk_id: {c['chunk_id']}] "
            f"(manual={p.get('manual')}, page={p.get('page')}, "
            f"type={p.get('content_type')}, section={p.get('section','')})"
        )
        lines.append(p.get("text", "").strip())
        lines.append("")
    if triples:
        lines.append("=== KNOWLEDGE GRAPH FACTS ===")
        for t in triples:
            lines.append(t)
    return "\n".join(lines)


def build_prompt(query: str, query_type: str,
                 chunks: List[Dict[str, Any]], triples: List[str]) -> str:
    instr = PROMPTS_BY_TYPE.get(query_type, PROMPTS_BY_TYPE["procedure"])
    ctx = _format_context(chunks, triples)
    return f"{instr}\n\n{ctx}\n\n=== QUESTION ===\n{query}\n\n=== ANSWER ==="


# =============================================================================
# 15. GENERATION
# =============================================================================

def generate_answer(prompt: str) -> str:
    client = get_genai_client()
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            temperature=0.1,
            # 4096 leaves room for Gemini 3.5 Flash's internal reasoning tokens
            # (which count against this limit) plus the actual answer.
            max_output_tokens=4096,
        ),
    )
    return (resp.text or "").strip()


# =============================================================================
# 16. CONFIDENCE
# =============================================================================

def compute_confidence(scores: List[float]) -> Tuple[str, float]:
    if not scores:
        return "low", 0.0
    top = max(scores)
    if top >= CONF_HIGH: return "high", top
    if top >= CONF_MED:  return "medium", top
    return "low", top


# =============================================================================
# 17. SQLITE LOGGING
# =============================================================================

def _init_log_db() -> None:
    SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(SQLITE_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS query_logs (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp    TEXT    NOT NULL,
            query        TEXT    NOT NULL,
            query_type   TEXT,
            answer       TEXT,
            confidence   TEXT,
            top_score    REAL,
            top_sources  TEXT,
            latency_ms   INTEGER
        )
    """)
    conn.commit(); conn.close()


def log_query(query, query_type, answer, confidence, top_score,
              top_sources, latency_ms):
    _init_log_db()
    conn = sqlite3.connect(SQLITE_PATH)
    conn.execute(
        "INSERT INTO query_logs "
        "(timestamp,query,query_type,answer,confidence,top_score,top_sources,latency_ms) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (
            time.strftime("%Y-%m-%d %H:%M:%S"),
            query, query_type, answer, confidence, top_score,
            json.dumps(top_sources), latency_ms,
        ),
    )
    conn.commit(); conn.close()


# =============================================================================
# 18. answer_query()  --  Phase 2 entry point
# =============================================================================

def answer_query(query: str, mode: str = "hybrid") -> Dict[str, Any]:
    """
    End-to-end answer.
    mode:
      "baseline" — vector + rerank + expansion. No KG.
      "kg_only"  — KG only. No vector search or reranking.
      "hybrid"   — everything (default).
    """
    assert mode in ("baseline", "kg_only", "hybrid"), f"unknown mode: {mode}"
    t0 = time.time()

    qtype = detect_query_type(query)

    if qtype == "out_of_scope":
        latency = int((time.time() - t0) * 1000)
        log_query(query, qtype, OUT_OF_SCOPE_MSG, "n/a", 0.0, [], latency)
        return {
            "answer":     OUT_OF_SCOPE_MSG,
            "sources":    [],
            "confidence": "n/a",
            "top_score":  0.0,
            "query_type": qtype,
            "kg_triples": [],
            "latency_ms": latency,
            "mode":       mode,
        }

    ret = retrieve(query, qtype, mode=mode)
    chunks, triples = ret["chunks"], ret["kg_triples"]

    if not chunks and not triples:
        latency = int((time.time() - t0) * 1000)
        log_query(query, qtype, OUT_OF_SCOPE_MSG, "low", 0.0, [], latency)
        return {
            "answer":     OUT_OF_SCOPE_MSG,
            "sources":    [],
            "confidence": "low",
            "top_score":  0.0,
            "query_type": qtype,
            "kg_triples": [],
            "latency_ms": latency,
            "mode":       mode,
        }

    prompt = build_prompt(query, qtype, chunks, triples)
    answer = generate_answer(prompt)

    # Confidence source depends on mode
    if mode == "kg_only":
        # No reranker scores in kg_only. Confidence is a fixed medium if we got any triples.
        conf, top_score = ("medium", 0.5) if triples else ("low", 0.0)
    else:
        conf, top_score = compute_confidence(ret["reranker_scores"])

    sources = [
        {
            "chunk_id":     c["chunk_id"],
            "manual":       c["payload"].get("manual"),
            "page":         c["payload"].get("page"),
            "content_type": c["payload"].get("content_type"),
            "section":      c["payload"].get("section", ""),
            "rerank_score": c.get("rerank_score", 0.0),
            "image_path":   c["payload"].get("image_path"),
        }
        for c in chunks
    ]

    latency = int((time.time() - t0) * 1000)
    log_query(query, qtype, answer, conf, top_score,
              [s["chunk_id"] for s in sources[:5]], latency)

    return {
        "answer":     answer,
        "sources":    sources,
        "confidence": conf,
        "top_score":  top_score,
        "query_type": qtype,
        "kg_triples": triples,
        "latency_ms": latency,
        "mode":       mode,
    }


# =============================================================================
# 19. SCRIPT ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", action="store_true", help="Build the Qdrant index")
    ap.add_argument("--ask",   type=str, default=None, help="One-shot question")
    args = ap.parse_args()

    if args.index:
        index_all()
    if args.ask:
        r = answer_query(args.ask)
        print("\n=== ANSWER ===")
        print(r["answer"])
        print(f"\nconfidence: {r['confidence']} (top_score={r['top_score']:.3f}) "
              f"[{r['latency_ms']} ms]")
        print("sources:", [s["chunk_id"] for s in r["sources"][:5]])
