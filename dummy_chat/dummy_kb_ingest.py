import os
import certifi
from datetime import datetime
from pymongo import MongoClient
from fastapi import UploadFile, HTTPException
from openai import OpenAI
import logging

from dummy_chat.dummy_utils import clean_text, chunk_text, generate_public_id
from dummy_chat.dummy_extractors import (
    extract_pdf,
    extract_docx,
    extract_txt,
    extract_link,
    LinkExtractionError,
)

logger = logging.getLogger(__name__)

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536

# Database configuration
DB_NAME = os.getenv("KB_DB_NAME", "skylix_rag")
COLLECTION_NAME = os.getenv("KB_COLLECTION_NAME", "kb_chunks")
VECTOR_INDEX_NAME = os.getenv("KB_VECTOR_INDEX", "kb_chunks_vector")

# ═══════════════════════════════════════════════════════════════════
# LAZY INITIALIZATION - Fixes the OPENAI_API_KEY loading issue
# ═══════════════════════════════════════════════════════════════════

_mongo_client = None
_openai_client = None


def get_mongo_client():
    """Lazy initialization of MongoDB client"""
    global _mongo_client
    if _mongo_client is None:
        mongodb_uri = os.getenv("MONGODB_URI")
        if not mongodb_uri:
            raise RuntimeError("MONGODB_URI environment variable not set")

        _mongo_client = MongoClient(
            mongodb_uri,
            tls=True,
            tlsCAFile=certifi.where()
        )
        logger.info(f"✓ MongoDB connected: {DB_NAME}.{COLLECTION_NAME}")
    return _mongo_client


def get_openai_client():
    """Lazy initialization of OpenAI client"""
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable not set")

        _openai_client = OpenAI(api_key=api_key)
        logger.info("✓ OpenAI client initialized")
    return _openai_client


def get_collection():
    """Get MongoDB collection"""
    mongo = get_mongo_client()
    db = mongo[DB_NAME]
    return db[COLLECTION_NAME]


def embed_chunks(chunks):
    """Generate embeddings for chunks"""
    logger.info(f"Generating embeddings for {len(chunks)} chunks...")

    openai_client = get_openai_client()

    res = openai_client.embeddings.create(
        model=EMBED_MODEL,
        input=chunks,
        encoding_format="float",
    )
    vectors = [d.embedding for d in res.data]

    for v in vectors:
        if len(v) != EMBED_DIM:
            raise RuntimeError("Embedding dimension mismatch")

    logger.info(f"✓ Generated {len(vectors)} embeddings")
    return vectors


async def ingest_kb(
    workspace_id: str,
    user_id: str,
    status: str,
    source_type: str,
    file: UploadFile | None = None,
    link: str | None = None,
    question: str | None = None,
    answer: str | None = None,
):
    """
    Single unified ingestion function
    Handles: file, link, qna
    """
    if not workspace_id:
        raise HTTPException(400, "workspace_id required")
    if not user_id:
        raise HTTPException(400, "user_id required")

    status_norm = (status or "published").strip().lower()
    if status_norm not in {"draft", "published"}:
        raise HTTPException(
            400, "Invalid status. Must be 'draft' or 'published'")

    logger.info("=" * 80)
    logger.info(f"📥 INGESTION START: {source_type}")
    logger.info(f"   Workspace: {workspace_id}")
    logger.info(f"   User: {user_id}")
    logger.info(f"   Status: {status_norm}")
    logger.info("=" * 80)

    # Get collection (lazy initialization)
    collection = get_collection()

    # ═══════════════════════════════════════════════════════════════
    # STEP 1: EXTRACTION
    # ═══════════════════════════════════════════════════════════════
    try:
        if source_type == "file":
            if not file:
                raise HTTPException(400, "file required")

            filename = file.filename
            logger.info(f"📄 Processing file: {filename}")

            try:
                file.file.seek(0)
            except Exception:
                pass

            name = filename.lower()
            if name.endswith(".pdf"):
                raw = extract_pdf(file)
            elif name.endswith(".docx"):
                raw = extract_docx(file)
            elif name.endswith(".txt"):
                raw = extract_txt(file)
            else:
                raise HTTPException(
                    400, "Unsupported file type. Supported: PDF, DOCX, TXT")

            source_name = filename

        elif source_type == "link":
            if not link:
                raise HTTPException(400, "link required")

            logger.info(f"🔗 Processing link: {link}")

            try:
                raw = extract_link(link)
            except LinkExtractionError as e:
                logger.error(f"Link extraction failed: {str(e)}")
                raise HTTPException(
                    400, "Extracted content is too short. Source is empty")

            source_name = link

        elif source_type == "qna":
            if not question or not answer:
                raise HTTPException(400, "question and answer required")

            raw = f"Q: {question}\nA: {answer}"
            source_name = f"Q&A: {question[:50]}..."
            logger.info(f"💬 Processing Q&A: {question[:50]}...")

        else:
            raise HTTPException(
                400, "Invalid source_type. Must be 'file', 'link', or 'qna'")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Extraction error: {e}", exc_info=True)
        raise HTTPException(500, f"Extraction failed: {str(e)}")

    # Log raw extraction
    logger.info(f"✓ Extracted {len(raw)} raw characters")
    logger.info(f"   Preview: {raw[:200]}...")

    # ═══════════════════════════════════════════════════════════════
    # STEP 2: CLEANING
    # ═══════════════════════════════════════════════════════════════
    raw = clean_text(raw)
    logger.info(f"✓ Cleaned to {len(raw)} characters")

    if len(raw) < 100:
        logger.warning(f"Insufficient text: {len(raw)} characters")
        raise HTTPException(
            400,
            f"Insufficient text ({len(raw)} chars). Minimum: 100 characters. "
            f"The source might be empty, behind paywall, or contain mostly images."
        )

    # ═══════════════════════════════════════════════════════════════
    # STEP 3: CHUNKING
    # ═══════════════════════════════════════════════════════════════
    try:
        chunks = chunk_text(raw)
        logger.info(f"✓ Created {len(chunks)} chunks")
        logger.info(
            f"   Chunk sizes: min={min(len(c) for c in chunks)}, max={max(len(c) for c in chunks)}")

    except Exception as e:
        logger.error(f"Chunking error: {e}", exc_info=True)
        raise HTTPException(500, f"Failed to chunk text: {str(e)}")

    # ═══════════════════════════════════════════════════════════════
    # STEP 4: EMBEDDING
    # ═══════════════════════════════════════════════════════════════
    try:
        embeddings = embed_chunks(chunks)
        logger.info(f"✓ Generated {len(embeddings)} embeddings")

    except Exception as e:
        logger.error(f"Embedding error: {e}", exc_info=True)
        raise HTTPException(500, f"Failed to generate embeddings: {str(e)}")

    # ═══════════════════════════════════════════════════════════════
    # STEP 5: STORE IN MONGODB
    # ═══════════════════════════════════════════════════════════════
    docs = []
    total = len(chunks)

    for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        docs.append({
            "public_id": generate_public_id(),
            "workspace_id": workspace_id,
            "user_id": user_id,
            "source_type": source_type,
            "source_name": source_name,
            "text": chunk,
            "embedding": emb,
            "chunk_index": idx,
            "total_chunks": total,
            "status": status_norm,
            "created_at": datetime.utcnow()
        })

    try:
        logger.info(f"💾 Inserting {total} documents into MongoDB...")
        logger.info(f"   Database: {DB_NAME}")
        logger.info(f"   Collection: {COLLECTION_NAME}")

        result = collection.insert_many(docs)
        logger.info(
            f"✓ Successfully inserted {len(result.inserted_ids)} documents")

        # Verify insertion
        verify_count = collection.count_documents(
            {"workspace_id": workspace_id})
        logger.info(
            f"✓ Total chunks in workspace '{workspace_id}': {verify_count}")

    except Exception as e:
        logger.error(f"Database insertion error: {e}", exc_info=True)
        raise HTTPException(500, f"Failed to store in MongoDB: {str(e)}")

    # ═══════════════════════════════════════════════════════════════
    # RETURN SUCCESS
    # ═══════════════════════════════════════════════════════════════
    logger.info("=" * 80)
    logger.info("✅ INGESTION COMPLETE")
    logger.info("=" * 80)

    return {
        "ok": True,
        "workspace_id": workspace_id,
        "user_id": user_id,
        "status": status_norm,
        "source_type": source_type,
        "source_name": source_name,
        "chunks_inserted": total,
        "characters_extracted": len(raw),
        "raw_text_preview": raw[:500],
        "chunk_preview": chunks[0][:200] if chunks else "",
        "database": DB_NAME,
        "collection": COLLECTION_NAME,
        "total_in_workspace": verify_count,
    }
