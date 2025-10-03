import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Iterable, List, Tuple
import fnmatch


def _ensure_dependencies() -> None:
    try:
        import pypdf  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "pypdf is required. Install with: pip install pypdf"
        ) from exc
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "sentence-transformers is required. Install with: pip install sentence-transformers"
        ) from exc
    try:
        import qdrant_client  # type: ignore  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "qdrant-client is required. Install with: pip install qdrant-client"
        ) from exc


_ensure_dependencies()

from pypdf import PdfReader  # type: ignore  # noqa: E402
from sentence_transformers import SentenceTransformer  # type: ignore  # noqa: E402
try:
    from qdrant_client import QdrantClient  # type: ignore  # noqa: E402
    from qdrant_client.http import models as rest  # type: ignore  # noqa: E402
except Exception:  # pragma: no cover
    QdrantClient = None  # type: ignore
    rest = None  # type: ignore


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class Chunk:
    text: str
    page: int
    source: str


def read_pdf_pages(pdf_path: str) -> List[Tuple[int, str]]:
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    # Some inputs may be mislabeled as .pdf but are plain text. Detect and fallback.
    try:
        with open(pdf_path, "rb") as f:
            header = f.read(5)
    except Exception:
        header = b""
    if not header.startswith(b"%PDF-"):
        # Fallback: treat as UTF-8 text file
        try:
            with open(pdf_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            return [(1, text)]
        except Exception:
            # If even text read fails, propagate error
            raise

    reader = PdfReader(pdf_path)
    pages: List[Tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages.append((i, text))
    return pages


def _split_into_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_page_text(page_text: str, page_num: int, source: str, chunk_size: int, chunk_overlap: int) -> List[Chunk]:
    sentences = _split_into_sentences(page_text)
    if not sentences:
        return []

    chunks: List[Chunk] = []
    current: List[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if current:
            text = " ".join(current).strip()
            if text:
                chunks.append(Chunk(text=text, page=page_num, source=source))
        current = []
        current_len = 0

    for sent in sentences:
        if current_len + len(sent) + 1 <= chunk_size:
            current.append(sent)
            current_len += len(sent) + 1
        else:
            flush()
            # Start new chunk; include overlap from the end of previous chunk
            if chunk_overlap > 0 and chunks:
                overlap_text = chunks[-1].text
                if len(overlap_text) > chunk_overlap:
                    overlap_text = overlap_text[-chunk_overlap:]
                current = [overlap_text, sent]
                current_len = len(overlap_text) + 1 + len(sent)
            else:
                current = [sent]
                current_len = len(sent)
    flush()
    return chunks


def chunk_pdf(pdf_path: str, chunk_size: int = 800, chunk_overlap: int = 120) -> List[Chunk]:
    pages = read_pdf_pages(pdf_path)
    chunks: List[Chunk] = []
    for page_num, text in pages:
        chunks.extend(chunk_page_text(text, page_num, os.path.basename(pdf_path), chunk_size, chunk_overlap))
    return chunks


class EmbeddingModel:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    def _load(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: Iterable[str]) -> List[List[float]]:
        model = self._load()
        embeddings = model.encode(list(texts), show_progress_bar=False, normalize_embeddings=True)
        return embeddings.tolist()  # type: ignore[no-any-return]


"""FAISS removal: Qdrant-only implementation"""


def _ensure_qdrant_collection(client: "QdrantClient", collection: str, vector_size: int) -> None:
    try:
        exists = client.collection_exists(collection_name=collection)  # type: ignore[attr-defined]
    except Exception:
        exists = False
    if not exists:
        client.create_collection(  # type: ignore[attr-defined]
            collection_name=collection,
            vectors_config=rest.VectorParams(size=vector_size, distance=rest.Distance.COSINE),
        )


def _upsert_qdrant(client: "QdrantClient", collection: str, vectors: List[List[float]], chunks: List[Chunk]) -> None:
    points = []
    for i, (vec, ch) in enumerate(zip(vectors, chunks)):
        payload = {"text": ch.text, "page": ch.page, "source": ch.source}
        points.append(rest.PointStruct(id=i, vector=vec, payload=payload))
    client.upsert(collection_name=collection, points=points)  # type: ignore[attr-defined]


def build_index_multi_qdrant(pdf_paths: List[str], model_name: str, qdrant_url: str, qdrant_port: int, qdrant_api_key: str | None, collection: str, chunk_size: int, chunk_overlap: int) -> None:
    if QdrantClient is None or rest is None:
        raise RuntimeError("qdrant-client is required. Install with: pip install qdrant-client")
    all_chunks: List[Chunk] = []
    for path in pdf_paths:
        try:
            all_chunks.extend(chunk_pdf(path, chunk_size=chunk_size, chunk_overlap=chunk_overlap))
        except Exception as exc:
            print(f"Warning: failed to process {path}: {exc}", file=sys.stderr)
    if not all_chunks:
        raise ValueError("No text extracted from provided PDFs.")
    embedder = EmbeddingModel(model_name)
    vectors = embedder.encode(c.text for c in all_chunks)
    if not vectors:
        raise ValueError("Failed to compute embeddings.")
    dim = len(vectors[0])
    client = QdrantClient(url=qdrant_url, port=qdrant_port, api_key=qdrant_api_key)  # type: ignore[call-arg]
    _ensure_qdrant_collection(client, collection, dim)
    _upsert_qdrant(client, collection, vectors, all_chunks)


def query_qdrant(qdrant_url: str, qdrant_port: int, qdrant_api_key: str | None, collection: str, query: str, k: int, model_name: str) -> List[Tuple[dict, float]]:
    if QdrantClient is None or rest is None:
        raise RuntimeError("qdrant-client is required. Install with: pip install qdrant-client")
    embedder = EmbeddingModel(model_name)
    query_vec = embedder.encode([query])[0]
    client = QdrantClient(url=qdrant_url, port=qdrant_port, api_key=qdrant_api_key)  # type: ignore[call-arg]
    hits = client.search(collection_name=collection, query_vector=query_vec, limit=k)  # type: ignore[attr-defined]
    results: List[Tuple[dict, float]] = []
    for h in hits:
        payload = h.payload or {}
        meta = {
            "id": h.id,
            "text": payload.get("text", ""),
            "page": payload.get("page"),
            "source": payload.get("source"),
        }
        results.append((meta, float(h.score)))
    return results


def _collect_pdfs(pdf_args: List[str] | None, pdf_dir: str | None, pattern: str) -> List[str]:
    collected: List[str] = []
    if pdf_args:
        for item in pdf_args:
            if "," in item:
                parts = [p.strip() for p in item.split(",") if p.strip()]
                collected.extend(parts)
            else:
                collected.append(item)
    if pdf_dir:
        for root, _dirs, files in os.walk(pdf_dir):
            for name in files:
                if fnmatch.fnmatch(name, pattern):
                    collected.append(os.path.join(root, name))
    # Normalize and de-duplicate while preserving order
    seen = set()
    unique: List[str] = []
    for p in collected:
        ap = os.path.abspath(p)
        if ap not in seen and (ap.lower().endswith(".pdf") or ap.lower().endswith(".txt")) and os.path.exists(ap):
            seen.add(ap)
            unique.append(ap)
    return unique


"""FAISS query removed"""


def _print_results(results: List[Tuple[dict, float]]) -> None:
    for rank, (meta, score) in enumerate(results, start=1):
        preview = meta["text"].replace("\n", " ")
        if len(preview) > 220:
            preview = preview[:220] + "…"
        print(f"#{rank} score={score:.4f} page={meta.get('page')} source={meta.get('source')}")
        print(preview)
        print("-")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build and query a Qdrant index for PDF/text RAG.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="Build Qdrant index from one or more PDFs/texts")
    p_build.add_argument("--pdf", action="append", help="Path to input PDF. Can be repeated or comma-separated.")
    p_build.add_argument("--pdf-dir", help="Directory to scan for PDFs (recursive)")
    p_build.add_argument("--pattern", default="*.pdf", help="Filename pattern for --pdf-dir, default *.pdf")
    p_build.add_argument("--qdrant-url", default="http://localhost", help="Qdrant URL")
    p_build.add_argument("--qdrant-port", type=int, default=6333, help="Qdrant port")
    p_build.add_argument("--qdrant-api-key", default=None, help="Qdrant API key (if any)")
    p_build.add_argument("--qdrant-collection", default="rag_chunks", help="Qdrant collection name")
    p_build.add_argument("--model", default=DEFAULT_MODEL_NAME, help="SentenceTransformer model name")
    p_build.add_argument("--chunk-size", type=int, default=800)
    p_build.add_argument("--chunk-overlap", type=int, default=120)

    p_query = sub.add_parser("query", help="Query an existing Qdrant index")
    p_query.add_argument("query", help="Query text")
    p_query.add_argument("--qdrant-url", default="http://localhost")
    p_query.add_argument("--qdrant-port", type=int, default=6333)
    p_query.add_argument("--qdrant-api-key", default=None)
    p_query.add_argument("--qdrant-collection", default="rag_chunks")
    p_query.add_argument("--k", type=int, default=5)
    p_query.add_argument("--model", default=DEFAULT_MODEL_NAME)

    args = parser.parse_args(argv)

    if args.command == "build":
        pdfs = _collect_pdfs(args.pdf, args.pdf_dir, args.pattern)
        if not pdfs:
            raise SystemExit("No PDFs found. Provide --pdf and/or --pdf-dir.")
        build_index_multi_qdrant(pdf_paths=pdfs, model_name=args.model, qdrant_url=args.qdrant_url, qdrant_port=args.qdrant_port, qdrant_api_key=args.qdrant_api_key, collection=args.qdrant_collection, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
        print(f"Indexed {len(pdfs)} file(s) -> Qdrant collection '{args.qdrant_collection}' at {args.qdrant_url}:{args.qdrant_port}")
        return 0
    if args.command == "query":
        results = query_qdrant(qdrant_url=args.qdrant_url, qdrant_port=args.qdrant_port, qdrant_api_key=args.qdrant_api_key, collection=args.qdrant_collection, query=args.query, k=args.k, model_name=args.model)
        _print_results(results)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

