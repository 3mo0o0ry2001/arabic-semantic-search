"""Dense retrieval with ONNX embedding models (CPU, no PyTorch required)."""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ModelSpec:
    name: str
    pooling: str           # "mean" or "cls"
    query_prefix: str = ""
    passage_prefix: str = ""
    max_length: int = 512


# Prefixes follow each model card: E5 needs "query: " / "passage: ", BGE-M3 needs none.
SPECS = {
    "e5-small": ModelSpec("multilingual-e5-small", "mean", "query: ", "passage: "),
    "e5-base": ModelSpec("multilingual-e5-base", "mean", "query: ", "passage: "),
    "bge-m3": ModelSpec("bge-m3", "cls", max_length=512),
}


def passage_text(p: dict) -> str:
    """Title is prepended so short paragraphs keep their topic."""
    return f"{p.get('title', '')}\n{p['text']}".strip()


def query_cache_key(queries: list[str]) -> str:
    return hashlib.sha256("\n".join(queries).encode()).hexdigest()[:12]


class OnnxEncoder:
    def __init__(self, model_dir: str | Path, spec: ModelSpec, onnx_file: str = "onnx/model_quantized.onnx",
                 threads: int = 0, session=None, tokenizer=None):
        self.spec = spec
        model_dir = Path(model_dir)
        if tokenizer is None:
            from tokenizers import Tokenizer
            tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        tokenizer.enable_truncation(max_length=spec.max_length)
        tokenizer.enable_padding()
        self.tok = tokenizer
        if session is None:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            if threads:
                opts.intra_op_num_threads = threads
            session = ort.InferenceSession(str(model_dir / onnx_file), opts, providers=["CPUExecutionProvider"])
        self.sess = session
        self.input_names = {i.name for i in session.get_inputs()}

    def _embed(self, texts: list[str], batch_size: int) -> np.ndarray:
        out = []
        for start in range(0, len(texts), batch_size):
            enc = self.tok.encode_batch(texts[start:start + batch_size])
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.input_names:
                feed["token_type_ids"] = np.zeros_like(ids)
            hidden = self.sess.run(None, feed)[0]          # (batch, seq, dim)
            if self.spec.pooling == "cls":
                vec = hidden[:, 0]
            else:
                m = mask[..., None].astype(hidden.dtype)
                vec = (hidden * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
            out.append(vec / np.linalg.norm(vec, axis=1, keepdims=True))
        return np.vstack(out).astype(np.float32)

    def encode_queries(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return self._embed([self.spec.query_prefix + t for t in texts], batch_size)

    def encode_passages(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        return self._embed([self.spec.passage_prefix + t for t in texts], batch_size)


class DenseRetriever:
    def __init__(self, encoder: OnnxEncoder, cache_dir: str | Path | None = None):
        self.encoder = encoder
        self.name = encoder.spec.name
        self.cache_dir = Path(cache_dir) if cache_dir else None

    def index(self, passages: list[dict], fingerprint: str = "") -> "DenseRetriever":
        self.ids = [p["id"] for p in passages]
        cache = self.cache_dir / f"{self.name}-{fingerprint}.npy" if self.cache_dir and fingerprint else None
        if cache and cache.exists():
            self.matrix = np.load(cache)
        else:
            t0 = time.perf_counter()
            self.matrix = self.encoder.encode_passages([passage_text(p) for p in passages])
            self.index_seconds = time.perf_counter() - t0
            if cache:
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache, self.matrix)
        return self

    def scores_batch(self, queries: list[str]) -> np.ndarray:
        qfile = self.cache_dir / f"{self.name}-{query_cache_key(queries)}.q.npy" if self.cache_dir else None
        if qfile and qfile.exists():
            q = np.load(qfile)
        else:
            q = self.encoder.encode_queries(queries)
            if qfile:
                qfile.parent.mkdir(parents=True, exist_ok=True)
                np.save(qfile, q)
        return q @ self.matrix.T

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        # Always encodes the query (no cache) so latency measurements are honest.
        s = (self.encoder.encode_queries([query]) @ self.matrix.T)[0]
        top = np.argsort(-s, kind="stable")[:k]
        return [(self.ids[i], float(s[i])) for i in top]
