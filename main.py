"""
LokiAI – Python/FastAPI backend
Exact functional replica of main.cpp (C++ / httplib implementation).

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8080 --reload
"""

from __future__ import annotations

import heapq
import math
import random
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# =====================================================================
#  CONSTANTS
# =====================================================================

DIMS = 16  # demo vectors are 16-dimensional

# =====================================================================
#  DATA TYPES
# =====================================================================

class VectorItem:
    __slots__ = ("id", "metadata", "category", "emb")

    def __init__(self, id: int, metadata: str, category: str, emb: List[float]):
        self.id = id
        self.metadata = metadata
        self.category = category
        self.emb = emb


DistFn = Callable[[List[float], List[float]], float]

# =====================================================================
#  DISTANCE METRICS
# =====================================================================

def euclidean(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a)
    nb = sum(y * y for y in b)
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    return 1.0 - dot / (math.sqrt(na) * math.sqrt(nb))


def manhattan(a: List[float], b: List[float]) -> float:
    return sum(abs(x - y) for x, y in zip(a, b))


def get_dist_fn(metric: str) -> DistFn:
    if metric == "cosine":
        return cosine
    if metric == "manhattan":
        return manhattan
    return euclidean

# =====================================================================
#  BRUTE FORCE
# =====================================================================

class BruteForce:
    def __init__(self):
        self.items: List[VectorItem] = []

    def insert(self, v: VectorItem) -> None:
        self.items.append(v)

    def knn(self, q: List[float], k: int, dist: DistFn) -> List[Tuple[float, int]]:
        scored = [(dist(q, v.emb), v.id) for v in self.items]
        scored.sort()
        return scored[:k]

    def remove(self, id: int) -> None:
        self.items = [v for v in self.items if v.id != id]

# =====================================================================
#  KD-TREE
# =====================================================================

class KDNode:
    __slots__ = ("item", "left", "right")

    def __init__(self, item: VectorItem):
        self.item = item
        self.left: Optional["KDNode"] = None
        self.right: Optional["KDNode"] = None


class KDTree:
    def __init__(self, dims: int):
        self.dims = dims
        self.root: Optional[KDNode] = None

    # ── insertion ──────────────────────────────────────────────────────
    def _ins(self, node: Optional[KDNode], v: VectorItem, depth: int) -> KDNode:
        if node is None:
            return KDNode(v)
        ax = depth % self.dims
        if v.emb[ax] < node.item.emb[ax]:
            node.left = self._ins(node.left, v, depth + 1)
        else:
            node.right = self._ins(node.right, v, depth + 1)
        return node

    def insert(self, v: VectorItem) -> None:
        self.root = self._ins(self.root, v, 0)

    # ── kNN search ─────────────────────────────────────────────────────
    def _knn(self, node: Optional[KDNode], q: List[float], k: int,
             depth: int, dist: DistFn, heap: list) -> None:
        """Max-heap of (-distance, id) so we can efficiently prune."""
        if node is None:
            return
        dn = dist(q, node.item.emb)
        if len(heap) < k:
            heapq.heappush(heap, (-dn, node.item.id))
        elif dn < -heap[0][0]:
            heapq.heapreplace(heap, (-dn, node.item.id))

        ax = depth % self.dims
        diff = q[ax] - node.item.emb[ax]
        closer  = node.left  if diff < 0 else node.right
        farther = node.right if diff < 0 else node.left

        self._knn(closer, q, k, depth + 1, dist, heap)
        if len(heap) < k or abs(diff) < -heap[0][0]:
            self._knn(farther, q, k, depth + 1, dist, heap)

    def knn(self, q: List[float], k: int, dist: DistFn) -> List[Tuple[float, int]]:
        heap: list = []
        self._knn(self.root, q, k, 0, dist, heap)
        result = [(-d, id_) for d, id_ in heap]
        result.sort()
        return result

    def rebuild(self, items: List[VectorItem]) -> None:
        self.root = None
        for v in items:
            self.insert(v)

# =====================================================================
#  HNSW — Hierarchical Navigable Small World
# =====================================================================

class HNSWNode:
    __slots__ = ("item", "max_lyr", "nbrs")

    def __init__(self, item: VectorItem, max_lyr: int):
        self.item = item
        self.max_lyr = max_lyr
        self.nbrs: List[List[int]] = [[] for _ in range(max_lyr + 1)]


class HNSW:
    def __init__(self, M: int = 16, ef_build: int = 200):
        self.M = M
        self.M0 = 2 * M
        self.ef_build = ef_build
        self.mL = 1.0 / math.log(float(M))
        self.rng = random.Random(42)

        self.G: Dict[int, HNSWNode] = {}
        self.top_layer = -1
        self.entry_pt = -1

    def _rand_level(self) -> int:
        return int(math.floor(-math.log(self.rng.random()) * self.mL))

    def _search_layer(self, q: List[float], ep: int, ef: int,
                      lyr: int, dist: DistFn) -> List[Tuple[float, int]]:
        vis: Dict[int, bool] = {}
        # cands: min-heap (distance, id)
        cands: list = []
        # found: max-heap stored as (-distance, id)
        found: list = []

        d0 = dist(q, self.G[ep].item.emb)
        vis[ep] = True
        heapq.heappush(cands, (d0, ep))
        heapq.heappush(found, (-d0, ep))

        while cands:
            cd, cid = heapq.heappop(cands)
            worst_found = -found[0][0]
            if len(found) >= ef and cd > worst_found:
                break
            node = self.G.get(cid)
            if node is None or lyr >= len(node.nbrs):
                continue
            for nid in node.nbrs[lyr]:
                if vis.get(nid) or nid not in self.G:
                    continue
                vis[nid] = True
                nd = dist(q, self.G[nid].item.emb)
                worst_found = -found[0][0]
                if len(found) < ef or nd < worst_found:
                    heapq.heappush(cands, (nd, nid))
                    heapq.heappush(found, (-nd, nid))
                    if len(found) > ef:
                        heapq.heappop(found)

        result = [(-d, id_) for d, id_ in found]
        result.sort()
        return result

    def _select_nbrs(self, cands: List[Tuple[float, int]], max_m: int) -> List[int]:
        return [id_ for _, id_ in cands[:max_m]]

    def insert(self, item: VectorItem, dist: DistFn) -> None:
        id_ = item.id
        lvl = self._rand_level()
        self.G[id_] = HNSWNode(item, lvl)

        if self.entry_pt == -1:
            self.entry_pt = id_
            self.top_layer = lvl
            return

        ep = self.entry_pt
        for lc in range(self.top_layer, lvl, -1):
            ep_node = self.G.get(ep)
            if ep_node and lc < len(ep_node.nbrs):
                W = self._search_layer(item.emb, ep, 1, lc, dist)
                if W:
                    ep = W[0][1]

        for lc in range(min(self.top_layer, lvl), -1, -1):
            W = self._search_layer(item.emb, ep, self.ef_build, lc, dist)
            max_m = self.M0 if lc == 0 else self.M
            sel = self._select_nbrs(W, max_m)
            self.G[id_].nbrs[lc] = sel

            for nid in sel:
                if nid not in self.G:
                    continue
                nbr_node = self.G[nid]
                if lc >= len(nbr_node.nbrs):
                    nbr_node.nbrs.extend([] for _ in range(lc + 1 - len(nbr_node.nbrs)))
                conn = nbr_node.nbrs[lc]
                conn.append(id_)
                if len(conn) > max_m:
                    ds = [(dist(nbr_node.item.emb, self.G[c].item.emb), c)
                          for c in conn if c in self.G]
                    ds.sort()
                    nbr_node.nbrs[lc] = [c for _, c in ds[:max_m]]
            if W:
                ep = W[0][1]

        if lvl > self.top_layer:
            self.top_layer = lvl
            self.entry_pt = id_

    def knn(self, q: List[float], k: int, ef: int,
            dist: DistFn) -> List[Tuple[float, int]]:
        if self.entry_pt == -1:
            return []
        ep = self.entry_pt
        for lc in range(self.top_layer, 0, -1):
            ep_node = self.G.get(ep)
            if ep_node and lc < len(ep_node.nbrs):
                W = self._search_layer(q, ep, 1, lc, dist)
                if W:
                    ep = W[0][1]
        W = self._search_layer(q, ep, max(ef, k), 0, dist)
        return W[:k]

    def remove(self, id_: int) -> None:
        if id_ not in self.G:
            return
        for nid, nd in self.G.items():
            for layer in nd.nbrs:
                if id_ in layer:
                    layer.remove(id_)
        if self.entry_pt == id_:
            self.entry_pt = -1
            for nid in self.G:
                if nid != id_:
                    self.entry_pt = nid
                    break
        del self.G[id_]

    def get_info(self) -> dict:
        top = self.top_layer
        max_l = max(top + 1, 1)
        nodes_per_layer = [0] * max_l
        edges_per_layer = [0] * max_l
        nodes = []
        edges = []

        for id_, nd in self.G.items():
            nodes.append({
                "id": id_,
                "metadata": nd.item.metadata,
                "category": nd.item.category,
                "maxLyr": nd.max_lyr,
            })
            for lc in range(min(nd.max_lyr + 1, max_l)):
                nodes_per_layer[lc] += 1
                if lc < len(nd.nbrs):
                    for nid in nd.nbrs[lc]:
                        if id_ < nid:
                            edges_per_layer[lc] += 1
                            edges.append({"src": id_, "dst": nid, "lyr": lc})

        return {
            "topLayer": top,
            "nodeCount": len(self.G),
            "nodesPerLayer": nodes_per_layer,
            "edgesPerLayer": edges_per_layer,
            "nodes": nodes,
            "edges": edges,
        }

    def size(self) -> int:
        return len(self.G)

# =====================================================================
#  VECTOR DATABASE  (demo 16D index)
# =====================================================================

class SearchOut:
    def __init__(self):
        self.hits: list = []
        self.us: int = 0
        self.algo: str = ""
        self.metric: str = ""


class BenchOut:
    def __init__(self, bf_us: int, kd_us: int, hnsw_us: int, n: int):
        self.bf_us = bf_us
        self.kd_us = kd_us
        self.hnsw_us = hnsw_us
        self.n = n


class VectorDB:
    def __init__(self, dims: int):
        self.dims = dims
        self._store: Dict[int, VectorItem] = {}
        self._bf = BruteForce()
        self._kdt = KDTree(dims)
        self._hnsw = HNSW(16, 200)
        self._lock = threading.Lock()
        self._next_id = 1

    def insert(self, meta: str, cat: str, emb: List[float], dist: DistFn) -> int:
        with self._lock:
            v = VectorItem(self._next_id, meta, cat, emb)
            self._next_id += 1
            self._store[v.id] = v
            self._bf.insert(v)
            self._kdt.insert(v)
            self._hnsw.insert(v, dist)
            return v.id

    def remove(self, id_: int) -> bool:
        with self._lock:
            if id_ not in self._store:
                return False
            del self._store[id_]
            self._bf.remove(id_)
            self._hnsw.remove(id_)
            self._kdt.rebuild(list(self._store.values()))
            return True

    def search(self, q: List[float], k: int, metric: str, algo: str) -> SearchOut:
        with self._lock:
            dfn = get_dist_fn(metric)
            t0 = time.perf_counter_ns()

            if algo == "bruteforce":
                raw = self._bf.knn(q, k, dfn)
            elif algo == "kdtree":
                raw = self._kdt.knn(q, k, dfn)
            else:
                raw = self._hnsw.knn(q, k, 50, dfn)

            us = (time.perf_counter_ns() - t0) // 1000

            out = SearchOut()
            out.us = us
            out.algo = algo
            out.metric = metric
            for d, id_ in raw:
                if id_ in self._store:
                    v = self._store[id_]
                    out.hits.append({
                        "id": id_,
                        "metadata": v.metadata,
                        "category": v.category,
                        "distance": d,
                        "emb": v.emb,
                    })
            return out

    def benchmark(self, q: List[float], k: int, metric: str) -> BenchOut:
        with self._lock:
            dfn = get_dist_fn(metric)

            def time_fn(fn) -> int:
                t = time.perf_counter_ns()
                fn()
                return (time.perf_counter_ns() - t) // 1000

            bf_us   = time_fn(lambda: self._bf.knn(q, k, dfn))
            kd_us   = time_fn(lambda: self._kdt.knn(q, k, dfn))
            hnsw_us = time_fn(lambda: self._hnsw.knn(q, k, 50, dfn))
            return BenchOut(bf_us, kd_us, hnsw_us, len(self._store))

    def all(self) -> List[VectorItem]:
        with self._lock:
            return list(self._store.values())

    def hnsw_info(self) -> dict:
        with self._lock:
            return self._hnsw.get_info()

    def size(self) -> int:
        with self._lock:
            return len(self._store)

# =====================================================================
#  DOCUMENT DATABASE  — HNSW over real Ollama embeddings
# =====================================================================

class DocItem:
    __slots__ = ("id", "title", "text", "emb")

    def __init__(self, id: int, title: str, text: str, emb: List[float]):
        self.id = id
        self.title = title
        self.text = text
        self.emb = emb


class DocumentDB:
    def __init__(self):
        self._store: Dict[int, DocItem] = {}
        self._hnsw = HNSW(16, 200)
        self._bf = BruteForce()
        self._lock = threading.Lock()
        self._next_id = 1
        self._dims = 0

    def insert(self, title: str, text: str, emb: List[float]) -> int:
        with self._lock:
            if self._dims == 0:
                self._dims = len(emb)
            item = DocItem(self._next_id, title, text, emb)
            self._next_id += 1
            self._store[item.id] = item
            vi = VectorItem(item.id, title, "doc", emb)
            self._hnsw.insert(vi, cosine)
            self._bf.insert(vi)
            return item.id

    def search(self, q: List[float], k: int,
               max_dist: float = 0.7) -> List[Tuple[float, DocItem]]:
        with self._lock:
            if not self._store:
                return []
            if len(self._store) < 10:
                raw = self._bf.knn(q, k, cosine)
            else:
                raw = self._hnsw.knn(q, k, 50, cosine)
            result = []
            for d, id_ in raw:
                if id_ in self._store and d <= max_dist:
                    result.append((d, self._store[id_]))
            return result

    def remove(self, id_: int) -> bool:
        with self._lock:
            if id_ not in self._store:
                return False
            del self._store[id_]
            self._hnsw.remove(id_)
            self._bf.remove(id_)
            return True

    def all(self) -> List[DocItem]:
        with self._lock:
            return list(self._store.values())

    def size(self) -> int:
        with self._lock:
            return len(self._store)

    def get_dims(self) -> int:
        return self._dims

# =====================================================================
#  TEXT CHUNKER
# =====================================================================

def chunk_text(text: str, chunk_words: int = 250, overlap_words: int = 30) -> List[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [text]

    chunks = []
    step = chunk_words - overlap_words
    i = 0
    while i < len(words):
        end = min(i + chunk_words, len(words))
        chunks.append(" ".join(words[i:end]))
        if end == len(words):
            break
        i += step
    return chunks

# =====================================================================
#  OLLAMA CLIENT
# =====================================================================

OLLAMA_HOST = "http://127.0.0.1:11434"
EMBED_MODEL = "nomic-embed-text"
GEN_MODEL   = "llama3.2"


def ollama_is_available() -> bool:
    try:
        with httpx.Client(timeout=2.0) as cli:
            r = cli.get(f"{OLLAMA_HOST}/api/tags")
            return r.status_code == 200
    except Exception:
        return False


def ollama_embed(text: str) -> List[float]:
    try:
        with httpx.Client(timeout=30.0) as cli:
            r = cli.post(
                f"{OLLAMA_HOST}/api/embeddings",
                json={"model": EMBED_MODEL, "prompt": text},
            )
            if r.status_code != 200:
                return []
            data = r.json()
            return [float(x) for x in data.get("embedding", [])]
    except Exception:
        return []


def ollama_generate(prompt: str) -> str:
    try:
        with httpx.Client(timeout=180.0) as cli:
            r = cli.post(
                f"{OLLAMA_HOST}/api/generate",
                json={"model": GEN_MODEL, "prompt": prompt, "stream": False},
            )
            if r.status_code != 200:
                return "ERROR: Ollama unavailable. Run: ollama serve"
            return r.json().get("response", "")
    except Exception:
        return "ERROR: Ollama unavailable. Run: ollama serve"

# =====================================================================
#  DEMO DATA   (16D categorical vectors, identical to C++ loadDemo)
# =====================================================================

DEMO_VECTORS = [
    ("Linked List: nodes connected by pointers", "cs",
     [0.90,0.85,0.72,0.68,0.12,0.08,0.15,0.10,0.05,0.08,0.06,0.09,0.07,0.11,0.08,0.06]),
    ("Binary Search Tree: O(log n) search and insert", "cs",
     [0.88,0.82,0.78,0.74,0.15,0.10,0.08,0.12,0.06,0.07,0.08,0.05,0.09,0.06,0.07,0.10]),
    ("Dynamic Programming: memoization overlapping subproblems", "cs",
     [0.82,0.76,0.88,0.80,0.20,0.18,0.12,0.09,0.07,0.06,0.08,0.07,0.08,0.09,0.06,0.07]),
    ("Graph BFS and DFS: breadth and depth first traversal", "cs",
     [0.85,0.80,0.75,0.82,0.18,0.14,0.10,0.08,0.06,0.09,0.07,0.06,0.10,0.08,0.09,0.07]),
    ("Hash Table: O(1) lookup with collision chaining", "cs",
     [0.87,0.78,0.70,0.76,0.13,0.11,0.09,0.14,0.08,0.07,0.06,0.08,0.07,0.10,0.08,0.09]),
    ("Calculus: derivatives integrals and limits", "math",
     [0.12,0.15,0.18,0.10,0.91,0.86,0.78,0.72,0.08,0.06,0.07,0.09,0.07,0.08,0.06,0.10]),
    ("Linear Algebra: matrices eigenvalues eigenvectors", "math",
     [0.20,0.18,0.15,0.12,0.88,0.90,0.82,0.76,0.09,0.07,0.08,0.06,0.10,0.07,0.08,0.09]),
    ("Probability: distributions random variables Bayes theorem", "math",
     [0.15,0.12,0.20,0.18,0.84,0.80,0.88,0.82,0.07,0.08,0.06,0.10,0.09,0.06,0.09,0.08]),
    ("Number Theory: primes modular arithmetic RSA cryptography", "math",
     [0.22,0.16,0.14,0.20,0.80,0.85,0.76,0.90,0.08,0.09,0.07,0.06,0.08,0.10,0.07,0.06]),
    ("Combinatorics: permutations combinations generating functions", "math",
     [0.18,0.20,0.16,0.14,0.86,0.78,0.84,0.80,0.06,0.07,0.09,0.08,0.06,0.09,0.10,0.07]),
    ("Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food",
     [0.08,0.06,0.09,0.07,0.07,0.08,0.06,0.09,0.90,0.86,0.78,0.72,0.08,0.06,0.09,0.07]),
    ("Sushi: vinegared rice raw fish and nori rolls", "food",
     [0.06,0.08,0.07,0.09,0.09,0.06,0.08,0.07,0.86,0.90,0.82,0.76,0.07,0.09,0.06,0.08]),
    ("Ramen: noodle soup with chashu pork and soft-boiled eggs", "food",
     [0.09,0.07,0.06,0.08,0.08,0.09,0.07,0.06,0.82,0.78,0.90,0.84,0.09,0.07,0.08,0.06]),
    ("Tacos: corn tortillas with carnitas salsa and cilantro", "food",
     [0.07,0.09,0.08,0.06,0.06,0.07,0.09,0.08,0.78,0.82,0.86,0.90,0.06,0.08,0.07,0.09]),
    ("Croissant: laminated pastry with buttery flaky layers", "food",
     [0.06,0.07,0.10,0.09,0.10,0.06,0.07,0.10,0.85,0.80,0.76,0.82,0.09,0.07,0.10,0.06]),
    ("Basketball: fast-paced shooting dribbling slam dunks", "sports",
     [0.09,0.07,0.08,0.10,0.08,0.09,0.07,0.06,0.08,0.07,0.09,0.06,0.91,0.85,0.78,0.72]),
    ("Football: tackles touchdowns field goals and strategy", "sports",
     [0.07,0.09,0.06,0.08,0.09,0.07,0.10,0.08,0.07,0.09,0.08,0.07,0.87,0.89,0.82,0.76]),
    ("Tennis: racket volleys groundstrokes and Wimbledon serves", "sports",
     [0.08,0.06,0.09,0.07,0.07,0.08,0.06,0.09,0.09,0.06,0.07,0.08,0.83,0.80,0.88,0.82]),
    ("Chess: openings endgames tactics strategic board game", "sports",
     [0.25,0.20,0.22,0.18,0.22,0.18,0.20,0.15,0.06,0.08,0.07,0.09,0.80,0.84,0.78,0.90]),
    ("Swimming: butterfly freestyle backstroke Olympic competition", "sports",
     [0.06,0.08,0.07,0.09,0.08,0.06,0.09,0.07,0.10,0.08,0.06,0.07,0.85,0.82,0.86,0.80]),
]


def load_demo(db: VectorDB) -> None:
    dist = get_dist_fn("cosine")
    for meta, cat, emb in DEMO_VECTORS:
        db.insert(meta, cat, emb, dist)

# =====================================================================
#  APPLICATION SETUP
# =====================================================================

app = FastAPI(title="LokiAI – VectorDB Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type"],
)

# Global state (mirrors C++ globals)
db     = VectorDB(DIMS)
doc_db = DocumentDB()

load_demo(db)

ollama_up = ollama_is_available()
print("=== VectorDB Engine ===")
print("http://localhost:8080")
print(f"{db.size()} demo vectors | {DIMS} dims | HNSW+KD-Tree+BruteForce")
print(f"Ollama: {'ONLINE' if ollama_up else 'OFFLINE (install from ollama.com)'}")
if ollama_up:
    print(f"  embed model: {EMBED_MODEL}  gen model: {GEN_MODEL}")

# =====================================================================
#  HELPER – parse comma-separated float vector from query param
# =====================================================================

def parse_vec(s: str) -> List[float]:
    result = []
    for part in s.split(","):
        try:
            result.append(float(part.strip()))
        except ValueError:
            pass
    return result

# =====================================================================
#  ROUTES — DEMO VECTOR ENDPOINTS
# =====================================================================

@app.get("/")
def serve_index():
    return FileResponse(
        "index.html",
        media_type="text/html",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/search")
def search(v: str = "", k: int = 5, metric: str = "cosine", algo: str = "hnsw"):
    q = parse_vec(v)
    if len(q) != DIMS:
        return JSONResponse({"error": f"need {DIMS}D vector"})
    out = db.search(q, k, metric, algo)
    return JSONResponse({
        "results": [
            {
                "id":        h["id"],
                "metadata":  h["metadata"],
                "category":  h["category"],
                "distance":  round(h["distance"], 6),
                "embedding": [round(x, 4) for x in h["emb"]],
            }
            for h in out.hits
        ],
        "latencyUs": out.us,
        "algo":      out.algo,
        "metric":    out.metric,
    })


@app.post("/insert")
async def insert(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid body"})

    meta = body.get("metadata", "")
    cat  = body.get("category", "")
    emb  = [float(x) for x in body.get("embedding", [])]

    if not meta or len(emb) != DIMS:
        return JSONResponse({"error": "invalid body"})

    id_ = db.insert(meta, cat, emb, get_dist_fn("cosine"))
    return JSONResponse({"id": id_})


@app.delete("/delete/{id_}")
def delete_item(id_: int):
    ok = db.remove(id_)
    return JSONResponse({"ok": ok})


@app.get("/items")
def get_items():
    items = db.all()
    return JSONResponse([
        {
            "id":        v.id,
            "metadata":  v.metadata,
            "category":  v.category,
            "embedding": [round(x, 4) for x in v.emb],
        }
        for v in items
    ])


@app.get("/benchmark")
def benchmark(v: str = "", k: int = 5, metric: str = "cosine"):
    q = parse_vec(v)
    if len(q) != DIMS:
        return JSONResponse({"error": f"need {DIMS}D vector"})
    b = db.benchmark(q, k, metric)
    return JSONResponse({
        "bruteforceUs": b.bf_us,
        "kdtreeUs":     b.kd_us,
        "hnswUs":       b.hnsw_us,
        "itemCount":    b.n,
    })


@app.get("/hnsw-info")
def hnsw_info():
    info = db.hnsw_info()
    return JSONResponse(info)


@app.get("/stats")
def stats():
    return JSONResponse({
        "count":      db.size(),
        "dims":       DIMS,
        "algorithms": ["bruteforce", "kdtree", "hnsw"],
        "metrics":    ["euclidean", "cosine", "manhattan"],
    })


@app.get("/status")
def status():
    up = ollama_is_available()
    return JSONResponse({
        "ollamaAvailable": up,
        "embedModel":      EMBED_MODEL,
        "genModel":        GEN_MODEL,
        "docCount":        doc_db.size(),
        "docDims":         doc_db.get_dims(),
        "demoDims":        DIMS,
        "demoCount":       db.size(),
    })

# =====================================================================
#  ROUTES — DOCUMENT + RAG ENDPOINTS
# =====================================================================

@app.post("/doc/insert")
async def doc_insert(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid body"})

    title = body.get("title", "").strip()
    text  = body.get("text",  "").strip()
    if not title or not text:
        return JSONResponse({"error": "need title and text"})

    chunks = chunk_text(text, 250, 30)
    ids: List[int] = []

    for i, chunk in enumerate(chunks):
        emb = ollama_embed(chunk)
        if not emb:
            return JSONResponse({
                "error": (
                    "Ollama unavailable. Install from https://ollama.com then run: "
                    "ollama pull nomic-embed-text && ollama pull llama3.2"
                )
            })
        chunk_title = (
            f"{title} [{i+1}/{len(chunks)}]" if len(chunks) > 1 else title
        )
        ids.append(doc_db.insert(chunk_title, chunk, emb))

    return JSONResponse({
        "ids":    ids,
        "chunks": len(chunks),
        "dims":   doc_db.get_dims(),
    })


@app.delete("/doc/delete/{id_}")
def doc_delete(id_: int):
    ok = doc_db.remove(id_)
    return JSONResponse({"ok": ok})


@app.get("/doc/list")
def doc_list():
    docs = doc_db.all()
    result = []
    for d in docs:
        preview = d.text[:120] + ("…" if len(d.text) > 120 else "")
        words   = len(d.text.split())
        result.append({
            "id":      d.id,
            "title":   d.title,
            "preview": preview,
            "words":   words,
        })
    return JSONResponse(result)


@app.post("/doc/search")
async def doc_search(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid body"})

    question = body.get("question", "").strip()
    k        = int(body.get("k", 3))
    if not question:
        return JSONResponse({"error": "need question"})

    q_emb = ollama_embed(question)
    if not q_emb:
        return JSONResponse({"error": "Ollama unavailable"})

    hits = doc_db.search(q_emb, k)
    return JSONResponse({
        "contexts": [
            {"id": item.id, "title": item.title, "distance": round(d, 4)}
            for d, item in hits
        ]
    })


@app.post("/doc/ask")
async def doc_ask(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid body"})

    question = body.get("question", "").strip()
    k        = int(body.get("k", 3))
    if not question:
        return JSONResponse({"error": "need question"})

    # Step 1: embed the question
    q_emb = ollama_embed(question)
    if not q_emb:
        return JSONResponse({"error": "Ollama unavailable"})

    # Step 2: retrieve top-k relevant chunks
    hits = doc_db.search(q_emb, k)

    # Step 3: build prompt (identical wording to C++)
    ctx_parts = []
    for i, (_, item) in enumerate(hits):
        ctx_parts.append(f"[{i+1}] {item.title}:\n{item.text}\n")
    context_str = "\n".join(ctx_parts)

    prompt = (
        "You are a helpful assistant. Answer the user's question directly. "
        "Use the provided context if it contains relevant information. "
        "If it doesn't, just use your own general knowledge. "
        "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like "
        "'the context doesn't mention'. "
        "Just answer the question naturally.\n\n"
        f"Context:\n{context_str}\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

    # Step 4: generate answer
    answer = ollama_generate(prompt)

    # Step 5: return everything
    return JSONResponse({
        "answer":   answer,
        "model":    GEN_MODEL,
        "contexts": [
            {
                "id":       item.id,
                "title":    item.title,
                "text":     item.text,
                "distance": round(d, 4),
            }
            for d, item in hits
        ],
        "docCount": doc_db.size(),
    })
