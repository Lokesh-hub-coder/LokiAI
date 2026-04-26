"""Quick smoke test for all API endpoints."""
import httpx
import json

BASE = "http://localhost:8080"

def check(label, r):
    d = r.json()
    print(f"{label}: HTTP {r.status_code}")
    return d

# /status
d = check("GET /status", httpx.get(BASE + "/status"))
print("  ", json.dumps(d, indent=4))

# /stats
d = check("GET /stats", httpx.get(BASE + "/stats"))
print("  ", d)

# /items
items = check("GET /items", httpx.get(BASE + "/items"))
print(f"   {len(items)} items, first metadata: {items[0]['metadata'][:40]}")

# /search (CS-biased 16D vector)
vec = ",".join(["0.9"] * 4 + ["0.1"] * 12)
d = check("GET /search", httpx.get(f"{BASE}/search?v={vec}&k=3&metric=cosine&algo=hnsw"))
print(f"   latency={d['latencyUs']}us  top={d['results'][0]['metadata'][:40]}")

# /benchmark
d = check("GET /benchmark", httpx.get(f"{BASE}/benchmark?v={vec}&k=5&metric=cosine"))
print(f"   bf={d['bruteforceUs']}us  kd={d['kdtreeUs']}us  hnsw={d['hnswUs']}us")

# /hnsw-info
d = check("GET /hnsw-info", httpx.get(BASE + "/hnsw-info"))
print(f"   topLayer={d['topLayer']}  nodeCount={d['nodeCount']}")

# /insert
d = check("POST /insert", httpx.post(
    BASE + "/insert",
    json={"metadata": "Test item", "category": "cs", "embedding": [0.1] * 16},
))
print(f"   new id={d['id']}")
new_id = d["id"]

# /delete/{id}
d = check(f"DELETE /delete/{new_id}", httpx.delete(f"{BASE}/delete/{new_id}"))
print(f"   ok={d['ok']}")

# /doc/list (empty initially)
d = check("GET /doc/list", httpx.get(BASE + "/doc/list"))
print(f"   {len(d)} docs")

print()
print("ALL ENDPOINTS OK ✓")
