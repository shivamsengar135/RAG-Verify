"""Verify candidate references against the arXiv API and Crossref.

Usage: .venv\\Scripts\\python.exe code\\verify_refs.py
Reads configs/candidate_refs.txt (one arXiv ID or DOI per line, '#' comments allowed),
writes data/refs_verified.json with the official title, authors, year and venue (DOIs only).
"""
import json
import ssl
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import certifi

ROOT = Path(__file__).resolve().parents[1]
ATOM = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
UA = {"User-Agent": "ragverify-refcheck/0.1 (academic research)"}
SSL_CTX = ssl.create_default_context(cafile=certifi.where())


def fetch(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60, context=SSL_CTX) as r:
        return r.read()


def check_arxiv(ids):
    url = "http://export.arxiv.org/api/query?max_results=200&id_list=" + ",".join(ids)
    root = ET.fromstring(fetch(url))
    found = {}
    for e in root.findall("a:entry", ATOM):
        aid = e.find("a:id", ATOM).text.rsplit("/abs/", 1)[-1].split("v")[0]
        title = e.find("a:title", ATOM)
        if title is None:  # arXiv returns an error entry for unknown IDs
            continue
        doi = e.find("arxiv:doi", ATOM)
        jref = e.find("arxiv:journal_ref", ATOM)
        found[aid] = {
            "id": f"arXiv:{aid}",
            "title": " ".join(title.text.split()),
            "authors": [a.find("a:name", ATOM).text for a in e.findall("a:author", ATOM)],
            "year": e.find("a:published", ATOM).text[:4],
            "doi": doi.text if doi is not None else None,
            "journal_ref": jref.text if jref is not None else None,
            "abstract": " ".join(e.find("a:summary", ATOM).text.split()),
        }
    return [found.get(i, {"id": f"arXiv:{i}", "error": "NOT FOUND"}) for i in ids]


def check_doi(doi):
    try:
        m = json.loads(fetch(f"https://api.crossref.org/works/{doi}"))["message"]
    except Exception as exc:  # noqa: BLE001 - report any failure as unverified
        return {"id": f"doi:{doi}", "error": str(exc)}
    return {
        "id": f"doi:{doi}",
        "title": " ".join(m.get("title", [""])[0].split()),
        "authors": [f"{a.get('given', '')} {a.get('family', '')}".strip() for a in m.get("author", [])],
        "year": str(m.get("issued", {}).get("date-parts", [[None]])[0][0]),
        "venue": (m.get("container-title") or [None])[0],
    }


def main():
    lines = (ROOT / "configs" / "candidate_refs.txt").read_text().splitlines()
    items = [l.split("#")[0].strip() for l in lines]
    items = [i for i in items if i]
    arxiv_ids = [i for i in items if not i.startswith("10.")]
    dois = [i for i in items if i.startswith("10.")]

    results = check_arxiv(arxiv_ids)
    for d in dois:
        time.sleep(1)
        results.append(check_doi(d))

    out = ROOT / "data" / "refs_verified.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    for r in results:
        if "error" in r:
            print(f"[FAIL] {r['id']}: {r['error']}")
        else:
            first = r["authors"][0] if r["authors"] else "?"
            print(f"[OK]   {r['id']} | {r['year']} | {first} et al. | {r['title']}")


if __name__ == "__main__":
    main()
