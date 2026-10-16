"""Parsers keep observation, annotation transfer, and missingness distinct."""
from __future__ import annotations

import re
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from .core import finite, read_json


def parse_blast(path: Path, query: dict) -> tuple[list[dict], dict]:
    root = ET.fromstring(path.read_bytes())
    for el in root.iter():
        el.tag = el.tag.split("}")[-1]
    old = root.tag == "BlastOutput"
    searches = root.findall(".//Iteration") if old else root.findall(".//Search")
    if len(searches) != 1:
        raise ValueError("BLAST XML must contain exactly one query")
    search = searches[0]
    qlen = search.findtext("Iteration_query-len" if old else "query-len")
    if int(qlen or 0) != query["length"]:
        raise ValueError("BLAST query length differs from FASTA")
    def value(el, key):
        return el.findtext(("Hsp_" if old else "") + key)
    hits = []
    for aln in search.findall(".//Hit"):
        desc = aln if old else aln.find(".//HitDescr")
        if desc is None:
            continue
        title = desc.findtext("Hit_def" if old else "title") or ""
        hit_id = desc.findtext("Hit_id" if old else "id") or ""
        accession = desc.findtext("Hit_accession" if old else "accession") or hit_id
        organism = desc.findtext("sciname") or ""
        if not organism:
            m = re.search(r"\[([^\[\]]+)\]\s*$", title)
            organism = m.group(1) if m else ""
        hsps = []
        covered = set()
        for h in aln.findall(".//Hsp"):
            start, end = int(value(h, "query-from")), int(value(h, "query-to"))
            hseq = (value(h, "qseq") or "").replace("-", "").upper()
            if not (1 <= start <= end <= query["length"]):
                raise ValueError("BLAST query coordinates out of bounds")
            if not hseq or hseq != query["sequence"][start - 1:end]:
                raise ValueError("BLAST aligned query does not match the supplied FASTA")
            length = int(value(h, "align-len"))
            identities = int(value(h, "identity"))
            if length <= 0 or not 0 <= identities <= length:
                raise ValueError("Invalid BLAST alignment statistics")
            covered.update(range(start, end + 1))
            hsps.append({"query_start": start, "query_end": end,
                         "subject_start": int(value(h, "hit-from")),
                         "subject_end": int(value(h, "hit-to")),
                         "alignment_length": length, "identities": identities,
                         "identity_pct": 100 * identities / length,
                         "query_coverage_pct": 100 * (end - start + 1) / query["length"],
                         "e_value": float(value(h, "evalue")),
                         "bit_score": float(value(h, "bit-score")),
                         "raw_score": float(value(h, "score")),
                         "query_aligned": value(h, "qseq"), "subject_aligned": value(h, "hseq")})
        if hsps:
            best = max(hsps, key=lambda x: x["bit_score"])
            hits.append({"accession": accession, "hit_id": hit_id, "description": title,
                         "organism": organism, "subject_length": int(aln.findtext("Hit_len" if old else "len")),
                         **{key: best[key] for key in ["identity_pct", "query_coverage_pct", "e_value", "bit_score"]},
                         "query_coverage_union_pct": 100 * len(covered) / query["length"],
                         "statistics_scope": "best HSP; union coverage reported separately",
                         "hsps": hsps})
    meta = {"database": root.findtext("BlastOutput_db") if old else root.findtext(".//Target/db"),
            "program_version": root.findtext("BlastOutput_version") if old else root.findtext(".//Report/version"),
            "query_binding": "length and all aligned query residues validated",
            "caution": "Identity supports homology, not orthology or experimental function"}
    return sorted(hits, key=lambda h: (-h["bit_score"], h["e_value"], h["accession"], h["hit_id"])), meta


def parse_interpro(path: Path, query: dict) -> tuple[list[dict], list[dict]]:
    data = read_json(path)
    records = data.get("results", [data]) if isinstance(data, dict) else data
    if not isinstance(records, list):
        raise ValueError("Unsupported InterProScan JSON")
    matches = [r for r in records if r.get("sequence", "").upper() == query["sequence"]]
    if len(matches) != 1:
        raise ValueError("InterProScan JSON must contain exactly one exact target sequence match")
    domains, gos = [], {}
    for m in matches[0].get("matches", []):
        sig = m.get("signature", {})
        entry = sig.get("entry") or {}
        lib = sig.get("signatureLibraryRelease") or {}
        d = {"accession": sig.get("accession"), "name": sig.get("name") or sig.get("description"),
             "database": lib.get("library"), "database_version": lib.get("version"),
             "interpro_accession": entry.get("accession"),
             "interpro_name": entry.get("name") or entry.get("description"),
             "e_value": finite(m.get("evalue")), "score": finite(m.get("score")),
             "locations": m.get("locations", []), "annotation_method": "InterProScan exact query"}
        domains.append(d)
        for container in (matches[0], m, sig, entry):
            for g in container.get("goXRefs", []) + container.get("goTerms", []):
                gid = g.get("id")
                if not gid:
                    continue
                category = g.get("category", {})
                if isinstance(category, dict):
                    category = category.get("name") or category.get("code")
                item = gos.setdefault(gid, {"id": gid, "name": g.get("name"), "category": category,
                                           "annotation_method": "InterPro2GO/computational transfer",
                                           "experimental_on_query": False, "source_signatures": []})
                if sig.get("accession") not in item["source_signatures"]:
                    item["source_signatures"].append(sig.get("accession"))
    for domain in domains:
        domain["locations"] = sorted(domain["locations"], key=lambda value: json.dumps(value, sort_keys=True))
    for go in gos.values():
        go["source_signatures"] = sorted(go["source_signatures"], key=lambda value: value or "")
    return domains, sorted(gos.values(), key=lambda g: g["id"])


def parse_domtbl(path: Path, query: dict) -> list[dict]:
    domains = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        p = line.split(maxsplit=22)
        if len(p) < 22 or p[3] != query["id"] or int(p[5]) != query["length"]:
            raise ValueError("HMMER query ID/length mismatch or malformed domtblout")
        if float(p[12]) <= 0.001:
            domains.append({"accession": p[1] if p[1] != "-" else p[0], "name": p[0],
                            "database": "Pfam", "e_value": float(p[12]),
                            "score": float(p[13]), "full_sequence_e_value": float(p[6]),
                            "locations": [{"start": int(p[17]), "end": int(p[18])}],
                            "annotation_method": "hmmscan; independent domain E-value <= 0.001"})
    return domains


def load_annotation(directory: Path, query: dict) -> dict:
    """Read real 4_cotton outputs; no synthetic final_report or ranked file required."""
    path = directory / "annotation_full.json"
    if not path.is_file():
        raise ValueError("annotation_full.json is required; score-only TSV is not protein evidence")
    data = read_json(path)
    if isinstance(data, dict) and "records" in data:
        data = data["records"]
    if isinstance(data, dict) and "seq_id" in data:
        data = [data]
    if isinstance(data, dict):
        records = [dict(v, seq_id=v.get("seq_id", k)) for k, v in data.items() if isinstance(v, dict)]
    elif isinstance(data, list):
        records = data
    else:
        raise ValueError("Unsupported annotation_full.json shape")
    selected = [r for r in records if r.get("seq_id") == query["id"]]
    if len(selected) != 1:
        raise ValueError("Exactly one annotation record must match the FASTA ID")
    rec = dict(selected[0])
    if int(rec.get("length", query["length"])) != query["length"]:
        raise ValueError("Annotation length differs from FASTA")
    if rec.get("sequence_sha256") and rec["sequence_sha256"] != query["sequence_sha256"]:
        raise ValueError("Annotation sequence hash differs from FASTA")
    if not rec.get("sequence_sha256"):
        rec["binding_warning"] = "Legacy input has no sequence SHA256; ID/length binding only"
    rec["domains"] = rec.get("domains", rec.get("interpro_domains", []))
    st = dict(rec.get("structure") or {})
    sidecar = directory / "structure_info.json"
    if sidecar.exists():
        extra = read_json(sidecar).get(query["id"], {})
        st = {**extra, **st}
    proxy = st.get("reason") == "proxy_score" or not (st.get("pdb_path") or st.get("source"))
    if proxy:
        st = {"status": "missing", "avg_plddt": None, "tm_score": None,
              "reason": "No measured structure evidence; legacy proxy scores excluded"}
    else:
        st.setdefault("status", "legacy_unverified")
        st["avg_plddt"] = finite(st.get("avg_plddt", st.get("avg_pLDDT", st.get("plddt"))))
        st["tm_score"] = finite(st.get("tm_score", st.get("best_tm_score")))
        st.pop("structure_score", None)
    rec["structure"] = st
    return rec
