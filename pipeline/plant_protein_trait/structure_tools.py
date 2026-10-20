"""Optional structure utilities: retrieval, conversion and exact chain mapping."""
import argparse
import gzip
import json
import re
import shutil
import zipfile
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

from . import annotation
from .core import digest, fasta, fresh_dir, manifest, read_json, run_lock, verify_manifest, write_json
from .jobs import cached
from .net import HTTPFailure, request, snapshot
from .parsers import parse_blast


def bio():
    try:
        from Bio.PDB import MMCIFParser, PDBParser, PDBIO, Select
        from Bio.Data.PDBData import protein_letters_3to1
        return MMCIFParser, PDBParser, PDBIO, Select, protein_letters_3to1
    except ImportError:
        raise RuntimeError("Biopython is required for structure conversion/mapping; activate the supplied environment") from None


def read_chains(path):
    _, parser, _, _, letters = bio()
    structure = parser(QUIET=False, PERMISSIVE=False).get_structure("model", str(path))
    model = next(structure.get_models())
    chains = {}
    for chain in model:
        residues = [r for r in chain if r.id[0] == " " or r.resname.upper() in letters]
        if residues:
            chains[chain.id] = ("".join(letters.get(r.resname.upper(), "X") for r in residues), residues)
    return structure, model, chains


def structure_evidence(path, query, origin, chain_id=None):
    _, _, chains = read_chains(path)
    matches = [(name, residues) for name, (seq, residues) in chains.items()
               if seq == query["sequence"] and (chain_id is None or name == chain_id)]
    if len(matches) != 1:
        raise ValueError("Structure must have one selected chain matching the full protein exactly; specify --structure-chain when needed")
    name, residues = matches[0]
    result = {"status": "available", "source": "user_supplied_" + origin,
              "source_kind_verification": "origin declared by user; full chain sequence checked",
              "chain": name, "sequence_sha256": query["sequence_sha256"],
              "avg_plddt": None, "tm_score": None, "foldseek": "not_run"}
    if origin != "experimental":
        values = [float(r["CA"].bfactor) for r in residues if "CA" in r]
        if len(values) != query["length"] or any(not 0 <= x <= 100 for x in values):
            raise ValueError("Predicted structure lacks one valid CA pLDDT value per query residue")
        result.update(avg_plddt=sum(values) / len(values), residues_scored=len(values),
                      fraction_plddt_below_70=sum(x < 70 for x in values) / len(values),
                      plddt_method="one CA per residue in the exact matched chain, first model")
    result["caution"] = "Structural confidence is not evidence of a phenotype; no structure similarity score was computed"
    return result


def finish(root, state, name):
    write_json(root / (name + ".json"), state)
    write_json(root / "manifest.json", manifest(root, [p.relative_to(root).as_posix()
               for p in root.rglob("*") if p.is_file() and p.name != "manifest.json"],
               operation=name, status=state["status"]))


def uniprot_candidates(hits):
    pattern = re.compile(r"(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})(?:-\d+)?")
    result = []
    for hit in hits:
        candidates = [hit.get("accession", "")]
        candidates += re.findall(r"(?:sp|tr)\|([^|]+)\|", hit.get("hit_id", "") + " " + hit.get("description", ""))
        for candidate in candidates:
            if pattern.fullmatch(candidate) and candidate not in result:
                result.append(candidate)
                break
    return result


def download_models(args):
    query = fasta(args.fasta)
    with run_lock(args.output):
        root = Path(args.output).resolve()
        spec = {"sequence_sha256": query["sequence_sha256"], "accessions": args.accession,
                "top_k": args.top_k, "format": args.prefer, "blast_xml": args.blast_xml,
                "blast_xml_sha256": digest(Path(args.blast_xml).read_bytes()) if args.blast_xml else None,
                "blast_db": args.local_db, "online": args.online_blast}
        if args.resume:
            if read_json(root / "request.json") != spec:
                raise ValueError("Structure retrieval settings changed; use a new output directory")
            if (root / "manifest.json").exists():
                verify_manifest(root, read_json(root / "manifest.json"))
                return root
        else:
            fresh_dir(root)
            write_json(root / "request.json", spec)
        if args.accession:
            accs = args.accession
        else:
            if not any([args.blast_xml, args.local_db, args.online_blast]):
                raise ValueError("Provide --accession, --blast-xml, --local-db (Swiss-Prot), or explicitly --online-blast")
            ann = annotation.parser().parse_args(["--fasta", args.fasta, "--output", str(root / "blast"),
                     "--database", "swissprot", "--max-hits", str(max(20, args.top_k)),
                     "--max-wait", str(args.max_wait)])
            ann.blast_xml, ann.blast_db, ann.online_blast = args.blast_xml, args.local_db, args.online_blast
            ann.email, ann.threads = args.email, args.threads
            ann.resume = args.resume and (root / "blast/request.json").exists()
            folder = annotation.collect(ann)
            hits, _ = parse_blast(folder / "raw/blast.xml", query)
            accs = uniprot_candidates(hits)[:args.top_k]
        raw = root / "raw"
        raw.mkdir(exist_ok=True)
        rows = []
        for acc in accs:
            if uniprot_candidates([{"accession": acc}]) != [acc]:
                raise ValueError("Invalid UniProt accession")
            api = "https://alphafold.ebi.ac.uk/api/prediction/" + acc
            saved = cached(raw, acc + ".json")
            try:
                data = json.loads(saved.read_bytes() if saved else request(api))
            except HTTPFailure as exc:
                if exc.code == 404:
                    rows.append({"accession": acc, "status": "not_found"})
                    continue
                raise
            if not saved:
                snapshot(raw, acc + ".json", json.dumps(data).encode(), api)
            entries = data if isinstance(data, list) else [data]
            for index, entry in enumerate(entries):
                link = entry.get("pdbUrl" if args.prefer == "pdb" else "cifUrl")
                if not link:
                    rows.append({"accession": acc, "status": "no_requested_format"})
                    continue
                parsed = urlsplit(link)
                if parsed.scheme != "https" or parsed.hostname != "alphafold.ebi.ac.uk":
                    raise ValueError("Unexpected AFDB model download host")
                name = f"AFDB_{acc}_{index + 1}.{args.prefer}"
                model = cached(raw, name) or snapshot(raw, name, request(link, timeout=120), link)
                exact = entry.get("uniprotSequence", "").upper() == query["sequence"] and entry.get("uniprotStart", 1) == 1 and entry.get("uniprotEnd", query["length"]) == query["length"]
                rows.append({"accession": acc, "model_id": entry.get("entryId"), "file": "raw/" + model.name,
                             "status": "downloaded", "role": "exact_target" if exact else "homolog_or_unverified_reference"})
        state = {"status": "completed" if any(x["status"] == "downloaded" for x in rows) else "no_models",
                 "query_sequence_sha256": query["sequence_sha256"], "models": rows,
                 "note": "Existing AFDB predictions only; homologs are not target structures or structure similarity evidence"}
        finish(root, state, "models")
        return root


def download_main(argv=None):
    p = argparse.ArgumentParser(description="Retrieve existing AFDB models; no de novo structure prediction")
    p.add_argument("--fasta", "--seq_or_fasta", required=True, help="One protein FASTA file")
    p.add_argument("--output", "--outdir", required=True)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--accession", action="append", help="Explicit UniProt accession; repeatable")
    group.add_argument("--blast-xml")
    group.add_argument("--local-db", "--local_db", help="Swiss-Prot BLAST database prefix for reference retrieval")
    group.add_argument("--online-blast", action="store_true")
    p.add_argument("--email")
    p.add_argument("--max-wait", type=int, default=1800)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--top-k", "--top_k", type=int, default=5)
    p.add_argument("--prefer", choices=["pdb", "cif"], default="pdb")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args(argv)
    if args.top_k < 1:
        p.error("top-k must be positive")
    print(download_models(args))


def collect_cifs(source, raw, max_bytes):
    candidates = sorted(source.rglob("*")) if source.is_dir() else [source]
    result = []
    used = 0
    def save(label, data):
        nonlocal used
        used += len(data)
        if used > max_bytes:
            raise ValueError("Uncompressed CIF input exceeds --max-input-mb")
        target = raw / (digest(label.encode())[:16] + ".cif")
        target.write_bytes(data)
        result.append((label, target))
    for path in candidates:
        if not path.is_file():
            continue
        if path.suffix.lower() in {".cif", ".mmcif"}:
            if path.stat().st_size > max_bytes - used:
                raise ValueError("CIF input exceeds --max-input-mb")
            save(str(path), path.read_bytes())
        elif path.name.lower().endswith((".cif.gz", ".mmcif.gz")):
            with gzip.open(path, "rb") as handle:
                save(str(path), handle.read(max_bytes - used + 1))
        elif path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                for info in sorted(archive.infolist(), key=lambda x: x.filename):
                    name = PurePosixPath(info.filename.replace("\\", "/"))
                    if name.is_absolute() or ".." in name.parts or ":" in info.filename:
                        raise ValueError("Unsafe archive member path")
                    if not info.is_dir() and name.suffix.lower() in {".cif", ".mmcif"}:
                        if info.file_size > max_bytes - used:
                            raise ValueError("Archive CIF input exceeds --max-input-mb")
                        save(str(path) + "::" + info.filename, archive.read(info))
    return result


def convert(args):
    source, out = Path(args.input).resolve(), Path(args.output).resolve()
    if not source.exists():
        raise ValueError("Input file/directory does not exist")
    if source.is_dir() and (out == source or out.is_relative_to(source)):
        raise ValueError("Conversion output must be outside the input directory")
    parser, _, writer, _, _ = bio()
    with run_lock(out):
        fresh_dir(out)
        raw = out / "raw"
        raw.mkdir()
        cifs = collect_cifs(source, raw, args.max_input_mb * 1024 * 1024)
        if not cifs:
            raise ValueError("No CIF/mmCIF input found")
        rows = []
        for label, path in cifs:
            structure = parser(QUIET=False).get_structure(path.stem, str(path))
            # PDB cannot represent multi-character chain IDs. Fail explicitly rather than silently rename.
            if any(len(chain.id) != 1 for model in structure for chain in model):
                raise ValueError("PDB requires one-character chain IDs; retain mmCIF or explicitly remap chains before conversion")
            target = out / (path.stem + ".pdb")
            io = writer()
            io.set_structure(structure)
            io.save(str(target))
            rows.append({"source": label, "source_sha256": digest(path.read_bytes()), "pdb": target.name})
        finish(out, {"status": "completed", "converted": rows, "note": "Format conversion only; no AlphaFold prediction"}, "conversion")
    return out


def convert_main(argv=None):
    p = argparse.ArgumentParser(description="Convert existing CIF/mmCIF, gzip or ZIP inputs to PDB; no model prediction")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-input-mb", type=int, default=2048)
    args = p.parse_args(argv)
    if args.max_input_mb < 1:
        p.error("max-input-mb must be positive")
    print(convert(args))


def map_models(args):
    query = fasta(args.fasta)
    source, out = Path(args.pdb_dir).resolve(), Path(args.output).resolve()
    if out == source or out.is_relative_to(source):
        raise ValueError("Mapping output must be outside the input PDB directory")
    files = sorted(source.rglob("*.pdb"))
    if not files:
        raise ValueError("No PDB inputs found")
    _, _, writer, select, _ = bio()
    rows, unmatched = [], []
    with run_lock(out):
        fresh_dir(out)
        for path in files:
            structure, model, chains = read_chains(path)
            found = False
            for name, (seq, residues) in chains.items():
                if seq != query["sequence"]:
                    continue
                found = True
                class ChainOnly(select):
                    def accept_model(self, candidate):
                        return candidate.id == model.id
                    def accept_chain(self, candidate):
                        return candidate.id == name
                target = out / ("target_" + digest((str(path) + "::" + name).encode())[:16] + ".pdb")
                io = writer()
                io.set_structure(structure)
                io.save(str(target), select=ChainOnly())
                rows.append({"query_id": query["id"], "source": str(path), "source_sha256": digest(path.read_bytes()),
                             "chain": name, "file": target.name, "match": "exact_full_sequence"})
            if not found:
                unmatched.append(str(path))
        finish(out, {"status": "completed" if rows else "no_exact_match", "mappings": rows,
                     "unmatched": unmatched, "query_sequence_sha256": query["sequence_sha256"]}, "mapping")
    if not rows:
        raise ValueError("No complete chain matches the query; no target structure was assigned. See mapping.json")
    return out


def map_main(argv=None):
    p = argparse.ArgumentParser(description="Match complete PDB chain sequences to one protein; first model, no approximate matching")
    p.add_argument("--fasta", required=True)
    p.add_argument("--pdb-dir", required=True)
    p.add_argument("--output", required=True)
    print(map_models(p.parse_args(argv)))
