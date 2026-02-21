from __future__ import annotations
from Bio.Seq import Seq
import requests
import os
import sys
import json
import math
import time
import hashlib
import logging
import argparse
import subprocess
import csv
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

try:
    from Bio import SeqIO
    from Bio.Blast import NCBIWWW, NCBIXML
except Exception:
    SeqIO = None
    NCBIWWW = None
    NCBIXML = None
CONFIG = {
    "paths": {
        "output_dir": "annotation_results",
        "af3_pred_dir": "af3_predictions",
        "pdb_lib_dir": "pdb_lib",
        "tmp_dir": "tmp",
    },
    "tools": {
        "blastp": "blastp",
        "hmmscan": "hmmscan",
        "tmalign": "TMalign",
        "foldseek": "foldseek",
        "interproscan": "interproscan.sh",
    },
    "weights": {
        "blast": 0.50,
        "domain": 0.30,
        "go": 0.10,
        "structure": 0.10
    },
    "thresholds": {
        "pass_score": 60,
        "plddt": 70.0,
        "tm_strong": 0.7,
        "tm_moderate": 0.5
    },
    "runtime": {
        "max_threads": 4,
        "top_n_for_structure": 50
    }
}

for p in CONFIG['paths'].values():
    os.makedirs(p, exist_ok=True)

log = logging.getLogger("protein_pipeline")
log.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
log.addHandler(handler)
import shutil
def which(tool: str) -> Optional[str]:
    path = shutil.which(tool)
    return path
TOOLS_AVAILABLE = {k: bool(which(v)) for k, v in CONFIG['tools'].items()}
log.info(f"tool available: {TOOLS_AVAILABLE}")
def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)
def md5_of_seq(seq: str) -> str:
    return hashlib.md5(seq.encode()).hexdigest()
def run_cmd(cmd: List[str], timeout: int = 300) -> Tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "timeout"
def clean_sequences(input_fasta: str, out_fasta: str, min_len: int = 50, max_len: int = 5000) -> Dict[str, Any]:
    if SeqIO is None:
        raise RuntimeError("Biopython is required for sequence parsing. Install biopython.")
    ensure_dir(os.path.dirname(out_fasta) or '.')
    records = list(SeqIO.parse(input_fasta, "fasta"))
    kept = []
    stats = {"total": len(records), "too_short": 0, "too_long": 0, "invalid": 0, "kept": 0}
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    md5_map = {}
    for rec in records:
        seq = str(rec.seq).upper().replace('*', '')
        if len(seq) < min_len:
            stats['too_short'] += 1
            continue
        if len(seq) > max_len:
            stats['too_long'] += 1
            continue
        if set(seq) - valid_aa:
            stats['invalid'] += 1
            continue
        rec.seq = Seq(seq)
        kept.append(rec)
        stats['kept'] += 1
        md5_map[rec.id] = {"md5": md5_of_seq(seq), "length": len(seq)}
    from Bio import SeqIO as _s
    _s.write(kept, out_fasta, "fasta")
    stats['out_fasta'] = out_fasta
    stats['md5_map'] = md5_map
    log.info(f"clean_sequences: {stats}")
    return stats
def run_blast_local(fasta: str, out_xml: str, db: str = "nr", evalue: float = 1e-5, max_hits: int = 5,
                    threads: int = 1) -> str:
    blastp_bin = which(CONFIG['tools']['blastp'])
    if blastp_bin:
        cmd = [blastp_bin, "-query", fasta, "-db", db, "-evalue", str(evalue), "-outfmt", "5", "-num_threads",
               str(threads), "-max_target_seqs", str(max_hits), "-out", out_xml]
        rc, out, err = run_cmd(cmd, timeout=3600)
        if rc == 0:
            log.info(f"blastp local finished: {out_xml}")
            return out_xml
        else:
            log.warning(f"blastp failed rc={rc} err={err} - falling back to remote")
    if NCBIWWW is None:
        raise RuntimeError("NCBIWWW not available. Install Biopython or install BLAST+ locally.")

    records = list(SeqIO.parse(fasta, "fasta"))
    results = {}
    for rec in records:
        seq = str(rec.seq)
        log.info(f"Submitting qblast for {rec.id}")
        handle = NCBIWWW.qblast("blastp", "nr", seq, expect=evalue, hitlist_size=max_hits)
        seq_xml = out_xml + f".{rec.id}.xml"
        with open(seq_xml, 'w') as fh:
            fh.write(handle.read())
        results[rec.id] = seq_xml
        time.sleep(2)
    return out_xml
def parse_blast_xml(xml_path: str) -> Dict[str, Any]:
    res = {}
    if os.path.isdir(xml_path):
        files = [os.path.join(xml_path, f) for f in os.listdir(xml_path) if f.endswith('.xml')]
    else:
        files = [xml_path]
    for fpath in files:
        try:
            with open(fpath) as fh:
                blast_iter = NCBIXML.parse(fh)
                for rec in blast_iter:
                    qid = rec.query.split()[0]
                    hits = []
                    for aln in rec.alignments[:5]:
                        for hsp in aln.hsps[:1]:
                            hits.append({
                                'accession': aln.accession,
                                'title': aln.title,
                                'length': aln.length,
                                'e_value': hsp.expect,
                                'score': hsp.score,
                                'identity': hsp.identities / hsp.align_length * 100,
                                'coverage': hsp.align_length / rec.query_length * 100 if rec.query_length else 0
                            })
                    res[qid] = {'hits': hits}
        except Exception as e:
            log.warning(f"Failed to parse BLAST XML {fpath}: {e}")
    return res
def run_interpro_local(fasta: str, out_json: str, interproscan_bin: Optional[str] = None) -> Optional[str]:
    bin_path = interproscan_bin or which(CONFIG['tools']['interproscan'])
    if not bin_path:
        return None
    cmd = [bin_path, "-i", fasta, "-f", "json", "-o", out_json, "-goterms", "-pa"]
    rc, out, err = run_cmd(cmd, timeout=3600)
    if rc == 0:
        log.info(f"InterProScan local finished: {out_json}")
        return out_json
    log.warning("InterProScan local failed")
    if err:
        log.warning(f"InterProScan stderr: {err[:2000]}")
    return None
def run_interpro_remote(fasta_seq: str, title: str, email: str, max_wait: int = 30) -> Optional[Dict]:
    url = "https://www.ebi.ac.uk/Tools/services/rest/iprscan5/run"
    params = {'email': email, 'sequence': fasta_seq, 'title': title, 'goterms': 'true', 'pathways': 'true'}
    try:
        r = requests.post(url, data=params, timeout=30)
        if r.status_code != 200:
            log.warning(f"InterPro remote submit failed: {r.status_code}")
            return None
        jobid = r.text
        for i in range(max_wait):
            time.sleep(10)
            status_url = f"https://www.ebi.ac.uk/Tools/services/rest/iprscan5/status/{jobid}"
            s = requests.get(status_url)
            st = s.text
            if st == "FINISHED":
                result_url = f"https://www.ebi.ac.uk/Tools/services/rest/iprscan5/result/{jobid}/json"
                r2 = requests.get(result_url)
                if r2.status_code == 200:
                    return r2.json()
                return None
            if st in ("ERROR", "FAILURE"):
                return None
        return None
    except Exception as e:
        log.warning(f"InterPro remote error: {e}")
        return None
def parse_interpro_local_json_to_map(json_path: str, valid_ids: set) -> Dict[str, Dict]:
    if not os.path.exists(json_path):
        return {}
    try:
        data = json.load(open(json_path, 'r', encoding='utf-8'))
    except Exception:
        return {}
    out: Dict[str, Dict] = {}
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        for item in data["results"]:
            sid = None
            xrefs = item.get("xref") or []
            if isinstance(xrefs, list):
                for x in xrefs:
                    if isinstance(x, dict):
                        for k in ("id", "name", "identifier", "accession"):
                            v = x.get(k)
                            if isinstance(v, str) and v in valid_ids:
                                sid = v
                                break
                    if sid:
                        break
            if not sid:
                for k in ("id", "identifier", "name", "accession"):
                    v = item.get(k)
                    if isinstance(v, str) and v in valid_ids:
                        sid = v
                        break
            if sid:
                out[sid] = item
        return out
    if isinstance(data, list):
        for item in data:
            sid = None
            xrefs = item.get("xref") or []
            if isinstance(xrefs, list):
                for x in xrefs:
                    if isinstance(x, dict):
                        for k in ("id", "name", "identifier", "accession"):
                            v = x.get(k)
                            if isinstance(v, str) and v in valid_ids:
                                sid = v
                                break
                    if sid:
                        break
            if not sid:
                for k in ("id", "identifier", "name", "accession"):
                    v = item.get(k)
                    if isinstance(v, str) and v in valid_ids:
                        sid = v
                        break
            if sid:
                out[sid] = item
        return out
    return {}
def run_hmmscan(fasta: str, domtblout: str, pfam_db: Optional[str] = None) -> Optional[str]:
    hmmscan_bin = which(CONFIG['tools']['hmmscan'])
    if not hmmscan_bin or not pfam_db or not os.path.exists(pfam_db):
        log.warning(f"hmmscan is unavailable or the Pfam database does not exist: bin={hmmscan_bin}, db={pfam_db}")
        return None
    cmd = [hmmscan_bin, '--domtblout', domtblout, pfam_db, fasta]
    rc, out, err = run_cmd(cmd, timeout=3600)
    if rc == 0 and os.path.exists(domtblout) and os.path.getsize(domtblout) > 0:
        log.info(f"hmmscan done: {domtblout}")
        return domtblout
    log.warning(f"hmmscan fail rc={rc}. stderr first 2KB:\n{(err or '')[:2048]}")
    return None


def parse_domtbl(domtbl_path: str) -> Dict[str, List[Dict]]:
    results: Dict[str, List[Dict]] = {}
    if not domtbl_path or not os.path.exists(domtbl_path):
        return results
    with open(domtbl_path) as fh:
        for line in fh:
            if line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 23:
                continue
            target = parts[0]
            query = parts[3]
            evalue = float(parts[6])
            score = float(parts[7])
            ali_from = int(parts[17])
            ali_to = int(parts[18])
            rec = {'hmm': target, 'query': query, 'evalue': evalue, 'score': score, 'ali_from': ali_from,
                   'ali_to': ali_to}
            results.setdefault(query, []).append(rec)
    return results
from sklearn.base import BaseEstimator


def apply_ml_model(features: Dict[str, float], model: BaseEstimator) -> float:
    if hasattr(model, 'feature_names_in_'):
        order = list(model.feature_names_in_)
    else:
        order = sorted(features.keys())
    x = [features.get(k, 0.0) for k in order]
    prob = model.predict_proba([x])[:, 1].item() if hasattr(model, 'predict_proba') else model.predict([x])[0]
    return float(prob * 100.0)
def parse_avg_plddt_from_pdb(pdb_path: str) -> Optional[float]:
    if not os.path.exists(pdb_path):
        return None
    vals = []
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith(('ATOM  ', 'HETATM')):
                try:
                    b = float(line[60:66].strip())
                    vals.append(b)
                except:
                    continue
    if not vals:
        return None
    return float(sum(vals) / len(vals))
def run_tmalign(pred_pdb: str, target_pdb: str) -> Optional[float]:
    tmalign_bin = which(CONFIG['tools']['tmalign'])
    if not tmalign_bin:
        return None
    rc, out, err = run_cmd([tmalign_bin, pred_pdb, target_pdb], timeout=300)
    if rc != 0:
        return None
    import re
    m = re.search(r"TM-score=.*?([0-9]+\.[0-9]+)", out)
    if not m:
        m2 = re.findall(r"TM-score=\s*([0-9]*\.[0-9]+)", out)
        if m2:
            return float(m2[0])
        return None
    return float(m.group(1))
def find_best_structural_match(pred_pdb: str, pdb_dir: str, workers: int = 4) -> Tuple[Optional[str], Optional[float]]:
    pdbs = [os.path.join(pdb_dir, f) for f in os.listdir(pdb_dir) if f.lower().endswith('.pdb')]
    if not pdbs:
        return None, None
    import concurrent.futures
    best = (None, 0.0)
    def _worker(tgt):
        tm = run_tmalign(pred_pdb, tgt)
        return tgt, tm
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in ex.map(_worker, pdbs):
            tgt, tm = fut
            if tm and tm > best[1]:
                best = (tgt, tm)
    return best
def fetch_uniprot_from_pdb_rcsb(pdb_id: str) -> Optional[str]:
    try:
        url = f"https://data.rcsb.org/rest/v1/core/struct/{pdb_id}"
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        refs = data.get('rcsb_entry_container_identifiers', {}).get('reference_sequence_identifiers', [])
        for ref in refs:
            if ref.get('database') and 'uniprot' in ref.get('database').lower():
                return ref.get('identifier')
        return None
    except Exception:
        return None
def compute_final_scores(scored_results: Dict[str, Dict], config: Dict = CONFIG) -> Dict[str, Dict]:
    weights = config['weights']
    for seq_id, rec in scored_results.items():
        blast_score = rec.get('blast_score', 0.0)
        domain_score = rec.get('domain_score', 0.0)
        go_score = rec.get('go_score', 0.0)
        struct_score = rec.get('structure', {}).get('structure_score', 0.0)
        base_final = (
                blast_score * weights['blast'] +
                domain_score * weights['domain'] +
                go_score * weights['go']
        )
        rec['final_score'] = base_final
        rec['final_score_with_structure'] = base_final + struct_score * weights['structure']
    return scored_results


def _structure_proxy_score(rec: dict) -> float:
    if not isinstance(rec, dict):
        return 0.0
    ds = float(rec.get('domain_score', 0.0) or 0.0)
    bs = float(rec.get('blast_score', 0.0) or 0.0)
    return max(0.0, min(100.0, ds * 0.7 + bs * 0.3))
def write_structure_outputs(scored_results: Dict[str, Dict], outdir: str, af3_pdb_dir: Optional[str] = None,
                           config: Optional[Dict] = None, logger: Optional[logging.Logger] = None):
    logx = logger or logging.getLogger("cotton_pipeline")
    thresholds = (config or {}).get('thresholds', {
        'plddt': 70.0, 'tm_strong': 0.70, 'tm_moderate': 0.50, 'pass_score': 60
    })

    Path(outdir).mkdir(parents=True, exist_ok=True)

    struct_map: Dict[str, Dict[str, Any]] = {}
    summary_rows: List[List[str]] = [[
        "seq_id", "source", "pdb_path", "plddt", "best_match", "tm_score", "structure_score"
    ]]

    total = len(scored_results)
    with_struct = 0
    with_score = 0
    strong_tm = 0
    moderate_tm = 0

    for sid, rec in scored_results.items():
        s = (rec or {}).get('structure') or {}
        pdb_path = s.get('pdb_path') or s.get('pdb_file')
        plddt = s.get('plddt') or s.get('avg_plddt')
        tm = s.get('tm_score')
        if tm is None:
            tm = s.get('best_tm') or s.get('best_tm_score')
        best_match = s.get('best_match') or s.get('best_match_pdb')
        sscore = s.get('structure_score', 0.0)
        source = ""
        if pdb_path and af3_pdb_dir:
            try:
                source = "AF3_local" if os.path.abspath(pdb_path).startswith(os.path.abspath(af3_pdb_dir)) else "External"
            except Exception:
                source = "AF3_local"
        if pdb_path:
            with_struct += 1
        if isinstance(sscore, (int, float)) and sscore > 0:
            with_score += 1
        if isinstance(tm, (int, float)):
            if tm >= thresholds.get('tm_strong', 0.70):
                strong_tm += 1
            elif tm >= thresholds.get('tm_moderate', 0.50):
                moderate_tm += 1

        struct_map[sid] = {
            "source": source or None,
            "pdb_path": pdb_path,
            "plddt": plddt,
            "best_match": best_match,
            "tm_score": tm,
            "structure_score": sscore,
        }

        summary_rows.append([
            sid,
            source,
            pdb_path or "",
            f"{plddt:.2f}" if isinstance(plddt, (int, float)) else "",
            best_match or "",
            f"{tm:.3f}" if isinstance(tm, (int, float)) else "",
            f"{sscore:.2f}" if isinstance(sscore, (int, float)) else "0.00",
        ])

    json_path = os.path.join(outdir, "structure_info.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(struct_map, f, ensure_ascii=False, indent=2)

    tsv_path = os.path.join(outdir, "structure_summary.tsv")
    with open(tsv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter='\t')
        for row in summary_rows:
            w.writerow(row)
    sum_path = os.path.join(outdir, "pipeline_summary.txt")
    with open(sum_path, "a", encoding="utf-8") as fh:
        fh.write("\n" + "=" * 80 + "\n")
        fh.write("Structural Enhancement Pipeline - Executive Summary(cotton.py out)\n")
        fh.write("=" * 80 + "\n\n")
        fh.write(f"recorded sequences {total}\n")
        fh.write(f"sequences with PDB {with_struct}\n")
        fh.write(f"sequences with structural scores {with_score}\n")
        fh.write(f"TM-score ≥ {thresholds.get('tm_strong',0.70):.2f}:{strong_tm}\n")
        fh.write(f"{thresholds.get('tm_moderate',0.50):.2f} ≤ TM-score < {thresholds.get('tm_strong',0.70):.2f}:{moderate_tm}\n")
        fh.write("\nScoring Rules (Structural Section):\n")
        fh.write(f"  pLDDT≥{thresholds.get('plddt',70.0):.0f}:pLDDT×0.5(limits 50);or×0.25\n")
        fh.write(f"  TM-score≥{thresholds.get('tm_strong',0.70):.2f}:+50;")
        fh.write(f"≥{thresholds.get('tm_moderate',0.50):.2f}:+TM×60;or +TM×30\n")

    logx.info(f"[structure] JSON: {json_path}")
    logx.info(f"[structure] TSV : {tsv_path}")
    logx.info(f"[structure] SUM : {sum_path}")

def export_results(scored_results: Dict[str, Dict], out_dir: str):
    ensure_dir(out_dir)
    tsv = os.path.join(out_dir, 'annotation_summary.tsv')

    headers = [
        'seq_id', 'blast_score', 'domain_score', 'go_score',
        'structure_score', 'final_score', 'final_score_with_structure', 'pass'
    ]

    with open(tsv, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh, delimiter='\t')
        writer.writerow(headers)
        for sid, rec in scored_results.items():
            row = [
                sid,
                rec.get('blast_score', 0.0),
                rec.get('domain_score', 0.0),
                rec.get('go_score', 0.0),
                rec.get('structure', {}).get('structure_score', 0.0),
                rec.get('final_score', 0.0),
                rec.get('final_score_with_structure', 0.0),
                rec.get('final_score_with_structure', rec.get('final_score', 0.0)) >= CONFIG['thresholds']['pass_score']
            ]
            writer.writerow(row)

    with open(os.path.join(out_dir, 'annotation_full.json'), 'w', encoding='utf-8') as fh:
        json.dump(scored_results, fh, indent=2, ensure_ascii=False)
    log.info(f"Exported results to {out_dir}")
    return tsv
def run_pipeline(
        input_fasta: str,
        output_dir: str = None,
        blast_db: str = "nr",
        pfam_db: str = None,
        af3_pdb_dir: str = None,
        pdb_lib_dir: str = None,
        email: str = "user@example.com",
        use_interpro: bool = True,
        use_hmmscan: bool = True,
        use_structure: bool = True,
        ml_model_path: str = None,
        config: Dict = None
) -> Dict[str, Dict]:
    if config is None:
        config = CONFIG

    if output_dir is None:
        output_dir = config['paths']['output_dir']

    ensure_dir(output_dir)

    log.info("=" * 80)
    log.info("Start the protein function prediction pipeline")
    log.info("=" * 80)

    # ============== step 1: sequence Cleaning ==============
    log.info("\n[step 1/6] Cleaning sequence...")
    cleaned_fasta = os.path.join(output_dir, "cleaned_sequences.fasta")
    clean_stats = clean_sequences(input_fasta, cleaned_fasta)
    valid_ids = set(clean_stats['md5_map'].keys())

    if clean_stats['kept'] == 0:
        log.error("No valid sequence! Exit.")
        return {}

    # ============== step 2: BLAST annotation ==============
    log.info("\n[step 2/6] Run the BLAST search...")
    blast_xml = os.path.join(output_dir, "blast_results.xml")
    try:
        run_blast_local(
            cleaned_fasta,
            blast_xml,
            db=blast_db,
            threads=config['runtime']['max_threads']
        )
        blast_results = parse_blast_xml(blast_xml)
        log.info(f"BLAST done, Obtained {len(blast_results)} sequences of results")
    except Exception as e:
        log.warning(f"BLAST fail: {e}")
        blast_results = {}

    # ============== step 3: Domain annotation ==============
    domain_results = {}

    # 3.1 InterProScan
    if use_interpro:
        log.info("\n[step 3a/6] running InterProScan...")
        interpro_json = os.path.join(output_dir, "interproscan_results.json")

        local_result = run_interpro_local(cleaned_fasta, interpro_json)

        if local_result:
            interpro_map = parse_interpro_local_json_to_map(local_result, valid_ids)
            log.info(f"InterProScan was completed locally and resulted in {len(interpro_map)} sequences.")
            domain_results.update(interpro_map)
        else:
            log.warning("InterProScan failed locally. Attempting remote connection...")
            records = list(SeqIO.parse(cleaned_fasta, "fasta"))[:5]
            for rec in records:
                remote_res = run_interpro_remote(str(rec.seq), rec.id, email)
                if remote_res:
                    domain_results[rec.id] = remote_res
                time.sleep(5)

    # 3.2 HMMER/Pfam
    if use_hmmscan and pfam_db:
        log.info("\n[step 3b/6] running HMMER scan...")
        domtbl = os.path.join(output_dir, "hmmscan_results.domtbl")
        hmmer_out = run_hmmscan(cleaned_fasta, domtbl, pfam_db)

        if hmmer_out:
            hmmer_results = parse_domtbl(hmmer_out)
            log.info(f"HMMER done, Obtained {len(hmmer_results)} sequences of results")

            for sid, domains in hmmer_results.items():
                if sid not in domain_results:
                    domain_results[sid] = {'matches': []}
                elif 'matches' not in domain_results[sid]:
                    domain_results[sid]['matches'] = []

                for d in domains:
                    domain_results[sid]['matches'].append({
                        'signature': {'accession': d['hmm']},
                        'evalue': d['evalue'],
                        'score': d['score']
                    })

    # ============== step 4: Initial scoring ==============
    log.info("\n[step 4/6] Calculate the initial annotation score...")
    scored_results = {}

    for seq_id in valid_ids:
        rec = {
            'seq_id': seq_id,
            'length': clean_stats['md5_map'][seq_id]['length'],
            'md5': clean_stats['md5_map'][seq_id]['md5']
        }

        blast_hits = blast_results.get(seq_id, {}).get('hits', [])
        if blast_hits:
            best_hit = blast_hits[0]
            identity = best_hit.get('identity', 0)
            coverage = best_hit.get('coverage', 0)
            evalue = best_hit.get('e_value', 1.0)

            blast_score = min(100, (identity * 0.5 + coverage * 0.3) * (1 - math.log10(evalue + 1e-100) / 100))
            rec['blast_score'] = max(0, blast_score)
            rec['blast_hits'] = blast_hits[:3]
        else:
            rec['blast_score'] = 0.0
            rec['blast_hits'] = []

        # Domain score
        domain_data = domain_results.get(seq_id, {})
        matches = domain_data.get('matches', [])

        if matches:
            significant_domains = [m for m in matches if m.get('evalue', 1) < 0.001]
            domain_score = min(100, len(significant_domains) * 20)
            rec['domain_score'] = domain_score
            rec['domains'] = matches[:5]
        else:
            rec['domain_score'] = 0.0
            rec['domains'] = []

        # GO score（from InterProScan）
        go_terms = []
        if 'goTerms' in domain_data:
            go_terms = domain_data['goTerms']
        elif isinstance(domain_data.get('matches'), list):
            for m in domain_data['matches']:
                if 'goTerms' in m:
                    go_terms.extend(m['goTerms'])

        go_score = min(100, len(set(gt.get('id', '') for gt in go_terms)) * 10)
        rec['go_score'] = go_score
        rec['go_terms'] = go_terms[:10]

        scored_results[seq_id] = rec

    # Structural verification
    if use_structure and af3_pdb_dir:
        log.info("\n[step 5/6] Carry out structural verification...")

        top_n = config['runtime'].get('top_n_for_structure', 50)
        sorted_ids = sorted(
            scored_results.keys(),
            key=lambda x: scored_results[x].get('blast_score', 0) + scored_results[x].get('domain_score', 0),
            reverse=True
        )[:top_n]

        log.info(f"Perform structural validation on the first {len(sorted_ids)} sequences...")

        for i, seq_id in enumerate(sorted_ids):
            if (i + 1) % 10 == 0:
                log.info(f"  processing: {i + 1}/{len(sorted_ids)}")

            rec = scored_results[seq_id]

            pdb_candidates = [
                os.path.join(af3_pdb_dir, f"{seq_id}.pdb"),
                os.path.join(af3_pdb_dir, f"{seq_id}_model.pdb"),
                os.path.join(af3_pdb_dir, f"{rec['md5']}.pdb"),
            ]

            pdb_path = None
            for candidate in pdb_candidates:
                if os.path.exists(candidate):
                    pdb_path = candidate
                    break

            if not pdb_path:
                rec['structure'] = {'structure_score': 0.0, 'reason': 'no_pdb'}
                continue

            avg_plddt = parse_avg_plddt_from_pdb(pdb_path)

            best_match = None
            best_tm = None

            if pdb_lib_dir and os.path.exists(pdb_lib_dir):
                best_match, best_tm = find_best_structural_match(
                    pdb_path,
                    pdb_lib_dir,
                    workers=config['runtime']['max_threads']
                )

            structure_score = 0.0

            # pLDDT
            if avg_plddt:
                plddt_threshold = config['thresholds']['plddt']
                if avg_plddt >= plddt_threshold:
                    structure_score += avg_plddt * 0.5
                else:
                    structure_score += avg_plddt * 0.25

            # TM-score
            if best_tm:
                tm_strong = config['thresholds']['tm_strong']
                tm_moderate = config['thresholds']['tm_moderate']

                if best_tm >= tm_strong:
                    structure_score += 50
                elif best_tm >= tm_moderate:
                    structure_score += best_tm * 60
                else:
                    structure_score += best_tm * 30

            rec['structure'] = {
                'pdb_path': pdb_path,
                'avg_plddt': avg_plddt,
                'best_match': best_match,
                'tm_score': best_tm,
                'structure_score': min(100, structure_score)
            }
    else:
        log.info("\n[step 5/6] Skip structure verification (not enabled or no PDB directory exists)")
        for seq_id, rec in scored_results.items():
            proxy_score = _structure_proxy_score(rec)
            rec['structure'] = {
                'structure_score': proxy_score,
                'reason': 'proxy_score'
            }
    # ============== step 6: Final score ==============
    log.info("\n[step 6/6] Calculate the final score...")
    ml_model = None
    if ml_model_path and os.path.exists(ml_model_path):
        try:
            import joblib
            ml_model = joblib.load(ml_model_path)
            log.info(f"Load the ML model: {ml_model_path}")
        except Exception as e:
            log.warning(f"Fail to load ML: {e}")

    if ml_model:
        for seq_id, rec in scored_results.items():
            features = {
                'blast_score': rec.get('blast_score', 0),
                'domain_score': rec.get('domain_score', 0),
                'go_score': rec.get('go_score', 0),
                'structure_score': rec.get('structure', {}).get('structure_score', 0),
                'seq_length': rec.get('length', 0)
            }
            ml_score = apply_ml_model(features, ml_model)
            rec['ml_score'] = ml_score

    scored_results = compute_final_scores(scored_results, config)
    pass_count = sum(
        1 for rec in scored_results.values()
        if rec.get('final_score_with_structure', rec.get('final_score', 0)) >= config['thresholds']['pass_score']
    )
    log.info("\n" + "=" * 80)
    log.info("DONE")
    log.info("=" * 80)
    log.info(f"Total seq: {len(scored_results)}")
    log.info(f"Passed the threshold (≥{config['thresholds']['pass_score']}): {pass_count}")
    log.info(f"Pass rate: {pass_count / len(scored_results) * 100:.1f}%")
    export_results(scored_results, output_dir)
    if use_structure:
        write_structure_outputs(
            scored_results,
            output_dir,
            af3_pdb_dir=af3_pdb_dir,
            config=config,
            logger=log
        )

    return scored_results
def main():
    parser = argparse.ArgumentParser(
        description="Protein Function Prediction Pipeline (BLAST + Domain + GO + Structure)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='path to input FASTA'
    )
    # 可选参数
    parser.add_argument(
        '-o', '--output',
        default=CONFIG['paths']['output_dir'],
        help=f'output dir'
    )
    parser.add_argument(
        '--blast-db',
        default='nr',
        help='BLAST database'
    )
    parser.add_argument(
        '--pfam-db',
        help='Pfam HMM database'
    )
    parser.add_argument(
        '--af3-pdb-dir',
        help='The directory of PDB files predicted by AlphaFold 3'
    )
    parser.add_argument(
        '--pdb-lib-dir',
        help='Local PDB library directory (for structure alignment)'
    )

    parser.add_argument(
        '--email',
        default='xx@xx',
        help='E-mail for remote services'
    )
    parser.add_argument(
        '--no-interpro',
        action='store_true',
        help='ban InterProScan'
    )
    parser.add_argument(
        '--no-hmmscan',
        action='store_true',
        help='ban HMMER/Pfam'
    )

    parser.add_argument(
        '--no-structure',
        action='store_true',
        help='ban structure validation'
    )
    parser.add_argument(
        '--ml-model',
        help='Machine learning model path (in joblib format)'
    )
    parser.add_argument(
        '--threads',
        type=int,
        default=CONFIG['runtime']['max_threads'],
        help=f'Maximum number of threads'
    )
    parser.add_argument(
        '--top-n-structure',
        type=int,
        default=CONFIG['runtime']['top_n_for_structure'],
        help=f'Perform structural validation on the first N sequences'
    )
    args = parser.parse_args()

    CONFIG['runtime']['max_threads'] = args.threads
    CONFIG['runtime']['top_n_for_structure'] = args.top_n_structure

    results = run_pipeline(
        input_fasta=args.input,
        output_dir=args.output,
        blast_db=args.blast_db,
        pfam_db=args.pfam_db,
        af3_pdb_dir=args.af3_pdb_dir,
        pdb_lib_dir=args.pdb_lib_dir,
        email=args.email,
        use_interpro=not args.no_interpro,
        use_hmmscan=not args.no_hmmscan,
        use_structure=not args.no_structure,
        ml_model_path=args.ml_model,
        config=CONFIG
    )
    log.info(f"\n FINISH, output to: {args.output}")

if __name__ == '__main__':
    main()