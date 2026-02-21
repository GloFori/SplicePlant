from __future__ import annotations
import os
import re
import json
import csv
import glob
import argparse
import hashlib
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import requests

try:
    from Bio import SeqIO
    from Bio.Align import PairwiseAligner
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord
    BIO_AVAILABLE = True
except ImportError:
    BIO_AVAILABLE = False
    print("Warning: BioPython is not installed. Some isoform analysis functions will be restricted.")

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


def get_first_record(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {}
    return records[0]


def tsv_head_to_md(path: str, max_rows: int = 20, max_cols: int = 16) -> str:
    p = Path(path)
    if not p.exists():
        return "_(empty)_"
    try:
        rows: List[List[str]] = []
        with p.open("r", encoding="utf-8", errors="ignore", newline="") as fh:
            reader = csv.reader(fh, delimiter="\t")
            for i, row in enumerate(reader):
                row = [
                    ("" if c is None else str(c).replace("\x00", "").replace("|", "\\|")
                     .replace("\r", " ").replace("\n", " "))
                    for c in row
                ]
                rows.append(row)
                if i >= max_rows:
                    break
        if not rows:
            return "_(empty)_"
        ncol = min(
            len(rows[0]) if rows and rows[0] else max((len(r) for r in rows), default=0),
            max_cols
        )
        if ncol <= 0:
            return "_(empty)_"
        header = rows[0] if rows else []
        header = (header[:ncol] + [""] * max(0, ncol - len(header))) if header else [""] * ncol

        seen = {}
        clean_header: List[str] = []
        for i, h in enumerate(header):
            h = (h or "").strip() or f"col_{i+1}"
            if h in seen:
                seen[h] += 1
                h = f"{h}_{seen[h]}"
            else:
                seen[h] = 0
            clean_header.append(h)

        body = []
        for r in rows[1:]:
            r = r[:ncol] + [""] * max(0, ncol - len(r))
            body.append(r)

        out: List[str] = []
        out.append("| " + " | ".join(clean_header) + " |")
        out.append("| " + " | ".join(["---"] * ncol) + " |")
        for r in body:
            out.append("| " + " | ".join(r) + " |")
        return "\n".join(out)
    except Exception:
        return "_(empty)_"


def text_head(path: str, max_chars: int = 4000) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    try:
        return p.read_text(encoding="utf-8", errors="ignore")[:max_chars]
    except Exception:
        return ""


def json_load(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def find_first(patterns: List[str]) -> Optional[str]:
    for pat in patterns:
        m = glob.glob(pat)
        if m:
            return sorted(m)[0]
    return None


def normalize_interpro_ids(md: str) -> str:
    # 保持你原来的写法，不动
    md = re.sub(r"\\bIP\\.R(\\d{5})\\b", r"IPR\\1", md)
    md = re.sub(r"\\bipr(\\d{5})\\b", lambda m: f"IPR{m.group(1)}", md, flags=re.IGNORECASE)
    return md


def parse_fasta_sequences(fasta_path: str) -> List[Dict[str, str]]:
    sequences = []
    if not fasta_path or not Path(fasta_path).exists():
        return sequences
    
    try:
        with open(fasta_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        current_seq_id = ""
        current_seq = ""
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_seq_id and current_seq:
                    sequences.append({
                        "id": current_seq_id,
                        "sequence": current_seq,
                        "header": line[1:]
                    })
                current_seq_id = line[1:].split()[0]
                current_seq = ""
            else:
                current_seq += line

        if current_seq_id and current_seq:
            sequences.append({
                "id": current_seq_id,
                "sequence": current_seq,
                "header": line[1:] if line.startswith(">") else current_seq_id
            })
    except Exception as e:
        print(f"Error occurred while parsing the FASTA file: {e}")
    
    return sequences


# ========= Isoform =========
def detect_isoforms(seq1_id: str, seq2_id: str, seq1_sequence: str, seq2_sequence: str) -> Dict[str, Any]:

    result = {
        "is_isoforms": False,
        "base_gene_id": "",
        "similarity_percentage": 0.0,
        "length_difference": 0,
        "detection_method": "none",
        "isoform_pattern": "",
        "confidence": "low"
    }

    isoform_patterns = [
        (r'(.*?)[\._](\d+)$', "numeric_suffix"),  # .1, _1
        (r'(.*?)-(\d+)$', "dash_suffix"),         # -1
        (r'(.*?)_isoform[_-]?(\d+)$', "isoform_suffix"),  # _isoform_1
        (r'(.*?)([a-zA-Z])(\d*)$', "letter_suffix"),  # A, B, C1
        (r'(.*?)_variant(\d+)$', "variant_suffix"),  # _variant1
    ]
    
    base_id = None
    detected_pattern = ""
    
    for pattern, pattern_name in isoform_patterns:
        match1 = re.match(pattern, seq1_id)
        match2 = re.match(pattern, seq2_id)
        if match1 and match2:
            base1 = match1.group(1)
            base2 = match2.group(1)
            if base1 == base2:
                result["is_isoforms"] = True
                result["base_gene_id"] = base1
                result["isoform_pattern"] = pattern_name
                result["detection_method"] = "id_pattern"
                result["confidence"] = "high"
                break

    if not result["is_isoforms"] and BIO_AVAILABLE:
        try:
            aligner = PairwiseAligner()
            aligner.mode = 'global'
            aligner.match_score = 2
            aligner.mismatch_score = -1
            aligner.open_gap_score = -5
            aligner.extend_gap_score = -2
            
            alignments = aligner.align(seq1_sequence, seq2_sequence)
            if alignments:
                alignment = alignments[0]
                aligned_str = str(alignment)
                lines = aligned_str.split('\n')
                
                if len(lines) >= 3:
                    aligned_seq1 = lines[0]
                    aligned_seq2 = lines[2]

                    matches = sum(1 for a, b in zip(aligned_seq1, aligned_seq2) if a == b and a != '-')
                    total_aligned = len([c for c in aligned_seq1 if c != '-'])
                    
                    similarity = matches / total_aligned if total_aligned > 0 else 0
                    
                    if similarity > 0.7:
                        result["is_isoforms"] = True
                        result["base_gene_id"] = seq1_id.split('_')[0] if '_' in seq1_id else seq1_id
                        result["similarity_percentage"] = similarity * 100
                        result["detection_method"] = "sequence_similarity"
                        result["confidence"] = "medium"
        except Exception as e:
            print(f"Sequence alignment failed: {e}")

    result["length_difference"] = abs(len(seq1_sequence) - len(seq2_sequence))
    
    return result


def analyze_isoform_differences(seq1_id: str, seq2_id: str, 
                               seq1_sequence: str, seq2_sequence: str,
                               seq1_ann: Dict, seq2_ann: Dict) -> Dict[str, Any]:

    analysis = {
        "sequence_comparison": {},
        "structural_impact": {},
        "functional_impact": {},
        "splicing_events": []
    }

    if BIO_AVAILABLE:
        try:
            aligner = PairwiseAligner()
            aligner.mode = 'global'
            alignments = aligner.align(seq1_sequence, seq2_sequence)
            
            if alignments:
                alignment = alignments[0]
                aligned_str = str(alignment)
                lines = aligned_str.split('\n')
                
                if len(lines) >= 3:
                    aligned_seq1 = lines[0]
                    alignment_line = lines[1]
                    aligned_seq2 = lines[2]

                    differences = []
                    current_diff = None
                    
                    for i, (c1, c2) in enumerate(zip(aligned_seq1, aligned_seq2)):
                        if c1 != c2 and c1 != '-' and c2 != '-':
                            if current_diff and current_diff['type'] == 'mismatch':
                                current_diff['end'] = i
                            else:
                                if current_diff:
                                    differences.append(current_diff)
                                current_diff = {
                                    'type': 'mismatch',
                                    'start': i,
                                    'end': i,
                                    'seq1_aa': c1,
                                    'seq2_aa': c2
                                }
                        elif c1 == '-':
                            if current_diff and current_diff['type'] == 'deletion_in_seq1':
                                current_diff['end'] = i
                            else:
                                if current_diff:
                                    differences.append(current_diff)
                                current_diff = {
                                    'type': 'deletion_in_seq1',
                                    'start': i,
                                    'end': i,
                                    'seq2_aa': c2
                                }
                        elif c2 == '-':
                            if current_diff and current_diff['type'] == 'deletion_in_seq2':
                                current_diff['end'] = i
                            else:
                                if current_diff:
                                    differences.append(current_diff)
                                current_diff = {
                                    'type': 'deletion_in_seq2',
                                    'start': i,
                                    'end': i,
                                    'seq1_aa': c1
                                }
                        else:
                            if current_diff:
                                differences.append(current_diff)
                                current_diff = None
                    
                    if current_diff:
                        differences.append(current_diff)
                    
                    analysis["sequence_comparison"] = {
                        "alignment_score": alignments[0].score,
                        "identity_percentage": sum(1 for a, b in zip(aligned_seq1, aligned_seq2) 
                                                 if a == b and a != '-') / len([c for c in aligned_seq1 if c != '-']) * 100,
                        "gap_percentage": (aligned_seq1.count('-') + aligned_seq2.count('-')) / (len(aligned_seq1) * 2) * 100,
                        "differences": differences[:10],
                        "total_differences": len(differences)
                    }
        except Exception as e:
            print(f"Detailed sequence alignment analysis failed: {e}")

    dom1 = seq1_ann.get('interpro_domains', []) or []
    dom2 = seq2_ann.get('interpro_domains', []) or []
    
    dom1_ids = {d.get('accession', '') for d in dom1}
    dom2_ids = {d.get('accession', '') for d in dom2}
    
    analysis["structural_impact"]["domain_differences"] = {
        "shared_domains": list(dom1_ids & dom2_ids),
        "unique_to_seq1": list(dom1_ids - dom2_ids),
        "unique_to_seq2": list(dom2_ids - dom1_ids)
    }

    go1 = seq1_ann.get('go_terms', []) or []
    go2 = seq2_ann.get('go_terms', []) or []
    
    go1_terms = {term.get('term', '') for term in go1}
    go2_terms = {term.get('term', '') for term in go2}
    
    analysis["functional_impact"]["go_differences"] = {
        "shared_go_terms": list(go1_terms & go2_terms),
        "unique_to_seq1": list(go1_terms - go2_terms),
        "unique_to_seq2": list(go2_terms - go1_terms)
    }

    if analysis.get("sequence_comparison", {}).get("differences"):
        differences = analysis["sequence_comparison"]["differences"]
        splicing_events = []

        for diff in differences:
            if diff['type'] in ['deletion_in_seq1', 'deletion_in_seq2']:
                length = diff['end'] - diff['start'] + 1
                if length >= 10:
                    splicing_events.append({
                        "type": "exon_skipping",
                        "position": f"{diff['start']}-{diff['end']}",
                        "length": length,
                        "description": f"Sequence deletion length {length}aa, possibly due to exon skipping"
                    })
                else:
                    splicing_events.append({
                        "type": "alternative_splice_site",
                        "position": f"{diff['start']}-{diff['end']}",
                        "length": length,
                        "description": f"Short sequence deletion, possibly a site for alternative splicing"
                    })
            elif diff['type'] == 'mismatch':
                if 'seq1_aa' in diff and 'seq2_aa' in diff:
                    if diff['seq1_aa'] == '*' or diff['seq2_aa'] == '*':
                        splicing_events.append({
                            "type": "alternative_polyadenylation",
                            "position": f"{diff['start']}",
                            "description": "The difference in termination codons may be due to variable polyadenylation."
                        })
        
        analysis["splicing_events"] = splicing_events
    
    return analysis


def generate_isoform_report(seq1_info: Dict, seq2_info: Dict, 
                           isoform_detection: Dict, 
                           isoform_analysis: Dict) -> str:
    """
    isoform report
    """
    report = []
    
    report.append("# Isoform report")
    report.append("")

    # 1. Basic Information
    report.append("## 1. Basic Information")
    report.append(f"- Sequence 1 ID: {seq1_info.get('id', 'Unknown')}")
    report.append(f"- Sequence 2 ID: {seq2_info.get('id', 'Unknown')}")
    report.append(
        f"- Length Difference: {abs(len(seq1_info.get('sequence', '')) - len(seq2_info.get('sequence', '')))} amino acids")
    report.append(f"- Isoform Status: {'Yes' if isoform_detection.get('is_isoforms') else 'No'}")
    report.append(f"- Parent Gene ID: {isoform_detection.get('base_gene_id', 'Unknown')}")
    report.append(f"- Detection Method: {isoform_detection.get('detection_method', 'Unknown')}")
    report.append(f"- Confidence Score: {isoform_detection.get('confidence', 'Unknown')}")
    report.append("")

    if isoform_detection.get('is_isoforms'):
        # 2. Sequence Alignment Analysis
        seq_comp = isoform_analysis.get('sequence_comparison', {})
        report.append("## 2. Sequence Alignment Analysis")
        if seq_comp:
            report.append(f"- Sequence Identity: {seq_comp.get('identity_percentage', 0):.1f}%")
            report.append(f"- Alignment Score: {seq_comp.get('alignment_score', 0):.1f}")
            report.append(f"- Total Differences: {seq_comp.get('total_differences', 0)}")
        else:
            report.append("- Alignment information missing")
        report.append("")

        # 3. Domain Differences
        dom_diff = isoform_analysis.get('structural_impact', {}).get('domain_differences', {})
        report.append("## 3. Domain Differences")
        report.append(f"- Shared Domains: {len(dom_diff.get('shared_domains', []))}")
        report.append(f"- Unique to Seq 1: {len(dom_diff.get('unique_to_seq1', []))}")
        report.append(f"- Unique to Seq 2: {len(dom_diff.get('unique_to_seq2', []))}")
        if dom_diff.get('unique_to_seq1'):
            report.append(f"  - Unique Domains (Seq 1): {', '.join(dom_diff['unique_to_seq1'][:5])}")
        if dom_diff.get('unique_to_seq2'):
            report.append(f"  - Unique Domains (Seq 2): {', '.join(dom_diff['unique_to_seq2'][:5])}")
        report.append("")

        # 4. Functional Differences
        go_diff = isoform_analysis.get('functional_impact', {}).get('go_differences', {})
        report.append("## 4. GO Functional Differences")
        report.append(f"- Shared GO Terms: {len(go_diff.get('shared_go_terms', []))}")
        report.append(f"- Unique to Seq 1: {len(go_diff.get('unique_to_seq1', []))}")
        report.append(f"- Unique to Seq 2: {len(go_diff.get('unique_to_seq2', []))}")
        report.append("")

        # 5. Predicted Splicing Events
        splicing_events = isoform_analysis.get('splicing_events', [])
        report.append("## 5. Predicted Splicing Events")
        if splicing_events:
            for i, event in enumerate(splicing_events, 1):
                report.append(f"{i}. **{event.get('type', 'Unknown')}**")
                report.append(f"   - Position: {event.get('position', 'Unknown')}")
                report.append(f"   - Description: {event.get('description', '')}")
        else:
            report.append("- No distinct splicing patterns detected")
        report.append("")

        # 6. Causal Chain Analysis Framework
        report.append("## 6. Causal Chain Analysis Framework")
        report.append("Based on the analysis, the potential molecular mechanism is as follows:")
        report.append("")
        report.append("```")
        report.append(
            "Transcription → Alternative Splicing → Sequence Variation → Structural Change → Functional Divergence → Phenotypic Variation")
        report.append("")

        # Constructing specific causal chain
        causal_chain = []

        # Splicing event impact
        if splicing_events:
            for event in splicing_events[:2]:  # Take the first two major events
                causal_chain.append(f"- {event['type']} at {event['position']}")

        # Domain impact
        if dom_diff.get('unique_to_seq1') or dom_diff.get('unique_to_seq2'):
            causal_chain.append("- Differences in domain composition")

        # Functional impact
        if go_diff.get('unique_to_seq1') or go_diff.get('unique_to_seq2'):
            causal_chain.append("- Functional specificity divergence")

        if causal_chain:
            report.append("Specific nodes:")
            for item in causal_chain:
                report.append(f"  {item}")

        report.append("```")
        report.append("")

        # 7. Experimental Validation Recommendations
        report.append("## 7. Experimental Validation Recommendations")
        report.append("1. **RT-PCR**: Design specific primers spanning the divergent regions.")
        report.append("2. **Western Blot**: Verify protein expression levels for both isoforms.")
        report.append("3. **Subcellular Localization**: Observe localization differences via GFP fusion proteins.")
        report.append("4. **Enzymatic Assay**: If predicted as an enzyme, measure catalytic activity differences.")
        report.append("5. **Interaction Analysis**: Use Co-IP to validate differences in interaction partners.")
    
    return "\n".join(report)


def collect_context(annotation_dir: str,
                    with_struct_dir: str,
                    final_dir: str,
                    structsearch_dir: Optional[str] = None,
                    fasta_path: Optional[str] = None,
                    heavy: bool = False) -> Dict[str, Any]:

    structsearch_dir = structsearch_dir or final_dir

    ctx: Dict[str, Any] = {}

    ann_json = find_first([f"{annotation_dir}/annotation_full.json", f"{annotation_dir}/*full.json"])
    ann_tsv = find_first([f"{annotation_dir}/annotation_summary.tsv", f"{annotation_dir}/*summary.tsv"])
    ctx["ann_json_path"] = ann_json or ""
    ctx["ann_tsv_path"] = ann_tsv or ""
    ctx["ann_json_head"] = text_head(ann_json, max_chars=20000 if heavy else 4000) if ann_json else ""
    ctx["ann_tsv_head_md"] = tsv_head_to_md(
        ann_tsv,
        max_rows=200 if heavy else 20,
        max_cols=32 if heavy else 16
    ) if ann_tsv else "_(missing)_"

    w_json = find_first([
        f"{with_struct_dir}/annotation_full_with_structure.json",
        f"{with_struct_dir}/*full_with_structure.json"
    ])
    w_tsv = find_first([
        f"{with_struct_dir}/annotation_summary_with_structure.tsv",
        f"{with_struct_dir}/*summary_with_structure.tsv"
    ])
    ctx["with_json_path"] = w_json or ""
    ctx["with_tsv_path"] = w_tsv or ""
    ctx["with_json_head"] = text_head(w_json, max_chars=20000 if heavy else 4000) if w_json else ""
    ctx["with_tsv_head_md"] = tsv_head_to_md(
        w_tsv,
        max_rows=200 if heavy else 20,
        max_cols=32 if heavy else 16
    ) if w_tsv else "_(missing)_"

    final_txt = find_first([
        f"{final_dir}/final_report_with_structure.txt",
        f"{final_dir}/*final_report*txt",
        f"{final_dir}/final_report.json",
        f"{final_dir}/*final*json"
    ])
    ranked_tsv = find_first([
        f"{final_dir}/annotation_ranked_with_structure.tsv",
        f"{final_dir}/*ranked*with_structure.tsv",
        f"{final_dir}/annotation_ranked.tsv",
        f"{final_dir}/*ranked*.tsv"
    ])
    ctx["final_txt_path"] = final_txt or ""
    ctx["final_txt_head"] = text_head(final_txt, max_chars=20000 if heavy else 4000) if final_txt else ""
    ctx["final_ranked_path"] = ranked_tsv or ""
    ctx["final_ranked_head_md"] = tsv_head_to_md(
        ranked_tsv,
        max_rows=200 if heavy else 20,
        max_cols=32 if heavy else 16
    ) if ranked_tsv else "_(missing)_"

    fs_m8 = find_first([f"{structsearch_dir}/foldseek.m8", f"{structsearch_dir}/*foldseek*.m8"])
    tm_tsv = find_first([f"{structsearch_dir}/tm_results.tsv", f"{structsearch_dir}/*tm*.tsv"])
    ctx["foldseek_m8_path"] = fs_m8 or ""
    if fs_m8 and Path(fs_m8).exists():
        try:
            head_lines = 200 if heavy else 50
            ctx["foldseek_m8_head"] = "\n".join(
                Path(fs_m8).read_text(encoding="utf-8", errors="ignore").splitlines()[:head_lines]
            )
        except Exception:
            ctx["foldseek_m8_head"] = ""
    else:
        ctx["foldseek_m8_head"] = ""
    ctx["tm_tsv_path"] = tm_tsv or ""
    ctx["tm_tsv_head_md"] = tsv_head_to_md(
        tm_tsv,
        max_rows=200 if heavy else 20,
        max_cols=32 if heavy else 16
    ) if tm_tsv else "_(missing)_"


    ctx["fasta_sequences"] = []
    ctx["fasta_head"] = ""
    if fasta_path:
        sequences = parse_fasta_sequences(fasta_path)
        ctx["fasta_sequences"] = sequences

        fasta_lines = []
        max_lines_per_seq = 300 if heavy else 100
        for i, seq in enumerate(sequences):
            fasta_lines.append(f">seq{i+1} ID: {seq['id']}")
            fasta_lines.append(f">seq{i+1} whole Header: {seq['header']}")
            seq_chunks = [seq['sequence'][j:j+80] for j in range(0, len(seq['sequence']), 80)]
            fasta_lines.extend(seq_chunks[:max_lines_per_seq])
            if len(seq_chunks) > max_lines_per_seq:
                fasta_lines.append(f"... (The remaining part of the sequence {i + 1} has been truncated)")
            fasta_lines.append("")
        
        ctx["fasta_head"] = "\n".join(fasta_lines)

        ctx["isoform_analysis"] = {}

        if len(sequences) >= 2:
            seq1 = sequences[0]
            seq2 = sequences[1]
            
            print(f"[*] Conduct Isoform analysis: {seq1['id']} vs {seq2['id']}")

            isoform_detection = detect_isoforms(
                seq1["id"], seq2["id"],
                seq1["sequence"], seq2["sequence"]
            )
            ctx["isoform_analysis"]["detection"] = isoform_detection

            if isoform_detection.get("is_isoforms"):
                print(f"[*] Isoforms were detected and a detailed analysis was conducted...")
                seq1_ann = {}
                seq2_ann = {}

                if ctx["ann_json_head"]:
                    try:
                        ann_data = json.loads(ctx["ann_json_head"])
                        if isinstance(ann_data, list):
                            for record in ann_data:
                                if isinstance(record, dict):
                                    seq_id = record.get("seq_id", "")
                                    if seq_id == seq1["id"]:
                                        seq1_ann = record
                                    elif seq_id == seq2["id"]:
                                        seq2_ann = record
                    except Exception as e:
                        print(f"[!] Failed to parse the annotation data: {e}")

                isoform_details = analyze_isoform_differences(
                    seq1["id"], seq2["id"],
                    seq1["sequence"], seq2["sequence"],
                    seq1_ann, seq2_ann
                )
                ctx["isoform_analysis"]["details"] = isoform_details

                isoform_report = generate_isoform_report(
                    seq1, seq2, isoform_detection, isoform_details
                )
                ctx["isoform_analysis"]["report"] = isoform_report
                print(f"[*] Isoform report done")
    
    return ctx


FINAL_TSV_HEADER = [
    "seq_id", "protein_id", "length",
    "avg_pLDDT", "best_tm_score",
    "num_domains", "num_go_terms", "num_blast_hits", "description"
]


def _to_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def _to_int(x, default=None):
    try:
        return int(float(x))
    except Exception:
        return default


def load_final_records_unified(final_dir: str) -> List[Dict[str, Any]]:
    d = Path(final_dir)

    for pat in [
        "annotation_ranked_with_structure.tsv",
        "annotation_ranked.tsv",
        "final_report.tsv",
        "*ranked*with_structure.tsv",
        "*ranked*.tsv",
        "*final*tsv"
    ]:
        for p in sorted(d.glob(pat)):
            recs: List[Dict[str, Any]] = []
            with p.open("r", encoding="utf-8", errors="ignore", newline="") as fh:
                reader = csv.DictReader(fh, delimiter="\t")
                for r in reader:
                    rec = {k: r.get(k, "") for k in FINAL_TSV_HEADER}
                    for k in ["length", "num_domains", "num_go_terms", "num_blast_hits"]:
                        rec[k] = _to_int(rec.get(k), 0)
                    for k in [
                        "final_score",
                        "final_score_with_structure",
                        "domain_score",
                        "go_score",
                        "blast_score",
                        "avg_pLDDT",
                        "best_tm_score",
                    ]:
                        rec[k] = _to_float(rec.get(k), None)
                    recs.append(rec)
            if recs:
                return recs

    json_path = None
    for pat in ["final_report.json", "*final*json"]:
        hits = sorted(d.glob(pat))
        if hits:
            json_path = hits[0]
            break
    if json_path and json_path.exists():
        data = json.loads(json_path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "records" in data:
            data = data["records"]
        recs: List[Dict[str, Any]] = []
        for x in data if isinstance(data, list) else []:
            st = x.get("structure", {}) or {}
            recs.append({
                "seq_id": x.get("seq_id", ""),
                "protein_id": x.get("protein_id", ""),
                "length": _to_int(x.get("length"), 0),
                "final_score": _to_float(x.get("final_score"), None),
                "final_score_with_structure": _to_float(
                    x.get("final_score_with_structure", x.get("final_score")), None
                ),
                "domain_score": _to_float(x.get("domain_score"), None),
                "go_score": _to_float(x.get("go_score"), None),
                "blast_score": _to_float(x.get("blast", None), None),
                "avg_pLDDT": _to_float(st.get("avg_pLDDT"), None),
                "best_tm_score": _to_float(st.get("best_tm_score"), None),
                "evidence_rating": st.get("evidence_rating", ""),
                "num_domains": len(x.get("interpro_domains", []) or []),
                "num_go_terms": len(x.get("go_terms", []) or []),
                "num_blast_hits": len(x.get("blast_hits", []) or []),
                "description": x.get("description", ""),
            })
        return recs
    return []


def pick_focus_record(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {}
    return max(
        records,
        key=lambda r: (
            r.get("final_score_with_structure") is not None,
            r.get("final_score_with_structure", float("-inf"))
        )
    )


def get_focus_text(rec: Dict[str, Any]) -> str:
    return (rec or {}).get("description", "").strip()


def build_candidates_table_md(records: List[Dict[str, Any]], top_k: int = 12) -> str:
    if not records:
        return "_(no candidates)_"

    head = ["ID", "Functional Assumption", "avg_pLDDT", "best_tm", "Annotation evidence"]
    out = [
        "| " + " | ".join(head) + " |",
        "| " + " | ".join(["---"] * len(head)) + " |"
    ]

    for r in records[:top_k]:
        z = []
        if r.get("num_domains"):
            z.append(f"IPR×{r['num_domains']}")
        if r.get("num_go_terms"):
            z.append(f"GO×{r['num_go_terms']}")
        if r.get("num_blast_hits"):
            z.append(f"BLAST×{r['num_blast_hits']}")

        row = [
            str(r.get("seq_id", "")),
            (r.get("description", "") or "")[:60],
            str(r.get("avg_pLDDT", "")),
            str(r.get("best_tm_score", "")),
            ", ".join(z) if z else "(empty)"
        ]
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def call_ollama_chat(model: str, messages: List[Dict[str, str]], temperature: float = 0.5,
                     max_tokens: int = 3000, num_ctx: int = 8192, base_url: str = None) -> str:
    base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    url = f"{base_url}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens, "num_ctx": num_ctx}
    }
    try:
        r = requests.post(url, json=payload, timeout=60000)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict):
            msg = data.get("message") or {}
            return msg.get("content", "")
        return ""
    except requests.RequestException as e:
        return f"Error calling Ollama: {e}"


def call_dashscope_chat(model_name: str, api_key: str, base_url: str, system_instruction: str,
                        user_prompt: str, temperature: float = 0.5, max_tokens: int = 3000) -> str:
    if OpenAI is None:
        return "Error calling DashScope API: openai Not installed (for compatibility interface)"
    try:
        api_key = api_key or os.getenv("DASHSCOPE_API_KEY", "")
        if not api_key:
            return "Error calling DashScope API: Missing API KEY (Please set --api_key or environment variable DASHSCOPE_API_KEY)"
        client = OpenAI(api_key=api_key, base_url=base_url)
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False
        )
        if hasattr(resp, "choices") and resp.choices:
            msg = resp.choices[0].message
            if msg and hasattr(msg, "content"):
                return msg.content or ""
        return ""
    except Exception as e:
        return f"Error calling DashScope API: {e}"


def _shared_sources_block(ctx: Dict[str, Any]) -> str:
    isoform_block = ""
    if ctx.get("isoform_analysis", {}).get("detection", {}).get("is_isoforms"):
        iso_det = ctx["isoform_analysis"]["detection"]
        iso_report = ctx["isoform_analysis"].get("report", "")
        # Part 1: Isoform Analysis Block
        isoform_block = f"""
        [ Isoform Analysis Results ]
        Detection Result: Confirmed as different splicing variants of the same gene
        Base Gene: {iso_det.get('base_gene_id', 'Unknown')}
        Similarity: {iso_det.get('similarity_percentage', 0):.1f}%
        Detailed Report:
        {iso_report[:2000] if iso_report else 'No detailed report available'}
        """

        return f"""
        [ Text/JSON Report ]
        Path: {ctx.get("final_txt_path", "")}
        Content Snippet:
        {ctx.get("final_txt_head", "")}

        [ Ranked Table (with structure) ]
        Path: {ctx.get("final_ranked_path", "")}
        Preview (Header & Top Rows):
        {ctx.get("final_ranked_head_md", "")}

        [ With-Structure TSV ]
        Path: {ctx.get("with_tsv_path", "")}
        Preview:
        {ctx.get("with_tsv_head_md", "")}

        [ With-Structure JSON ]
        Path: {ctx.get("with_json_path", "")}
        Snippet:
        {ctx.get("with_json_head", "")}

        [ Annotation Summary TSV ]
        Path: {ctx.get("ann_tsv_path", "")}
        Preview:
        {ctx.get("ann_tsv_head_md", "")}

        [ Annotation Full JSON ]
        Path: {ctx.get("ann_json_path", "")}
        Snippet:
        {ctx.get("ann_json_head", "")}

        [ Foldseek .m8 (top lines) ]
        Path: {ctx.get("foldseek_m8_path", "")}
        Snippet:
        {ctx.get("foldseek_m8_head", "")}

        [ TM results TSV ]
        Path: {ctx.get("tm_tsv_path", "")}
        Preview:
        {ctx.get("tm_tsv_head_md", "")}

        [ FASTA Sequence Info ]
        {ctx.get("fasta_head", "")}
        {isoform_block}
        """

    # Part 2: Rigorous Prompt Construction
    def build_prompt_rigorous(ctx: Dict[str, Any], top_k: int = 12) -> str:
        return f"""
        You are a **Rigorous Bioinformatics Analyst**. Integrate multi-source evidence into a **traceable and auditable** report. Do not hallucinate; if data is missing, mark as "Missing/Not Provided".

        **Key Hard Requirements:**
        - **The first sentence of Top-1 MUST follow this template**: "The function of this protein is [Function], it influences the phenotype via [Mechanism], and the resulting impact is [Outcome]."
        - If avg_pLDDT < 70 or best_tm < 0.5, "High Confidence" must NOT be granted; explain the reasons and remedies in the "Uncertainty" section.
        - Provide ≥2 lines of evidence, with ≥1 being annotation evidence (InterPro/GO/BLAST), using the format: [Path -> Field = Value].
        - Foldseek results should only be included if the m8 snippet is non-empty and contains hits.

        Available Materials:
        {_shared_sources_block(ctx)}

        # Required Output Structure:
        # Function and Phenotypic Impact Conclusion (Top-1)
        # Executive Summary
        # Candidate Overview Table (Top {top_k})
        # Detailed Evidence per Candidate
        # Methods and Key Parameters
        # Reproducibility and Data Lineage
        # Validation Checklist (Self-Check)
        # Machine-Readable YAML Summary
        """


def build_prompt_rigorous(ctx: Dict[str, Any], top_k: int = 12) -> str:
    return f"""
You are a **Rigorous Bioinformatics Analyst**. Please integrate multi-source evidence into a **traceable and auditable** report. Fabrication is strictly prohibited; if data is missing, mark as "Missing/Not Provided".
**Key Hard Requirements:**
- **The first sentence of Top-1 MUST follow this template**: "The function of this protein is __, it influences the phenotype via __, and the resulting impact is __."
- If avg_pLDDT < 70 or best_tm < 0.5, "High Confidence" must NOT be granted; explain the reasons and remedies in the "Uncertainty" section.
- Provide ≥2 lines of evidence, with ≥1 being annotation evidence (InterPro/GO/BLAST), using the format: [Path -> Field = Value].
- Foldseek results should only be included if the m8 snippet is non-empty and contains hits.
Available materials:
{_shared_sources_block(ctx)}
# Required Output Structure
# Function and Phenotypic Impact (Top-1)
# Executive Summary
# Candidate Overview Table (Top {top_k})
# Per-Candidate Argumentation
# Methods and Key Parameters
# Reproducibility and Data Lineage
# Self-Check List
# Machine-Readable YAML Summary
"""


def build_prompt_description_verify(ctx: Dict[str, Any], trait: Optional[str], focus_rec: Dict[str, Any],
                                    candidates_table_md: str, top_k: int = 12) -> str:
    """
    Verification Mode v4.0: Integrated isoform causal chain analysis
    """
    focus_text = (focus_rec or {}).get("description", "").strip() or "(No detailed description provided)"
    focus_id = focus_rec.get("seq_id", "")
    plddt = focus_rec.get("avg_pLDDT")
    tm = focus_rec.get("best_tm_score")
    rating = focus_rec.get("evidence_rating", "")

    # Get FASTA sequence info
    fasta_sequences = ctx.get("fasta_sequences", [])
    seq_info = ""

    if len(fasta_sequences) >= 2:
        seq1 = fasta_sequences[0]
        seq2 = fasta_sequences[1]
        seq_info = f"""
[ Protein Sequence Information ]
Sequence 1:
- ID: {seq1.get('id', 'Unknown')}
- Full Header: {seq1.get('header', 'Unknown')}
- Length: {len(seq1.get('sequence', ''))}

Sequence 2:
- ID: {seq2.get('id', 'Unknown')}
- Full Header: {seq2.get('header', 'Unknown')}
- Length: {len(seq2.get('sequence', ''))}
"""

    # Isoform analysis info
    isoform_info = ""
    if ctx.get("isoform_analysis", {}).get("detection", {}).get("is_isoforms"):
        iso_det = ctx["isoform_analysis"]["detection"]
        iso_details = ctx["isoform_analysis"].get("details", {})

        isoform_info = f"""

[ ISOFORM CONFIRMATION & DETAILED ANALYSIS ]
✅ These two proteins are detected as different splicing variants (isoforms) of the same gene:
- Parent Gene: {iso_det.get('base_gene_id', 'Unknown')}
- Detection Method: {iso_det.get('detection_method', 'Unknown')} (Confidence: {iso_det.get('confidence', 'Unknown')})
- Sequence Similarity: {iso_det.get('similarity_percentage', 0):.1f}%
- Length Difference: {iso_det.get('length_difference', 0)} amino acids

[ Detailed Divergence Analysis ]
1. Sequence Alignment:
   - Identity: {iso_details.get('sequence_comparison', {}).get('identity_percentage', 0):.1f}%
   - Primary Divergent Positions: {[f"{d['start']}-{d['end']}" for d in iso_details.get('sequence_comparison', {}).get('differences', [])[:3]] if iso_details.get('sequence_comparison', {}).get('differences') else 'Unknown'}

2. Domain Differences:
   - Shared Domains: {len(iso_details.get('structural_impact', {}).get('domain_differences', {}).get('shared_domains', []))}
   - Unique Domains: Sequence 1 has {len(iso_details.get('structural_impact', {}).get('domain_differences', {}).get('unique_to_seq1', []))}, Sequence 2 has {len(iso_details.get('structural_impact', {}).get('domain_differences', {}).get('unique_to_seq2', []))}

3. Predicted Splicing Events:
   {[f"{e['type']} ({e['position']})" for e in iso_details.get('splicing_events', [])[:2]] if iso_details.get('splicing_events') else 'Not detected'}

[ Mandatory Causal Chain Analysis ]
Please strictly follow the chain below for analysis (each link needs evidence support):
🔗 **Genomic Level**: Same Gene → Alternative Splicing
🔗 **Transcriptional Level**: Splicing Pattern Divergence → mRNA Sequence Variation
🔗 **Translational Level**: mRNA Variation → Amino Acid Sequence Divergence
🔗 **Structural Level**: Sequence Divergence → Secondary/Tertiary Structure Variation → Domain Integrity
🔗 **Functional Level**: Structural Variation → Enzyme Activity/Binding Affinity/Stability Divergence
🔗 **Pathway Level**: Functional Divergence → Signaling Pathway Involvement Variation
🔗 **Phenotypic Level**: Pathway Variation → {trait if trait else 'Target Trait'} Performance Divergence

Special Requirement: Explain why subtle sequence differences (possibly only a few amino acids) lead to significant functional and phenotypic differences.
"""

    trait_text = (f"target trait (e.g., {trait})" if trait else "plant growth, development, or adaptation")

    return f"""
You are an **extremely rigorous Molecular Biology Peer Reviewer**. You need to evaluate the function of candidate protein {focus_id} and its potential involvement in {trait_text} based on the provided bioinformatics evidence.

[ Core Tasks ]
1. Analyze the potential function of each protein and its involvement in {trait_text}.
2. Focus on the relationship between the two proteins (if two sequences are provided), including but not limited to:
   - Sequence Homology/Similarity
   - Commonalities and differences in domain composition
   - Functional redundancy or complementarity
   - Potential synergistic effects or regulatory relationships
   - Evolutionary relationship (Orthologs/Paralogs)
3. Write an **Evidence-based** functional review. Your goal is to "seek truth from facts," not to "tell a good story."

[ Strict Evidence Grading Rules ]
1. **Tier 1 (Conclusive)**: InterPro/GO/BLAST directly hits a specific enzyme (e.g., "Glucanase") or transcription factor family (e.g., "MYB").
   -> *Handling*: Describe its known substrates, catalytic reactions, and classical pathways directly.
2. **Tier 2 (Putative)**: Only structural homology (Foldseek/TM-score > 0.6) or hits to a Superfamily with unknown function.
   -> *Handling*: Must use conservative terms like "likely," "putatively possesses similar biochemical activity to...", etc. **Fabricating specific downstream substrates is strictly prohibited.**
3. **Tier 3 (Weak)**: Large number of "Uncharacterized" or "Hypothetical protein" hits, and pLDDT < 70.
   -> *Handling*: Honestly declare "currently lacks specific literature support," describe only basic physicochemical properties or domain categories. **Forced construction of pathways is prohibited.**

[ Anti-Hallucination Constraints ]
- **Strictly prohibit** fabricating specific interacting proteins (e.g., do not invent "interacts with MPK6" unless BLAST results clearly state MPK6).
- **Strictly prohibit** mimicking "cell wall/secondary metabolism" content from examples unless the protein is actually related to the cell wall.
- **Strictly prohibit** inventing non-existent upstream/downstream signals for the sake of sentence completeness.

[ Conflict Resolution Rules ]
- If sequence evidence (BLAST) conflicts with structural evidence (Foldseek):
  -> Prioritize sequence evidence (if BLAST Identity > 30%).
  -> If sequence homology is extremely low, rely on structural evidence but note "inferred based on structural homology."
  -> Inconsistencies must be pointed out (e.g., "Despite low sequence homology, structural search shows high similarity to [X]").

{isoform_info if isoform_info else '''
[ General Analysis Requirements ]
If these two proteins are from different genes, please analyze:
- Sequence Homology/Similarity
- Commonalities and differences in domain composition
- Functional redundancy or complementarity
- Potential synergistic effects or regulatory relationships
- Evolutionary relationship (Orthologs/Paralogs)
'''}

[ Output Format Requirements ]
Please output **coherent natural paragraphs** following this logical flow:

1. **Identity Definition**: Define the family/enzyme class for each protein and compare their sequence features.
2. **Biochemical Function**: Describe the core biochemical functions of each protein and analyze their functional similarities or differences.
3. **Pathway Association**: Analyze whether the two proteins participate in the same/different pathways and whether there is synergy/antagonism.
4. **Phenotypic Connection**: Synthesize the joint impact or unique roles of both proteins on {trait_text}.
5. **Relationship Summary**: Clearly summarize the relationship type (e.g., homologs, functional redundancy, upstream/downstream regulation).
6. **Limitations Statement**: Point out limitations of the analysis and parts requiring experimental validation.

{seq_info}
[ Background Data ]
- Candidate ID: {focus_id}
- Evidence Rating: {rating}
- Structural Quality: avg_pLDDT={plddt} (treated as low confidence if <70), best_tm={tm} (treated as no structural homology if <0.5)
- Original Description: {focus_text}

[ Available Materials (Your ONLY information source) ]
{_shared_sources_block(ctx)}
"""


def build_prompt_description_explore(ctx: Dict[str, Any], trait: Optional[str], focus_rec: Dict[str, Any],
                                     candidates_table_md: str, top_k: int = 12) -> str:
    """
    Exploration Mode v3.0: Emphasizing causal chain construction from splicing to phenotype
    """
    focus_text = (focus_rec or {}).get("description", "").strip() or "(No description provided)"
    focus_id = focus_rec.get("seq_id", "")
    plddt = focus_rec.get("avg_pLDDT")

    # Get FASTA info and add isoform detection
    fasta_sequences = ctx.get("fasta_sequences", [])
    seq_info = ""
    causal_chain_emphasis = ""

    if len(fasta_sequences) >= 2:
        seq1 = fasta_sequences[0]
        seq2 = fasta_sequences[1]

        # Isoform detection info
        is_isoforms = ctx.get("isoform_analysis", {}).get("detection", {}).get("is_isoforms", False)

        seq_info = f"""
[ Protein Sequence Information ]
Sequence 1:
- ID: {seq1.get('id', 'Unknown')} {'(Likely isoform)' if is_isoforms else ''}
- Length: {len(seq1.get('sequence', ''))}

Sequence 2:
- ID: {seq2.get('id', 'Unknown')} {'(Likely isoform)' if is_isoforms else ''}
- Length: {len(seq2.get('sequence', ''))}

[ Sequence Divergence Hints ]:
- Length Difference: {abs(len(seq1.get('sequence', '')) - len(seq2.get('sequence', '')))} amino acids
- Recommendation: Focus on domain retention/loss potentially caused by alternative splicing.
"""

        if is_isoforms:
            iso_det = ctx["isoform_analysis"]["detection"]
            causal_chain_emphasis = f"""
[ Core Analysis Focus: Causal Chain of Splicing Variants ]
You need to construct a complete **molecular mechanism narrative** explaining:
1. **Splicing Event**: What splicing changes occurred (Exon skipping? Alternative termination?)
2. **Sequence Consequence**: How splicing altered the coding sequence (Insertion/Deletion/Frameshift)
3. **Structural Transition**: How sequence variation reshaped the protein structure (Core domain deformation? Binding pocket alteration?)
4. **Gain/Loss of Function**: How structural changes created new functions or disrupted old ones
5. **Phenotypic Emergence**: How functional changes ultimately affect the {trait if trait else 'target trait'}

Please use vivid metaphors to describe this chain, for example:
"Like using the same architectural blueprint but choosing different construction plans, resulting in entirely different functional zones in the final building..."
- Parent Gene: {iso_det.get('base_gene_id', 'Unknown')}
- Similarity: {iso_det.get('similarity_percentage', 0):.1f}%
"""

    trait_prompt = f"Target Trait: {trait}" if trait else "Target Trait: Not specified (please hypothesize potential impact on plant growth/adaptation)"

    return f"""
You are an **Intuitive Molecular Biology Data Mining Expert**. Facing candidate proteins (possibly two) whose functions are not fully clarified, you need to propose a **Reasonable Working Hypothesis** based on limited bioinformatics clues.

[ Core Tasks ]
1. Analyze the potential function of each protein based on clues.
2. Focus on analyzing the relationship and synergistic mechanisms between the two proteins.
3. Construct a complete "Protein Interaction -> Function -> Trait" logic chain.

[ Mode Definition: Exploration ]
Unlike "Verification Mode," you don't have to stick to conclusive evidence. Your task is to **Connect the Dots**:
1. Utilize weak clues (e.g., low-homology BLAST, generic domains, GO terms).
2. Combine these with your extensive biological knowledge.
3. **Construct a logically self-consistent explanatory narrative** explaining how these proteins *theoretically* jointly influence {trait_prompt}.

{causal_chain_emphasis if causal_chain_emphasis else '''
[ General Analysis Framework ]
Even if they are not explicitly isoforms, explore:
- How minor sequence variations are amplified into significant functional differences.
- How domain rearrangements create new biochemical functions.
- How functional divergence adapts to different environmental demands.
'''}

[ Reasoning Toolbox ]
During analysis, please consider the following molecular mechanisms (select as applicable):
✓ Exon Skipping → Domain Loss → Loss of Enzyme Activity
✓ Intron Retention → Increased Disordered Regions → Altered Interaction Partners
✓ Alternative Promoter → N-terminal Divergence → Subcellular Localization Variation
✓ Alternative PolyA → C-terminal Modification → Stability Divergence
✓ Frameshift Mutation → Entirely Different C-terminus → Neo-functionalization

[ Output Style Requirements ]
Please write in a **scientific research** style:
"According to sequence evidence, we suspect these two proteins are products of the same gene. Divergence starts with alternative splicing of exon X...
This splicing variation leads to the loss of domain Y, thereby disrupting the binding affinity with factor Z...
Ultimately, this molecular-level change manifests as different response strategies of the plant under stress..."

[ Key Constraints ]
- Reasonable speculation is allowed, but each speculation must be supported by clues.
- Use qualifiers like "possibly," "suggesting," "hypothesized," etc.
- Focus on describing the **chain reaction of how changes are transmitted and amplified**.

{seq_info}
[ Background Data ]
- Candidate ID: {focus_id}
- Structural Quality: avg_pLDDT={plddt}
- Original Description: {focus_text}

[ Available Materials ]
{_shared_sources_block(ctx)}
"""


def build_prompt_evidence(ctx: Dict[str, Any], trait: Optional[str], focus_rec: Dict[str, Any],
                          top_k: int = 12) -> str:
    trait_text = (f"**{trait}**" if trait else "**Not Provided**")

    # Get sequence info
    fasta_sequences = ctx.get("fasta_sequences", [])

    # Check if isoforms
    isoform_evidence_note = ""
    isoform_analysis = ctx.get("isoform_analysis", {})
    if isoform_analysis.get("detection", {}).get("is_isoforms"):
        iso_det = isoform_analysis["detection"]
        iso_details = isoform_analysis.get("details", {})

        isoform_evidence_note = f"""
[ IMPORTANT ] Detected potential splicing variants of the same gene:
- Parent Gene ID: {iso_det.get('base_gene_id', 'Unknown')}
- Similarity: {iso_det.get('similarity_percentage', 0):.1f}%
- Divergence Analysis: Detected {iso_details.get('sequence_comparison', {}).get('total_differences', 0)} total divergent positions

Please specifically collect the following evidence:
1. Sequence alignment results (Similarity percentage)
2. Domain retention/loss evidence
3. Characteristics of alternative splicing (e.g., exon boundary sequences)
4. Functional divergence evidence (different GO terms, interaction partners, etc.)
"""

    seq_note = ""
    if len(fasta_sequences) >= 2:
        seq_note = f"(Note: Must include specific evidence related to splicing variants) {isoform_evidence_note}"

    return f"""
You are an **Evidence Curator**. Please extract evidence items one by one **ONLY from the following outputs**.

Special attention: {isoform_evidence_note if isoform_evidence_note else 'Divergence analysis evidence for the two proteins'}

Output a Markdown table (at least 6 items; if insufficient, list all):
| ID | Source Path | Position/Field | Excerpt/Value | Key Point Supported (One sentence) | Causal Chain Link |
|---|---|---|---|---|---|
- **Added "Causal Chain Link" column**: Indicate which link the evidence belongs to (Splicing Divergence, Sequence Divergence, Structural Divergence, Functional Divergence, Phenotypic Divergence).
- Must include at least 2 items of evidence directly supporting the "splicing variant" hypothesis (if any).
- Must include at least 1 item of evidence regarding "Sequence Divergence → Structural Impact".
- Must include at least 1 item of evidence regarding "Structural Divergence → Functional Impact".

{_shared_sources_block(ctx)}
"""


def postprocess(md: str, ctx: Dict[str, Any]) -> str:
    return normalize_interpro_ids(md)


def prepare_aligned_fasta(fasta_path: str, output_dir: str = "."):

    fasta_path = Path(fasta_path)
    if not fasta_path.exists():
        print(f"Error: FASTA not exist: {fasta_path}")
        return None

    try:
        if BIO_AVAILABLE:
            records = list(SeqIO.parse(fasta_path, "fasta"))
        else:
            records = []
            with open(fasta_path, 'r') as f:
                current_id = ""
                current_seq = ""
                for line in f:
                    if line.startswith(">"):
                        if current_id:
                            records.append({"id": current_id, "seq": current_seq})
                        current_id = line[1:].strip().split()[0]
                        current_seq = ""
                    else:
                        current_seq += line.strip()
                if current_id:
                    records.append({"id": current_id, "seq": current_seq})
    except Exception as e:
        print(f"Fail to read fasta file: {e}")
        return None

    if len(records) < 2:
        print("Warning: At least 2 sequences are required for analysis.")
        return None

    print(f"Read {len(records)} sequences.")

    if BIO_AVAILABLE and len(records) == 2:
        try:
            aligner = PairwiseAligner()
            aligner.mode = 'global'
            alignments = aligner.align(str(records[0].seq), str(records[1].seq))

            if alignments:
                alignment = alignments[0]
                aligned_str = str(alignment)

                output_aln = Path(output_dir) / f"{fasta_path.stem}_aligned.txt"
                with open(output_aln, 'w') as f:
                    f.write(aligned_str)
                print(f"Simple comparison completed: {output_aln}")
        except Exception as e:
            print(f"Sequence alignment failed. Use the original sequence: {e}")

    output_fasta = Path(output_dir) / f"{fasta_path.stem}_for_analysis.fasta"
    try:
        if BIO_AVAILABLE:
            SeqIO.write(records, output_fasta, "fasta")
        else:
            with open(output_fasta, 'w') as f:
                for rec in records:
                    f.write(f">{rec['id']}\n")
                    f.write(f"{rec['seq']}\n")
        print(f"Sequence saved: {output_fasta}")
        return str(output_fasta)
    except Exception as e:
        print(f"fail to save: {e}")
        return str(fasta_path)


# ========= CLI =========
def main():
    ap = argparse.ArgumentParser(
        description="Evidence-grounded phenotype/mechanism inference with dual outputs (description + evidence)."
    )
    ap.add_argument("--annotation_dir", required=True)
    ap.add_argument("--with_struct_dir", required=True)
    ap.add_argument("--final_dir", required=True)
    ap.add_argument(
        "--structsearch_dir",
        required=False,
        default=None,
        help="Optional; defaults to the same value as --final_dir"
    )
    ap.add_argument("--fasta", default=None)
    ap.add_argument("--prepare_sequences", action="store_true",
                    help="Pre-process sequence files (alignment, feature extraction)")

    # Backend and Model (dashscope / ollama only)
    ap.add_argument("--backend", choices=["dashscope", "ollama"], default="dashscope")
    ap.add_argument("--model", default="qwen3-max")
    ap.add_argument("--api_key", default=os.getenv("DASHSCOPE_API_KEY", ""))
    ap.add_argument(
        "--dashscope_base_url",
        default=os.getenv(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    )

    # Generation Parameters
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--max_tokens", type=int, default=8000)
    ap.add_argument("--num_ctx", type=int, default=8192)  # Used by ollama only
    ap.add_argument("--top_k", type=int, default=12)

    # Mode and Phenotype Traits
    ap.add_argument(
        "--mode",
        choices=["verify", "explore", "report", "inference"],
        default="explore",
        help=(
            "verify = Verification mode (literature consensus, conservative); "
            "explore = Exploration mode (reasonable inference based on evidence); "
            "report/inference are for backward compatibility, equivalent to verify/explore respectively"
        ),
    )
    ap.add_argument("--trait", default=None)

    # evidence_heavy: Expand context truncation
    ap.add_argument(
        "--evidence_heavy",
        action="store_true",
        help="Expand context truncation (TSV max 200 rows / Text max 20,000 chars / FASTA max 300 rows) for long-form generation"
    )

    # Output
    ap.add_argument("--out_description", required=True, help="Output path for plain text mechanism narrative")
    ap.add_argument("--out_evidence", required=True, help="Output path for evidence checklist")
    ap.add_argument("--out_markdown", required=False, help="Optional: Output path for integrated report")
    ap.add_argument("--out_isoform_report", required=False, help="Optional: Output path for isoform analysis report")

    args = ap.parse_args()

    # Pre-process sequences if specified
    if args.prepare_sequences and args.fasta:
        print("[*] Pre-processing sequence files...")
        prepared_fasta = prepare_aligned_fasta(args.fasta, Path(args.fasta).parent)
        if prepared_fasta:
            print(f"[*] Using pre-processed sequence: {prepared_fasta}")
            args.fasta = prepared_fasta
        else:
            print("[!] Sequence pre-processing failed, using original file")

    # 1) Collect Context
    print("[*] Collecting context data...")
    ctx = collect_context(
        args.annotation_dir,
        args.with_struct_dir,
        args.final_dir,
        args.structsearch_dir,
        args.fasta,
        heavy=args.evidence_heavy
    )

    # Save Isoform report
    if args.out_isoform_report and ctx.get("isoform_analysis", {}).get("report"):
        isoform_report_path = Path(args.out_isoform_report)
        isoform_report_path.parent.mkdir(parents=True, exist_ok=True)
        isoform_report_path.write_text(ctx["isoform_analysis"]["report"], encoding="utf-8")
        print(f"[OK] Wrote isoform analysis report to: {isoform_report_path}")

    # 2) Load final.py products; determine focus entry and candidate table
    print("[*] Loading final records...")
    records = load_final_records_unified(args.final_dir)
    focus_rec = pick_focus_record(records)  # Select Top-1 based on composite score
    candidates_table_md = build_candidates_table_md(records, top_k=args.top_k)

    # 3) Build System Prompt dynamically based on mode
    if args.mode in ["verify", "report"]:
        # Verification Mode: Strict constraints, zero tolerance for hallucination
        system_content = (
            "You are a rigorous Bioinformatics/Crop Biology Assistant. Your core principle: No evidence, no conclusion. "
            "All inferences must be strictly based on the provided context (BLAST/InterPro/Foldseek, etc.). "
            "If avg_pLDDT < 70 or best_tm < 0.5, you must explicitly lower the confidence level in the narrative. "
            "Do not mimic specific gene names or pathways from examples unless they actually exist in the evidence. "
            "Special Requirement: You must perform a detailed analysis of the relationships between multiple protein sequences, "
            "including sequence similarity, domain composition, and functional relevance. "
            "If isoforms (splicing variants of the same gene) are detected, you must analyze the full causal chain from splicing to trait."
        )
    else:
        # Exploration Mode: Encourages reasonable inference based on clues
        system_content = (
            "You are a biological hypothesis construction expert with keen insight. Your task is to build reasonable working models "
            "based on limited clues. You are allowed to use terms like 'possibly', 'suggests', or 'potential association' to connect "
            "faint evidence points. While speculation is permitted, fabricating data (e.g., inventing non-existent domains) is strictly prohibited. "
            "Your goal is to provide valuable experimental directions for researchers. "
            "Special Requirement: Focus on analyzing relationships and potential interaction mechanisms between multiple protein sequences. "
            "If isoforms are detected, construct the complete Splicing → Sequence → Structure → Function → Trait causal chain."
        )

    # 4) Select Prompt based on mode
    mode = args.mode
    if mode == "report":
        mode = "verify"
    elif mode == "inference":
        mode = "explore"

    if mode == "verify":
        prompt_desc = build_prompt_description_verify(
            ctx, args.trait, focus_rec, candidates_table_md, top_k=args.top_k
        )
    else:  # explore
        prompt_desc = build_prompt_description_explore(
            ctx, args.trait, focus_rec, candidates_table_md, top_k=args.top_k
        )

    prompt_evi = build_prompt_evidence(ctx, args.trait, focus_rec, top_k=args.top_k)

    # Adjust temperature for verification mode to reduce hallucinations
    run_temperature = args.temperature
    if args.mode in ["verify", "report"]:
        if args.temperature == 0.5:  # Only override if user hasn't manually passed a parameter
            run_temperature = 0.2
            print(
                f"[*] Mode is {args.mode}; automatically reducing Temperature to {run_temperature} to minimize hallucination.")

    def _call_llm(prompt_text: str) -> str:
        if args.backend == "ollama":
            system_msg = {"role": "system", "content": system_content}
            user_msg = {"role": "user", "content": prompt_text}
            return call_ollama_chat(
                args.model,
                [system_msg, user_msg],
                run_temperature,
                args.max_tokens,
                args.num_ctx
            )
        else:  # dashscope
            return call_dashscope_chat(
                args.model,
                args.api_key,
                args.dashscope_base_url,
                system_content,
                prompt_text,
                run_temperature,
                args.max_tokens
            )

    # 5) Generation and Writing
    print("[*] Generating mechanism narrative...")
    try:
        out_desc = _call_llm(prompt_desc)
        if out_desc.strip().startswith("Error calling"):
            out_desc = "# Generation Failed (Mechanism Narrative)\n" + out_desc.strip()
    except Exception as e:
        out_desc = f"# Generation Failed (Mechanism Narrative)\nException: {e}"

    print("[*] Generating evidence checklist...")
    try:
        out_evi = _call_llm(prompt_evi)
        if out_evi.strip().startswith("Error calling"):
            out_evi = "# Generation Failed (Evidence Checklist)\n" + out_evi.strip()
    except Exception as e:
        out_evi = f"# Generation Failed (Evidence Checklist)\nException: {e}"

    out_desc = postprocess(out_desc, ctx)
    out_evi = postprocess(out_evi, ctx)

    out_path_desc = Path(args.out_description)
    out_path_desc.parent.mkdir(parents=True, exist_ok=True)
    out_path_desc.write_text(out_desc, encoding="utf-8")
    print(f"[OK] Wrote mechanism description to: {out_path_desc}")

    out_path_evi = Path(args.out_evidence)
    out_path_evi.parent.mkdir(parents=True, exist_ok=True)
    out_path_evi.write_text(out_evi, encoding="utf-8")
    print(f"[OK] Wrote evidence table to: {out_path_evi}")

    if args.out_markdown:
        if mode == "verify":
            prompt_md = build_prompt_rigorous(ctx, top_k=args.top_k)
        else:
            prompt_md = build_prompt_description_explore(
                ctx, args.trait, focus_rec, candidates_table_md, top_k=args.top_k
            )
        print("[*] Generating integrated report...")
        try:
            out_md = _call_llm(prompt_md)
        except Exception as e:
            out_md = f"# Generation Failed (Integrated Report)\nException: {e}"
        out_md = postprocess(out_md, ctx)
        out_path_md = Path(args.out_markdown)
        out_path_md.parent.mkdir(parents=True, exist_ok=True)
        out_path_md.write_text(out_md, encoding="utf-8")
        print(f"[OK] Wrote integrated report to: {out_path_md}")

    print("[*] Analysis complete!")


if __name__ == "__main__":
    main()