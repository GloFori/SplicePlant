"""The same analysis settings are translated into local and remote arguments."""
import math
import re


def analysis_settings(args):
    apps = sorted(set(x.strip() for x in args.interpro_applications.split(",") if x.strip()))
    if any(not re.fullmatch(r"[A-Za-z0-9_.-]+", x) for x in apps):
        raise ValueError("Invalid InterPro application name")
    if not math.isfinite(args.evalue) or args.evalue <= 0 or args.max_hits < 1:
        raise ValueError("E-value and maximum hit count must be positive")
    if args.word_size < 2 or args.gap_open < 0 or args.gap_extend < 0:
        raise ValueError("Invalid BLAST word size or gap cost")
    return {
        "blast": {"database": args.remote_database or "nr", "evalue": args.evalue,
                  "max_hits": args.max_hits, "seg": args.seg, "matrix": args.matrix,
                  "word_size": args.word_size, "gap_open": args.gap_open,
                  "gap_extend": args.gap_extend, "comp_based_stats": args.comp_based_stats},
        "interpro": {"applications": apps, "goterms": True, "pathways": True},
    }


def local_blast_options(settings):
    b = settings["blast"]
    return ["-evalue", str(b["evalue"]), "-max_target_seqs", str(b["max_hits"]),
            "-seg", b["seg"], "-matrix", b["matrix"], "-word_size", str(b["word_size"]),
            "-gapopen", str(b["gap_open"]), "-gapextend", str(b["gap_extend"]),
            "-comp_based_stats", str(b["comp_based_stats"])]


def remote_blast_options(settings):
    b = settings["blast"]
    return {"DATABASE": b["database"], "EXPECT": str(b["evalue"]), "HITLIST_SIZE": str(b["max_hits"]),
            "FILTER": "L" if b["seg"] == "yes" else "F", "MATRIX": b["matrix"],
            "WORD_SIZE": str(b["word_size"]), "GAPCOSTS": f'{b["gap_open"]} {b["gap_extend"]}',
            "COMPOSITION_BASED_STATISTICS": str(b["comp_based_stats"])}
