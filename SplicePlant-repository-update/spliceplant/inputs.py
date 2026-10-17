"""Input validation shared by the sequence-model entry points."""
import csv
DNA_ALPHABET = set("ACGTRYSWKMBDHVN")

def validate_sequence(sequence, context="sequence"):
    sequence = sequence.upper()
    if not sequence or set(sequence) - DNA_ALPHABET:
        raise ValueError(f"{context}: expected nonempty DNA (ACGT or IUPAC ambiguity codes)")
    return sequence

def read_fasta(path):
    records, header, chunks = [], None, []
    with open(path, encoding="utf-8-sig") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, validate_sequence("".join(chunks), header)))
                header, chunks = line[1:].strip(), []
                if not header:
                    raise ValueError("Empty FASTA header")
            else:
                if header is None:
                    raise ValueError("Sequence before first FASTA header")
                chunks.append(line)
    if header is not None:
        records.append((header, validate_sequence("".join(chunks), header)))
    if not records:
        raise ValueError("No FASTA records found")
    return records

def read_labeled_csv(path):
    sequences, labels = [], []
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not {"seq", "label"} <= set(reader.fieldnames or []):
            raise ValueError("CSV requires seq and label columns")
        for row_number, row in enumerate(reader, 2):
            seq = validate_sequence(row.get("seq") or "", f"CSV row {row_number}")
            try:
                label = int(row["label"])
            except (ValueError, TypeError) as exc:
                raise ValueError(f"CSV row {row_number}: label must be 0, 1 or 2") from exc
            if label not in (0, 1, 2):
                raise ValueError(f"CSV row {row_number}: label must be 0, 1 or 2")
            sequences.append(seq)
            labels.append(label)
    lengths = {len(seq) for seq in sequences}
    if len(lengths) != 1 or next(iter(lengths)) % 2 != 1:
        raise ValueError("CSV must contain fixed-length, nonempty, odd-length DNA windows")
    return sequences, labels
