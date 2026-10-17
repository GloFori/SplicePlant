import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import glob
# Label mapping used by the model:
# class 0 = Acceptor, class 1 = Donor, class 2 = None
# The prediction string written to .txt uses these indices directly.

DNA_VOCAB = {"A": 0, "C": 1, "G": 2, "T": 3}
COMPLEMENT = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}

def reverse_complement(seq):
    complemented = ''.join(COMPLEMENT.get(base, 'N') for base in seq.upper())
    return complemented[::-1]

def is_negative_strand(header):
    negative_indicators = ['(-)', '(-strand)', '(minus)', '(negative)', 'complement', 'reverse', 'negative']
    header_lower = header.lower()
    for indicator in negative_indicators:
        if indicator in header_lower:
            return True
    return False

def one_hot_encode(seq: str) -> np.ndarray:
    L = len(seq)
    x = np.zeros((L, 4), dtype=np.float32)
    for i, ch in enumerate(seq.upper()):
        if ch in DNA_VOCAB:
            x[i, DNA_VOCAB[ch]] = 1.0
    return x

# -----------------------------
# Dataset
# -----------------------------
class LongSeqDataset(Dataset):
    def __init__(self, long_sequence, window_size=1001, stride=1):
        self.seq = long_sequence.upper()
        self.window_size = window_size
        self.stride = stride

        valid_chars = set('ACGT')
        self.seq = ''.join([c if c in valid_chars else 'N' for c in self.seq])

        self.windows = []
        self.start_positions = []
        
        seq_len = len(self.seq)
        for start in range(0, seq_len - window_size + 1, stride):
            end = start + window_size
            window_seq = self.seq[start:end]
            if 'N' not in window_seq:
                self.windows.append(window_seq)
                self.start_positions.append(start)
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        window_seq = self.windows[idx]
        start_pos = self.start_positions[idx]
        x = one_hot_encode(window_seq)
        return torch.tensor(x, dtype=torch.float32), start_pos

# -----------------------------
# Student model
# -----------------------------
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spliceplant.model import StudentModel, load_student, read_checkpoint, checkpoint_window
from spliceplant.inputs import read_fasta, read_labeled_csv

# -----------------------------
# predict_per_base
# -----------------------------
def predict_per_base(model, loader, device, seq_len, window_size):
    model.eval()
    
    predictions_sum = np.zeros((seq_len, 3), dtype=np.float32)
    predictions_count = np.zeros(seq_len, dtype=np.int32)
    
    with torch.no_grad():
        for x, start_pos in tqdm(loader, desc="Predicting", leave=False):
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            
            batch_size = probs.shape[0]
            for i in range(batch_size):
                center_pos = start_pos[i].item() + window_size // 2
                if center_pos < seq_len:
                    predictions_sum[center_pos] += probs[i]
                    predictions_count[center_pos] += 1
    
    per_base_predictions = []
    for i in range(seq_len):
        if predictions_count[i] > 0:
            avg_probs = predictions_sum[i] / predictions_count[i]
            predicted_class = np.argmax(avg_probs)
        else:
            predicted_class = 2
        per_base_predictions.append(str(predicted_class))
    
    return ''.join(per_base_predictions)

def process_fasta_file(fasta_path, model, device, args):
    sequences = read_fasta(fasta_path)

    results = []
    
    for header, sequence in sequences:
        if args.strand == "reverse" or (args.strand == "header" and is_negative_strand(header)):
            sequence = reverse_complement(sequence)

        valid_chars = set('ACGT')
        cleaned_seq = ''.join([c if c in valid_chars else 'N' for c in sequence.upper()])
        
        if len(cleaned_seq) < args.window_size:
            print(f"Warning: The length of sequence '{header}' ({len(cleaned_seq)} base pairs) is less than the window size. It has been skipped.")
            continue

        ds = LongSeqDataset(cleaned_seq, window_size=args.window_size, stride=args.stride)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

        predictions = predict_per_base(model, loader, device, len(cleaned_seq), args.window_size)

        results.append((header, cleaned_seq, predictions))
    
    return results

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="FASTA sequence splice site prediction")
    parser.add_argument("--input_dir", type=str, required=True, help="path to fasta fold")
    parser.add_argument("--weights", type=str, required=True, help="path to model weights")
    parser.add_argument("--output_dir", required=True, help="Separate directory for two-line prediction files")
    parser.add_argument("--strand", choices=["header", "as-is", "reverse"], default="header",
                        help="header preserves historical header-based reverse-complement behavior")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--window_size", type=int, default=0, help="0: infer window from checkpoint")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.batch_size < 1 or args.stride < 1 or args.window_size < 0:
        parser.error("batch_size/stride must be positive; window_size cannot be negative")
    if not Path(args.input_dir).is_dir():
        parser.error("input_dir does not exist")
    device = torch.device(args.device)
    model, args.window_size = load_student(args.weights, device, args.window_size)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    extensions = ['*.fasta', '*.fa', '*.fna', '*.FASTA', '*.FA']
    fasta_files = []
    for ext in extensions:
        fasta_files.extend(glob.glob(os.path.join(args.input_dir, ext)))

    fasta_files = sorted(list(set(fasta_files)))

    if not fasta_files:
        print(f"Warning: No FASTA file was found in {args.input_dir}.")
        raise SystemExit(1)

    print(f"A total of {len(fasta_files)} FASTA files were found and processing has begun...")
    print("-" * 50)

    total_files_processed = 0

    for i, fasta_file in enumerate(fasta_files):
        filename = os.path.basename(fasta_file)

        file_prefix = filename
        print(f"[{i + 1}/{len(fasta_files)}] processing: {filename}")

        results = process_fasta_file(fasta_file, model, device, args)

        if not results:
            print(f"  -> Skip: Unable to generate valid prediction results in this file")
            continue

        for record_number, (header, sequence, predictions) in enumerate(results, 1):
            # One record per file: attribution consumes paired sequence/label lines.
            output_path = Path(args.output_dir) / f"{file_prefix}.record{record_number:04d}.txt"
            if output_path.exists():
                raise FileExistsError(f"Refusing to overwrite {output_path}")
            output_path.write_text(f"{sequence}\n{predictions}\n", encoding="utf-8")
            import json
            output_path.with_suffix(".json").write_text(json.dumps({"source_file": str(Path(fasta_file).resolve()),
                "record_header": header, "strand_policy": args.strand, "window": args.window_size,
                "stride": args.stride, "unscored_label": 2}, indent=2), encoding="utf-8")
        total_files_processed += 1

    print("-" * 50)
    print(f"Done. Successfully processed: {total_files_processed}/{len(fasta_files)}")
    if total_files_processed == 0:
        raise SystemExit("No sequences could be processed; check input lengths and model window.")


if __name__ == "__main__":
    main()
