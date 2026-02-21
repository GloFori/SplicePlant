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
class ResBlock(nn.Module):
    def __init__(self, L, W, AR):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(L)
        self.conv1 = nn.Conv1d(L, L, W, dilation=AR, padding='same')
        self.bn2 = nn.BatchNorm1d(L)
        self.conv2 = nn.Conv1d(L, L, W, dilation=AR, padding='same')
    def forward(self, x):
        out = self.bn1(x); out = torch.relu(out); out = self.conv1(out)
        out = self.bn2(out); out = torch.relu(out); out = self.conv2(out)
        return out + x

class AttnBlock(nn.Module):
    def __init__(self, dim=40, pos_len=1200):
        super().__init__()
        self.pos_emb = nn.Embedding(pos_len, dim)
        self.attn = nn.Identity()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        b, dim, Lp = x.size()
        pos_idx = torch.arange(0, Lp, device=x.device).unsqueeze(0).expand(b, Lp)
        x = x.transpose(1, 2).contiguous()
        x = x + self.pos_emb(pos_idx)
        x = self.attn(x)
        x = self.norm(x)
        x = x.transpose(1, 2).contiguous()
        return x

class SpEncoder(nn.Module):
    def __init__(self, L=50):
        super().__init__()
        self.W  = np.asarray([11,11,11,11,11,11,11,11, 21,21,21,21,21,21,21,21])
        self.AR = np.asarray([ 1, 1, 1, 1, 4, 4, 4, 4, 10,10,10,10,20,20,20,20])
        self.conv1 = nn.Conv1d(4, L, 1)
        self.skip  = nn.Conv1d(L, L, 1)
        self.resblocks, self.convs = nn.ModuleList(), nn.ModuleList()
        for i in range(len(self.W)):
            self.resblocks.append(ResBlock(L, self.W[i], self.AR[i]))
            if ((i+1)%4==0) or ((i+1)==len(self.W)):
                self.convs.append(nn.Conv1d(L, L, 1))
        self.splice_output = nn.Conv1d(L, 40, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        conv = self.conv1(x)
        skip = self.skip(conv)
        j = 0
        for i in range(len(self.W)):
            conv = self.resblocks[i](conv)
            if ((i+1)%4==0) or ((i+1)==len(self.W)):
                dense = self.convs[j](conv); j += 1
                skip = skip + dense
        out = self.splice_output(skip)
        return out

class StudentModel(nn.Module):
    def __init__(self, output_classes=3, dim=40, seq_len=1001):
        super().__init__()
        self.encoder = SpEncoder(L=50)
        self.attn    = AttnBlock(dim=dim, pos_len=max(1200, seq_len))
        self.fc1     = nn.Linear(seq_len, 512)
        self.conv1   = nn.Conv1d(dim, output_classes, 1)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2     = nn.Linear(512, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.attn(x)
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.mean(x, dim=2)
        return x

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
    sequences = []
    current_header = None
    current_seq = []
    
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if current_header is not None:
                    full_seq = ''.join(current_seq)
                    sequences.append((current_header, full_seq))
                current_header = line[1:]
                current_seq = []
            else:
                current_seq.append(line.upper())
    
    if current_header is not None:
        full_seq = ''.join(current_seq)
        sequences.append((current_header, full_seq))
    
    results = []
    
    for header, sequence in sequences:
        if is_negative_strand(header):
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
    parser.add_argument("--weights", type=str, default="../LLM/save_model/student_best.pth", help="path to model weights")
    parser.add_argument("--output_file", type=str, default="predictions.txt", help="path to output")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--window_size", type=int, default=401)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not os.path.exists(args.input_dir) or not os.path.isdir(args.input_dir):
        print(f"Error: The input folder does not exist or is not a directory: {args.input_dir}")
        return

    if not os.path.exists(args.weights):
        print(f"Error: The weight file does not exist: {args.weights}")
        return

    print(f"Loading model weights: {args.weights} ...")
    device = torch.device(args.device)
    model = StudentModel(output_classes=3, dim=40, seq_len=args.window_size).to(device)

    try:
        state = torch.load(args.weights, map_location=device)
        model.load_state_dict(state, strict=True)
    except Exception as e:
        print(f"Error: Failed to load model weights: {e}")
        return

    extensions = ['*.fasta', '*.fa', '*.fna', '*.FASTA', '*.FA']
    fasta_files = []
    for ext in extensions:
        fasta_files.extend(glob.glob(os.path.join(args.input_dir, ext)))

    fasta_files = sorted(list(set(fasta_files)))

    if not fasta_files:
        print(f"Warning: No FASTA file was found in {args.input_dir}.")
        return

    print(f"A total of {len(fasta_files)} FASTA files were found and processing has begun...")
    print("-" * 50)

    total_files_processed = 0

    for i, fasta_file in enumerate(fasta_files):
        filename = os.path.basename(fasta_file)

        file_prefix = filename.split('.')[0]
        output_filename = f"{file_prefix}.txt"
        output_path = os.path.join(args.input_dir, output_filename)

        print(f"[{i + 1}/{len(fasta_files)}] processing: {filename} -> output: {output_filename}")

        results = process_fasta_file(fasta_file, model, device, args)

        if not results:
            print(f"  -> Skip: Unable to generate valid prediction results in this file")
            continue

        try:
            with open(output_path, 'w') as f:
                for header, sequence, predictions in results:
                    f.write(f"{sequence}\n")
                    f.write(f"{predictions}\n")
            print(f"  -> FINISH")
            total_files_processed += 1

            total_bases = sum(len(seq) for _, seq, _ in results)
            acceptor_count = sum(p.count('0') for _, _, p in results)
            donor_count = sum(p.count('1') for _, _, p in results)
            print(f"     Statistics: Length {total_bases}, Acceptor {acceptor_count}, Donor {donor_count}")

        except Exception as e:
            print(f"  -> Error: Failed to write to file {output_path}: {e}")

    print("-" * 50)
    print(f"DOWN! Successfully processed: {total_files_processed}/{len(fasta_files)}")


if __name__ == "__main__":
    main()