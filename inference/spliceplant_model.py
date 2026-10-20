"""Student architecture; compatible with the archived state_dict tensor names."""
import numpy as np
import torch
from torch import nn

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
    def __init__(self, dim=40, pos_len=1200, depth=6, causal=False, reversible=False, attention="identity"):
        super().__init__()
        self.pos_emb = nn.Embedding(pos_len, dim)
        if attention == "sinkhorn":
            try:
                from sinkhorn_transformer import SinkhornTransformer
            except ImportError as exc:
                raise RuntimeError("This checkpoint needs sinkhorn_transformer from its training environment.") from exc
            self.attn = SinkhornTransformer(
                dim, depth,
                heads=8, n_local_attn_heads=2,
                attn_layer_dropout=0.1, layer_dropout=0.1,
                ff_dropout=0.1, ff_chunks=10,
                causal=causal, reversible=reversible, non_permutative=True
            )
        else:
            self.attn = nn.Identity()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        b, dim, Lp = x.size()
        pos_idx = torch.arange(0, Lp, device=x.device).unsqueeze(0).expand(b, Lp)
        x = x.transpose(1,2).contiguous()  # [B, L', dim]
        x = x + self.pos_emb(pos_idx)
        x = self.attn(x)
        x = self.norm(x)
        x = x.transpose(1,2).contiguous()
        return x

class SpEncoder(nn.Module):
    def __init__(self, L=50):
        super().__init__()
        self.W = np.asarray([11,11,11,11,11,11,11,11, 21,21,21,21,21,21,21,21])
        self.AR= np.asarray([ 1, 1, 1, 1, 4, 4, 4, 4, 10,10,10,10,20,20,20,20])
        self.conv1 = nn.Conv1d(4, L, 1)
        self.skip  = nn.Conv1d(L, L, 1)
        self.resblocks, self.convs = nn.ModuleList(), nn.ModuleList()
        for i in range(len(self.W)):
            self.resblocks.append(ResBlock(L, self.W[i], self.AR[i]))
            if ((i+1)%4==0) or ((i+1)==len(self.W)):
                self.convs.append(nn.Conv1d(L, L, 1))
        self.splice_output = nn.Conv1d(L, 40, 1)

    def forward(self, x):
        x = x.permute(0,2,1)
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
    def __init__(self, output_classes=3, dim=40, seq_len=1001, attention="identity"):
        super().__init__()
        self.encoder = SpEncoder(L=50)
        self.attn = AttnBlock(dim=dim, pos_len=max(1200, seq_len), attention=attention)
        self.fc1 = nn.Linear(seq_len, 512)
        self.conv1 = nn.Conv1d(dim, output_classes, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):             # [B, L, 4]
        x = self.encoder(x)           # [B, 40, L]
        x = self.fc1(x)               # [B, 40, 512]
        x = self.relu(x)
        x = self.attn(x)              # [B, 40, 512]
        x = self.conv1(x)             # [B, 3, 512]
        x = self.dropout(x)
        x = self.fc2(x)               # [B, 3, 1]
        x = torch.mean(x, dim=2)      # [B, 3]
        return x


def read_checkpoint(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    if not isinstance(state, dict) or not state:
        raise ValueError("Checkpoint must contain a nonempty state_dict.")
    if all(k.startswith("module.") for k in state):
        state = {k[7:]: v for k, v in state.items()}
    if "fc1.weight" not in state or state["fc1.weight"].ndim != 2:
        raise ValueError("Missing valid fc1.weight in checkpoint.")
    return state

def checkpoint_window(state):
    win = int(state["fc1.weight"].shape[1])
    if win < 1 or win % 2 == 0:
        raise ValueError("Checkpoint window must be a positive odd number.")
    return win

def checkpoint_attention(state):
    return "sinkhorn" if any(k.startswith("attn.attn.") for k in state) else "identity"

def load_student(path, device="cpu", window=0):
    state = read_checkpoint(path)
    expected = checkpoint_window(state)
    if window and window != expected:
        raise ValueError(f"Window {window} does not match checkpoint window {expected}.")
    model = StudentModel(seq_len=expected, attention=checkpoint_attention(state))
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), expected
