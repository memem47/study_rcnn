import torch
import torch.nn as nn
from collections import OrderedDict

# ────────────────────────── 基本ブロック ──────────────────────────
def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )

class ConvLSTMCell(nn.Module):
    """2D ConvLSTM (single step)."""
    def __init__(self, in_ch, hid_ch, k=3):
        super().__init__()
        padding = k // 2
        self.conv = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, k, padding=padding)

    def forward(self, x, h, c):
        # x: (B, Cin, H, W),  h,c: (B, Chid, H, W)
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = conv_block(in_ch, out_ch)

    def forward(self, x, enc_feat):
        x = self.up(x)
        # Pad if needed (odd dims)
        diffY = enc_feat.size(2) - x.size(2)
        diffX = enc_feat.size(3) - x.size(3)
        x = nn.functional.pad(x, [diffX // 2, diffX - diffX // 2,
                                  diffY // 2, diffY - diffY // 2])
        x = torch.cat([enc_feat, x], dim=1)
        return self.conv(x)

# ──────────────────────── ConvLSTM-U-Net 本体 ───────────────────────
class ConvLSTMUNet(nn.Module):
    """
    入力:  (B, T, C, H, W)
    出力:  (B, T, n_classes, H, W)
    """
    def __init__(self, in_ch=3, n_classes=2, base=32):
        super().__init__()
        # ---- Encoder 2D CNN ----
        self.enc1 = conv_block(in_ch, base)
        self.enc2 = conv_block(base, base * 2)
        self.enc3 = conv_block(base * 2, base * 4)
        pool = nn.MaxPool2d(2)
        self.pool = pool

        # ---- Bottleneck ConvLSTM ----
        self.clstm = ConvLSTMCell(base * 4, base * 4)

        # ---- Decoder ----
        self.up2 = UpBlock(base * 4, base * 2)
        self.up1 = UpBlock(base * 2, base)
        self.out_conv = nn.Conv2d(base, n_classes, 1)

    def forward(self, x_seq):
        B, T, C, H, W = x_seq.shape
        h = x_seq.new_zeros(B, 4 * 32, H // 4, W // 4)  # base*4
        c = h.clone()

        outs = []
        for t in range(T):
            x = x_seq[:, t]                 # (B,C,H,W)
            e1 = self.enc1(x)               # H,W
            e2 = self.enc2(self.pool(e1))   # H/2
            e3 = self.enc3(self.pool(e2))   # H/4

            h, c = self.clstm(e3, h, c)     # ConvLSTM

            d2 = self.up2(h, e2)            # H/2
            d1 = self.up1(d2, e1)           # H
            out = self.out_conv(d1)         # logits
            outs.append(out)

        return torch.stack(outs, dim=1)      # (B,T,Cout,H,W)
