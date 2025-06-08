# train_conv_lstm_unet.py --------------------------------------------------
import torch, numpy as np
from torch import nn
from tqdm import tqdm

from dataset import build_loaders
from model_conv_lstm_unet import ConvLSTMUNet   # ← 前回提示コードを別ファイル化したもの

# ──────────────── Dice 損失 & メトリック ────────────────
def dice_coef(pred, gt, eps=1e-6):
    pred = (pred > 0.5).float()
    inter = (pred * gt).sum()
    union = pred.sum() + gt.sum()
    return (2*inter + eps) / (union + eps)

def dice_loss(pred_logits, gt):
    pred = torch.sigmoid(pred_logits)
    return 1 - dice_coef(pred, gt)

# ──────────────── 学習ループ ────────────────
def train_epoch(model, loader, criterion, opt, device):
    model.train(); running_loss = 0
    for x, y in tqdm(loader, desc='train', leave=False):
        # x:(B,T,C,H,W) → ConvLSTM は (B,T,C,H,W) が欲しい
        x, y = x.to(device), y.float().to(device)
        opt.zero_grad()
        out = model(x)           # (B,T,1,H,W)
        loss = criterion(out, y) + 0.5*dice_loss(out, y)
        loss.backward()
        opt.step()
        running_loss += loss.item()*x.size(0)
    return running_loss / len(loader.dataset)

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval(); tot_dice = 0
    for x, y in tqdm(loader, desc='val', leave=False):
        x, y = x.to(device), y.float().to(device)
        out = model(x)
        dice = dice_coef(torch.sigmoid(out), y)
        tot_dice += dice.item()*x.size(0)
    return tot_dice / len(loader.dataset)

# ──────────────── テスト ────────────────
@torch.no_grad()
def test(model, loader, device):
    model.eval(); dices = []
    for x, y in tqdm(loader, desc='test'):
        x, y = x.to(device), y.float().to(device)
        out = model(x)
        dices.append(dice_coef(torch.sigmoid(out), y).item())
    print(f"[TEST]  Dice: {np.mean(dices):.4f} ± {np.std(dices):.4f}")

# ──────────────── エントリーポイント ────────────────
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # ------ Data ------
    train_ld, val_ld, test_ld = build_loaders(
        'train.csv', 'val.csv', 'test.csv',
        seq_len=8, batch=2
    )

    # ------ Model -----
    model = ConvLSTMUNet(in_ch=3, n_classes=1, base=32).to(device)

    # ------ Loss & Opt ----
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    # ------ Training loop -----
    epochs = 30
    best_dice = 0
    for ep in range(1, epochs+1):
        tr_loss = train_epoch(model, train_ld, criterion, optimizer, device)
        val_dice = eval_epoch(model, val_ld, device)
        scheduler.step()

        print(f"[{ep:02}/{epochs}] train loss {tr_loss:.4f} | val Dice {val_dice:.4f}")
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), 'best_model.pth')

    # ------ Test -----
    model.load_state_dict(torch.load('best_model.pth'))
    test(model, test_ld, device)

if __name__ == "__main__":
    main()
