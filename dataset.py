# dataset.py ---------------------------------------------------------------
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

class VideoSegDataset(Dataset):
    """
    CSV で与えた (image_path, mask_path) を clip_id ごとにまとめて
    連続 T フレームを返す Dataset
    """
    def __init__(self, csv_file, seq_len=8, transform=None):
        self.df = pd.read_csv(csv_file)
        self.seq_len = seq_len
        self.transform = transform
        
        # clip_id → [row_indices …] を作成
        self.index_map = []
        for clip_id, grp in self.df.groupby('clip_id'):
            grp = grp.sort_values('frame_idx')
            frames = grp.to_dict('records')
            # スライディングウィンドウ (stride = 1)
            for i in range(len(frames) - seq_len + 1):
                self.index_map.append(frames[i:i+seq_len])

    def __len__(self):
        return len(self.index_map)

    def _load_img(self, path):
        img = Image.open(path).convert('RGB')
        return torch.from_numpy(np.array(img)).permute(2,0,1).float() / 255.

    def _load_mask(self, path):
        msk = Image.open(path)
        return torch.from_numpy(np.array(msk, dtype='int64')).unsqueeze(0)  # (1,H,W)

    def __getitem__(self, idx):
        frames = self.index_map[idx]
        imgs, masks = [], []
        for row in frames:
            img = self._load_img(row['image_path'])
            msk = self._load_mask(row['mask_path'])
            if self.transform:
                img = self.transform(img)
            imgs.append(img)
            masks.append(msk)
        # Stack to (T,C,H,W)
        imgs  = torch.stack(imgs,  dim=0)
        masks = torch.stack(masks, dim=0)
        return imgs, masks        # 返り値: (T,3,H,W), (T,1,H,W)

# dataloader ---------------------------------------------------------------
def build_loaders(csv_train, csv_val, csv_test, seq_len=8, batch=2, num_workers=4):
    train_ds = VideoSegDataset(csv_train, seq_len)
    val_ds   = VideoSegDataset(csv_val,   seq_len)
    test_ds  = VideoSegDataset(csv_test,  seq_len)

    train_ld = DataLoader(train_ds, batch_size=batch, shuffle=True,
                          num_workers=num_workers, pin_memory=True)
    val_ld   = DataLoader(val_ds,   batch_size=batch, shuffle=False,
                          num_workers=num_workers, pin_memory=True)
    test_ld  = DataLoader(test_ds,  batch_size=1,    shuffle=False,
                          num_workers=num_workers, pin_memory=True)  # テストは 1 clip ずつ
    return train_ld, val_ld, test_ld
