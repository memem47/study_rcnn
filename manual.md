### datapath.csv
- clip_id : 同じ動画クリップを識別するキー
- frame_idx : クリップ内のフレーム番号（昇順で並んでいると楽）
- image_path / mask_path : RGB PNG とラベルマスク PNG（整数 0/1/…）

### 使い方メモ
1. CSV を作る
- `train.csv`, `val.csv`, `test.csv` に上記フォーマットで書き出す
- クリップ長（`seq_len`）より短いクリップは除外してください
2. 画像サイズ
- モデル実装は 2 回の MaxPool 前提なので、`H, W` が 4 の倍数だとスムーズです
3. 学習／推論
```bash
python train_conv_lstm_unet.py # CUDA があれば自動使用`
```
4. 自前モデルを差し替える場合
- `model_conv_lstm_unet.py` にクラスを置き換えるだけで OK

これで CSV＋フォルダ構成 さえ用意すれば、ConvLSTM-U-Net を学習 → 検証 → テストまで一括で回せます。あとは `seq_len` や `base` チャネル数を目的・GPU メモリに応じて調整してみてください。

### データ拡張
