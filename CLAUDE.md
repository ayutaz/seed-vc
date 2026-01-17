# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

Seed-VCは、ゼロショット音声変換（Voice Conversion）および歌声変換（Singing Voice Conversion）システムです。1〜30秒の参照音声のみで、訓練なしに音声のクローンと変換が可能です。

**主な機能**:
- ゼロショット音声変換（V1/V2モデル）
- リアルタイム音声変換（遅延約300ms）
- 歌声変換（44kHz）
- 少量データでのファインチューニング

## コマンド

### Web UI起動

```bash
# 統合Web UI（すべてのモデル）
python app.py --enable-v1 --enable-v2

# V1音声変換UI
python app_vc.py --fp16 True

# V2音声変換UI
python app_vc_v2.py --compile

# 歌声変換UI
python app_svc.py --fp16 True

# リアルタイムGUI
python real-time-gui.py
```

### コマンドライン推論

```bash
# V1推論
python inference.py --source <source.wav> --target <reference.wav> --output <output_dir> --diffusion-steps 25

# V2推論
python inference_v2.py --source <source.wav> --target <reference.wav> --output <output_dir>
```

### ファインチューニング

#### V1ファインチューニング（CFMベース）

**データ準備**:
- フォーマット: .wav, .mp3, .flac, .ogg, .m4a, .opus
- 長さ: 1〜30秒/ファイル
- データ量: 最小10〜20分、推奨30〜60分
- 品質: クリーンな録音（ノイズ最小化）

**学習コマンド**:
```bash
# 音声変換用（22kHz）
uv run python train.py \
  --config configs/presets/config_dit_mel_seed_uvit_whisper_small_wavenet.yml \
  --dataset-dir <audio_dir> \
  --run-name <name> \
  --batch-size 2 \
  --max-steps 5000 \
  --save-every 500 \
  --num-workers 0

# 歌声変換用（44kHz、F0対応）
uv run python train.py \
  --config configs/presets/config_dit_mel_seed_uvit_whisper_base_f0_44k.yml \
  --dataset-dir <audio_dir> \
  --run-name <name> \
  --batch-size 2 \
  --max-steps 5000 \
  --save-every 500 \
  --num-workers 0
```

**ファインチューニング済みモデルで推論**:
```bash
# 歌声変換
uv run python app_svc.py \
  --checkpoint ./runs/<name>/ft_model.pth \
  --config configs/presets/config_dit_mel_seed_uvit_whisper_base_f0_44k.yml \
  --fp16 True
```

**学習対象コンポーネント**:
| コンポーネント | 学習 | 説明 |
|--------------|------|------|
| CFM | ✅ | メルスペクトログラム生成 |
| Length Regulator | ✅ | トークン→メル対応 |
| CAMPPlus | ❌ | スタイルエンコーダー（固定） |
| Vocoder | ❌ | BigVGAN/HiFiGAN（固定） |
| Whisper | ❌ | コンテンツ抽出（固定） |

#### V2ファインチューニング（マルチGPU）

```bash
accelerate launch train_v2.py \
  --dataset-dir <audio_dir> \
  --run-name <name> \
  --train-cfm \
  --batch-size 4 \
  --max-steps 5000 \
  --num-workers 0
```

### 評価

```bash
python eval.py --source <src_dir> --target <ref_dir> --output <out_dir> --max-samples 100
```

## アーキテクチャ

### V1パイプライン

```
入力音声 → コンテンツ抽出(XLSR/Whisper) → スタイル抽出(CAMPPlus)
         → Duration調整 → DiT(CFM) → Vocoder(HiFiGAN/BigVGAN) → 出力音声
```

### V2パイプライン

```
入力音声 → ASTRAL量子化（話者分離コンテンツ）
         → CFMパス（音色変換）+ ARパス（アクセント/感情）
         → BigVGAN → 出力音声
```

### ディレクトリ構造

- `modules/` - コアMLモジュール
  - `diffusion_transformer.py` - DiT（U-ViT）モデル
  - `flow_matching.py` - CFM訓練ロジック
  - `v2/` - V2モデルコンポーネント（AR、CFM、DiT）
  - `astral_quantization/` - ASTRAL量子化（話者分離）
  - `bigvgan/`, `hifigan/` - Vocoder実装
  - `campplus/` - スピーカーエンコーダー
- `configs/` - YAML設定ファイル
  - `presets/` - 各モデルのプリセット設定
- `data/ft_dataset.py` - ファインチューニングデータセット
- `baselines/` - ベースライン実装（OpenVoice、CosyVoice）

### 利用可能モデル

| モデル | 用途 | SR | パラメータ数 |
|--------|------|------|------------|
| seed-uvit-tat-xlsr-tiny | リアルタイムVC | 22050 | 25M |
| seed-uvit-whisper-small | オフラインVC | 22050 | 98M |
| seed-uvit-whisper-base | 歌声変換 | 44100 | 200M |
| hubert-bsqvae (V2) | VC+アクセント | 22050 | 157M |

モデルは初回実行時にHugging Faceから自動ダウンロードされます。

## 開発上の注意

- **デバイス選択**: CUDA > MPS (Apple Silicon) > CPU の順で自動選択
- **FP16**: 推論時は `--fp16 True` で高速化
- **Windows DataLoader**: `num_workers=0` が必要
- **torch.compile**: V2のARモデルで `--compile` オプションにより約6倍高速化
- **ファインチューニングデータ**: .wav, .flac, .mp3, .m4a, .opus, .ogg形式、1〜30秒/ファイル
- **チェックポイント保存先**: `./runs/<run-name>/ft_model.pth`

## 設定ファイル

主要な設定パラメータ（YAML形式）:

```yaml
preprocess_params:
  sr: 22050/44100  # サンプリングレート

DiT:
  hidden_dim: 384/512/768  # モデル次元
  depth: 9/13/17           # レイヤー数
  f0_condition: true       # 歌声変換時に有効化
```

## インストール

```bash
# Windows/Linux
pip install -r requirements.txt

# macOS (Apple Silicon)
pip install -r requirements-mac.txt

# Windows triton（オプション、コンパイル用）
pip install triton-windows==3.2.0.post13
```
