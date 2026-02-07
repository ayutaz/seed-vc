# V2モデル歌声変換（SVC）対応 実装ガイド

> **ステータス**: 推論コード実装完了、訓練コード・UI未実装（2026年2月現在）
>
> **完了**:
> - `configs/v2/vc_wrapper_svc.yaml` - SVC用設定ファイル（44kHz, F0条件付け）
> - `modules/v2/vc_wrapper.py` - F0抽出・調整・SVC推論メソッド
> - `modules/v2/length_regulator.py` - F0条件付けLength Regulator
> - `tests/test_v2_svc.py` - ユニットテスト（12件）
>
> **未実装**:
> - `train_v2.py` - F0対応の訓練ループ
> - `app_svc_v2.py` - Gradio Web UI
> - SVC用事前学習済みチェックポイント（F0条件付き訓練が必要）

このドキュメントでは、Seed-VC V2モデルを歌声変換（Singing Voice Conversion）に対応させるための実装手順を説明します。

## 目次

1. [V2モデルのV1に対する改善点](#v2モデルのv1に対する改善点)
2. [アーキテクチャ概要](#アーキテクチャ概要)
3. [実装手順](#実装手順)
4. [訓練手順](#訓練手順)
5. [検証方法](#検証方法)
6. [修正ファイル一覧](#修正ファイル一覧)

---

## V2モデルのV1に対する改善点

### 1. ASTRAL量子化による話者分離

| 量子化タイプ | トークン数 | 役割 |
|------------|-----------|------|
| Narrow | 32 | 話者依存コンテンツ |
| Wide | 2048 | 話者非依存コンテンツ |

**効果**: ソース話者の特徴がより完全に除去され、クロス言語変換でのなまりが軽減されます。

### 2. CFG（Classifier-Free Guidance）

```python
# 推論時のパラメータ
inference_cfg_rate = 0.5  # CFGレート（0.0〜1.0）
```

**効果**: CFGレートにより変換品質を調整可能です。

### 3. ARパス（AutoRegressive）

- アクセント・感情・抑揚を個別に調整
- 日本語リファレンスの歌い方をより正確に反映
- ビブラート等の再現性向上

### 4. torch.compile対応

- ARモデルで約6倍の高速化が可能

---

## アーキテクチャ概要

### V2 SVCパイプライン

```
入力歌声 → ASTRAL量子化（話者分離コンテンツ）
         ↓
         F0抽出（RMVPE）→ F0調整（自動/手動ピッチシフト）
         ↓
         CFMパス（音色変換 + F0条件付け）
         ↓
         [オプション] ARパス（アクセント/感情変換）
         ↓
         BigVGAN (44kHz) → 出力歌声
```

### 主要コンポーネント

| コンポーネント | ファイル | 役割 |
|--------------|---------|------|
| VoiceConversionWrapper | `modules/v2/vc_wrapper.py` | 全体のラッパー |
| InterpolateRegulator | `modules/v2/length_regulator.py` | 長さ調整 + F0条件付け |
| DiT | `modules/v2/dit_wrapper.py` | 拡散トランスフォーマー |
| CFM | `modules/v2/cfm.py` | Conditional Flow Matching |
| RMVPE | `modules/rmvpe.py` | F0抽出器 |
| BigVGAN | `modules/bigvgan/` | ボコーダー |

---

## 実装手順

### Phase 1: 設定ファイル作成 ✅ 完了

**ファイル**: `configs/v2/vc_wrapper_svc.yaml`

V1 SVCとの主な変更点:

| パラメータ | V2 VC (22kHz) | V2 SVC (44kHz) |
|-----------|---------------|----------------|
| sr | 22050 | 44100 |
| hop_size | 256 | 512 |
| n_fft | 1024 | 2048 |
| win_size | 1024 | 2048 |
| num_mels | 80 | 128 |
| f0_condition | false | true |
| vocoder | bigvgan_v2_22khz_80band_256x | bigvgan_v2_44khz_128band_512x |

```yaml
_target_: modules.v2.vc_wrapper.VoiceConversionWrapper
sr: 44100
hop_size: 512

mel_fn:
  _target_: modules.audio.mel_spectrogram
  _partial_: true
  n_fft: 2048
  win_size: 2048
  hop_size: 512
  num_mels: 128
  sampling_rate: 44100
  fmin: 0
  fmax: null
  center: False

cfm:
  _target_: modules.v2.cfm.CFM
  estimator:
    _target_: modules.v2.dit_wrapper.DiT
    time_as_token: true
    style_as_token: true
    uvit_skip_connection: false
    block_size: 8192
    depth: 13
    num_heads: 8
    hidden_dim: 512
    in_channels: 128          # 80→128
    content_dim: 512
    style_encoder_dim: 192
    class_dropout_prob: 0.1
    dropout_rate: 0.0
    attn_dropout_rate: 0.0

cfm_length_regulator:
  _target_: modules.v2.length_regulator.InterpolateRegulator
  channels: 512
  is_discrete: true
  codebook_size: 2048
  sampling_ratios: [ 1, 1, 1, 1 ]
  f0_condition: true          # false→true
  n_f0_bins: 512              # 追加

ar_length_regulator:
  _target_: modules.v2.length_regulator.InterpolateRegulator
  channels: 768
  is_discrete: true
  codebook_size: 32
  sampling_ratios: [ ]
  f0_condition: true          # false→true
  n_f0_bins: 512              # 追加

# AR, style_encoder, content_extractorは既存と同じ

vocoder:
  _target_: modules.bigvgan.bigvgan.BigVGAN.from_pretrained
  pretrained_model_name_or_path: "nvidia/bigvgan_v2_44khz_128band_512x"
  use_cuda_kernel: false
```

### 重要な注意事項: 事前学習済みチェックポイント

> **⚠ 現在の制約**: デフォルトの事前学習済みチェックポイント（`v2/cfm_small.pth`）は `f0_condition=false` で訓練されています。SVC用設定（`vc_wrapper_svc.yaml`）では `f0_condition=true` を指定していますが、チェックポイントにはF0エンベディング層の重みが含まれていません。`strict=False` でロードするためエラーにはなりませんが、**F0条件付けは実質的に無効（ランダム初期化のまま）** です。
>
> V2 SVCを実用的に使用するには、F0条件付きでCFMモデルを再訓練する必要があります。

#### F0伝搬のアーキテクチャ

F0情報はパイプライン内で以下のように流れます：

1. **F0抽出**: `extract_f0()` がRMVPEを使用して16kHz音声からフレーム単位のF0を抽出
2. **F0調整**: `adjust_f0()` がソース/ターゲット間のピッチレンジを調整
3. **Length Regulator**: `InterpolateRegulator` でF0がエンベディングに変換され、コンテンツ特徴に加算される
4. **CFM/DiT**: Length Regulatorの出力（F0が統合済み）を受け取る。DiTに直接F0を渡す必要はない（設計意図）

```
F0 → F0 Embedding (n_f0_bins=512) → コンテンツ特徴に加算 → CFM入力
```

### Phase 2: vc_wrapper.pyの修正 ✅ 完了

**ファイル**: `modules/v2/vc_wrapper.py`

実装済みメソッド:
- `_init_f0_extractor()` - RMVPE初期化
- `extract_f0()` - F0抽出
- `adjust_f0()` - F0自動調整・ピッチシフト
- `forward_cfm()` - f0パラメータ追加
- `convert_singing_voice()` - SVC推論メソッド

#### 2.1 RMVPE F0抽出器の初期化追加

```python
class VoiceConversionWrapper(torch.nn.Module):
    def __init__(
            self,
            # ... 既存パラメータ ...
            f0_condition: bool = False,  # 追加
            ):
        super(VoiceConversionWrapper, self).__init__()
        # ... 既存コード ...

        # F0抽出器の初期化
        self.f0_condition = f0_condition
        if f0_condition:
            from modules.rmvpe import RMVPE
            from hf_utils import load_custom_model_from_hf
            rmvpe_path = load_custom_model_from_hf(
                "lj1995/VoiceConversionWebUI", "rmvpe.pt", None
            )
            self.rmvpe = RMVPE(rmvpe_path, is_half=False, device="cpu")
```

#### 2.2 F0抽出メソッドの追加

```python
def extract_f0(self, audio_16k: np.ndarray) -> torch.Tensor:
    """
    RMVPEを使用してF0を抽出

    Args:
        audio_16k: 16kHzにリサンプリングされた音声（numpy配列）

    Returns:
        F0: (T,) - フレーム単位のF0値（torch.Tensor）
    """
    if self.rmvpe is None:
        raise RuntimeError("F0 extractor not initialized. Set f0_condition=True.")
    f0 = self.rmvpe.infer_from_audio(audio_16k, thred=0.03)
    return torch.from_numpy(f0).float()
```

#### 2.3 F0調整メソッドの追加

```python
def adjust_f0(
    self,
    f0_source: torch.Tensor,
    f0_target: torch.Tensor,
    auto_adjust: bool = True,
    pitch_shift: int = 0
) -> torch.Tensor:
    """
    F0をターゲット話者に合わせて調整

    Args:
        f0_source: ソース音声のF0
        f0_target: ターゲット音声のF0
        auto_adjust: 自動でF0レベルを調整するか
        pitch_shift: 半音単位のピッチシフト量

    Returns:
        調整されたF0
    """
    adjusted_f0 = f0_source.clone()

    if auto_adjust:
        # 有声部分のみを使用して中央値を計算
        voiced_source = f0_source[f0_source > 1]
        voiced_target = f0_target[f0_target > 1]

        if len(voiced_source) > 0 and len(voiced_target) > 0:
            median_source = torch.median(torch.log(voiced_source + 1e-5))
            median_target = torch.median(torch.log(voiced_target + 1e-5))

            log_f0 = torch.log(f0_source + 1e-5)
            log_f0[f0_source > 1] = (
                log_f0[f0_source > 1] - median_source + median_target
            )
            adjusted_f0 = torch.exp(log_f0)

    # ピッチシフト適用
    if pitch_shift != 0:
        factor = 2 ** (pitch_shift / 12)
        adjusted_f0[adjusted_f0 > 1] *= factor

    return adjusted_f0
```

#### 2.4 forward_cfmにF0渡し追加

```python
def forward_cfm(self, content_indices_wide, content_lens, mels, mel_lens,
                style_vectors, f0=None):  # f0パラメータ追加
    device = content_indices_wide.device
    B = content_indices_wide.size(0)

    # F0を渡して長さ調整
    cond, _ = self.cfm_length_regulator(
        content_indices_wide,
        ylens=mel_lens,
        f0=f0  # 追加
    )

    # ... 残りは既存コードと同じ ...
```

#### 2.5 convert_voice_with_streamingにF0処理追加

```python
@torch.no_grad()
@torch.inference_mode()
def convert_singing_voice(
        self,
        source_audio_path: str,
        target_audio_path: str,
        diffusion_steps: int = 30,
        length_adjust: float = 1.0,
        inference_cfg_rate: float = 0.5,
        auto_f0_adjust: bool = True,
        pitch_shift: int = 0,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
):
    """
    歌声変換のメインメソッド
    """
    # 音声読み込み
    source_wave = librosa.load(source_audio_path, sr=self.sr)[0]
    target_wave = librosa.load(target_audio_path, sr=self.sr)[0]

    # 16kHzにリサンプリング
    source_wave_16k = librosa.resample(source_wave, orig_sr=self.sr, target_sr=16000)
    target_wave_16k = librosa.resample(target_wave, orig_sr=self.sr, target_sr=16000)

    source_wave_16k_tensor = torch.tensor(source_wave_16k).unsqueeze(0).to(device)
    target_wave_16k_tensor = torch.tensor(target_wave_16k).unsqueeze(0).to(device)

    # F0抽出（numpy配列を渡す）
    if self.f0_condition:
        f0_source = self.extract_f0(source_wave_16k)
        f0_target = self.extract_f0(target_wave_16k)

        # F0調整
        f0_adjusted = self.adjust_f0(
            f0_source, f0_target,
            auto_adjust=auto_f0_adjust,
            pitch_shift=pitch_shift
        )
    else:
        f0_adjusted = None

    # ... コンテンツ抽出、スタイル抽出 ...

    # 長さ調整（F0を渡す）
    cond, _ = self.cfm_length_regulator(
        source_content_indices,
        ylens=source_mel_lens,
        f0=f0_adjusted
    )

    # ... CFM推論、ボコーダー ...
```

### Phase 3: train_v2.pyの修正 ⏳ 未実装

**ファイル**: `train_v2.py`

> **注**: 訓練コードの修正は今後の課題です。現在は推論のみ対応しています。

#### 3.1 F0抽出器の初期化

```python
def _init_main_model(self, train_cfm=True, train_ar=False):
    # ... 既存コード ...

    # F0条件付けの確認と初期化
    f0_condition = self.config.get('cfm_length_regulator', {}).get('f0_condition', False)
    if f0_condition:
        from modules.rmvpe import RMVPE
        from hf_utils import load_custom_model_from_hf
        rmvpe_path = load_custom_model_from_hf(
            "lj1995/VoiceConversionWebUI", "rmvpe.pt", None
        )
        self.rmvpe = RMVPE(rmvpe_path, is_half=False, device=self.device)
        self.f0_condition = True
    else:
        self.f0_condition = False
```

#### 3.2 バッチ処理でF0抽出

```python
def _process_batch(self, epoch, i, batch):
    waves, mels, wave_lens, mel_lens = batch
    waves_16k = torchaudio.functional.resample(waves, self.sr, 16000)
    wave_lengths_16k = (wave_lens.float() * 16000 / self.sr).long()

    # F0抽出
    if self.f0_condition:
        f0_list = []
        for b in range(waves_16k.size(0)):
            f0 = self.rmvpe.infer_from_audio(
                waves_16k[b, :wave_lengths_16k[b]].cpu().numpy(),
                thred=0.03
            )
            f0_list.append(torch.from_numpy(f0).float())
        f0 = torch.nn.utils.rnn.pad_sequence(
            f0_list, batch_first=True
        ).to(self.device)
    else:
        f0 = None

    # Forward pass（f0を渡す）
    with self.accelerator.autocast():
        loss_ar, loss_cfm = self.model(
            waves_16k.to(self.device),
            mels.to(self.device),
            wave_lengths_16k.to(self.device),
            mel_lens.to(self.device),
            forward_ar=self.train_ar,
            forward_cfm=self.train_cfm,
            f0=f0,  # 追加
        )
    # ... 残りは既存コードと同じ ...
```

### Phase 4: Gradio UIの作成 ⏳ 未実装

**ファイル**: `app_svc_v2.py`

> **注**: Gradio UIは今後の課題です。現在はPythonコードから直接`convert_singing_voice()`を呼び出してください。

```python
import os
os.environ['HF_HUB_CACHE'] = './checkpoints/hf_cache'
import gradio as gr
import torch
import yaml
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig
from modules.commons import str2bool
import argparse

# デバイス設定
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

dtype = torch.float16 if device.type == "cuda" else torch.float32

# グローバル変数
vc_wrapper = None
sr = 44100

def load_models(args):
    global vc_wrapper, sr

    config_path = args.config or "configs/v2/vc_wrapper_svc.yaml"
    cfg = DictConfig(yaml.safe_load(open(config_path, "r")))

    vc_wrapper = instantiate(cfg)
    sr = cfg.sr

    # チェックポイント読み込み
    vc_wrapper.load_checkpoints(
        cfm_checkpoint_path=args.checkpoint,
        ar_checkpoint_path=args.ar_checkpoint
    )

    vc_wrapper.to(device)
    vc_wrapper.eval()

    if args.compile:
        vc_wrapper.compile_ar()

@torch.no_grad()
@torch.inference_mode()
def convert_singing_voice(
    source, target,
    diffusion_steps, length_adjust,
    inference_cfg_rate,
    auto_f0_adjust, pitch_shift
):
    # 変換処理
    result = vc_wrapper.convert_singing_voice(
        source_audio_path=source,
        target_audio_path=target,
        diffusion_steps=diffusion_steps,
        length_adjust=length_adjust,
        inference_cfg_rate=inference_cfg_rate,
        auto_f0_adjust=auto_f0_adjust,
        pitch_shift=pitch_shift,
        device=device,
        dtype=dtype,
    )

    return (sr, result)

def main(args):
    load_models(args)

    description = """
    V2モデルによる歌声変換（Singing Voice Conversion）

    **特徴**:
    - ASTRAL量子化による高精度な話者分離
    - CFGレートによる変換品質の調整
    - F0条件付けによる正確なピッチ再現
    """

    inputs = [
        gr.Audio(type="filepath", label="Source Audio (歌声)"),
        gr.Audio(type="filepath", label="Reference Audio (参照音声)"),
        gr.Slider(1, 100, value=30, step=1, label="Diffusion Steps"),
        gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="Length Adjust"),
        gr.Slider(0.0, 1.0, value=0.5, step=0.1, label="Inference CFG Rate"),
        gr.Checkbox(label="Auto F0 Adjust", value=True),
        gr.Slider(-24, 24, value=0, step=1, label="Pitch Shift (semitones)"),
    ]

    outputs = gr.Audio(label="Output Audio", format='wav')

    gr.Interface(
        fn=convert_singing_voice,
        inputs=inputs,
        outputs=outputs,
        title="Seed-VC V2 Singing Voice Conversion",
        description=description,
    ).launch(share=args.share)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--ar-checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--share", type=str2bool, default=False)
    parser.add_argument("--compile", type=str2bool, default=False)
    args = parser.parse_args()
    main(args)
```

---

## 訓練手順

### 1. データ準備

| 項目 | 要件 |
|------|------|
| 形式 | .wav, .flac, .mp3 |
| サンプリングレート | 44kHz推奨 |
| ファイル長 | 1〜30秒 |
| 内容 | 歌声（話し声ではない） |
| 最小データ量 | 10時間 |
| 推奨データ量 | 50時間以上 |
| 言語 | 日本語歌声を含むこと |

### 2. 訓練コマンド

#### CFM訓練（F0条件付け）

```bash
accelerate launch train_v2.py \
  --config configs/v2/vc_wrapper_svc.yaml \
  --dataset-dir /path/to/singing_dataset \
  --run-name svc_v2_cfm \
  --train-cfm \
  --max-steps 100000 \
  --batch-size 4 \
  --save-every 5000
```

#### AR訓練（オプション、アクセント変換用）

```bash
accelerate launch train_v2.py \
  --config configs/v2/vc_wrapper_svc.yaml \
  --pretrained-cfm-ckpt ./runs/svc_v2_cfm/CFM_epoch_*_step_*.pth \
  --dataset-dir /path/to/singing_dataset \
  --run-name svc_v2_ar \
  --train-ar \
  --max-steps 50000 \
  --batch-size 4
```

### 3. 訓練リソース

| リソース | 推奨 | 最低 |
|---------|------|------|
| GPU | 8x A100 (80GB) | 1x A100 or 24GB VRAM |
| CFM訓練時間 | 2〜3日 | 1〜2週間 |
| AR訓練時間 | 1〜2日 | 3〜5日 |

---

## 検証方法

### 1. 訓練中の確認

```bash
# ログ確認
tail -f runs/svc_v2_cfm/training.log

# Loss目安
# CFM loss < 0.5 が収束の目安
```

### 2. 推論テスト

```bash
# コマンドライン
python inference_v2.py \
  --source english_song.wav \
  --target japanese_reference.wav \
  --output ./output \
  --cfm-checkpoint-path ./runs/svc_v2_cfm/CFM_*.pth \
  --f0-condition \
  --auto-f0-adjust

# Web UI
python app_svc_v2.py --checkpoint ./runs/svc_v2_cfm/CFM_*.pth
```

### 3. 評価指標

| 指標 | 目標値 | 説明 |
|------|--------|------|
| MCD | < 7.0 dB | Mel Cepstral Distortion |
| F0 RMSE | < 50 Hz | ピッチ精度 |
| MOS | > 3.5 | 主観評価（5段階） |

---

## 修正ファイル一覧

| ファイル | 変更内容 | ステータス |
|---------|---------|--------|
| `configs/v2/vc_wrapper_svc.yaml` | 新規作成：44kHz、F0条件付け設定 | ✅ 完了 |
| `modules/v2/vc_wrapper.py` | RMVPE統合、F0抽出・調整メソッド追加 | ✅ 完了 |
| `tests/test_v2_svc.py` | 新規作成：V2 SVCユニットテスト | ✅ 完了 |
| `train_v2.py` | F0抽出・訓練ループ修正 | ⏳ 未実装 |
| `app_svc_v2.py` | 新規作成：SVC用Gradio UI | ⏳ 未実装 |

**注**: `modules/v2/length_regulator.py`にはすでにF0条件付け機能が実装されています（`f0_condition`パラメータ）。設定ファイルで`f0_condition: true`を指定するだけで有効化されます。

## クイックスタート（推論）

> **⚠ 注意**: デフォルトのチェックポイントはF0条件付けなしで訓練されています。以下のコードは構造的には動作しますが、F0条件付けの効果を得るにはF0対応のチェックポイントで再訓練が必要です。現時点ではSVC専用の事前学習済みチェックポイントは公開されていません。

```python
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
import yaml

# 設定読み込み
cfg = DictConfig(yaml.safe_load(open("configs/v2/vc_wrapper_svc.yaml")))
wrapper = instantiate(cfg)

# チェックポイント読み込み
wrapper.load_checkpoints()

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wrapper.to(device)
wrapper.eval()

# SVC推論
result = wrapper.convert_singing_voice(
    source_audio_path="source_song.wav",
    target_audio_path="reference_voice.wav",
    diffusion_steps=30,
    inference_cfg_rate=0.5,
    auto_f0_adjust=True,
    pitch_shift=0,  # 半音単位
    device=device,
)
```

## テスト実行

```bash
uv run pytest tests/test_v2_svc.py -v
```

---

## 期待される結果

V2 SVC訓練後、以下の改善が期待されます：

1. **声質変換精度向上**: ASTRAL話者分離により元話者の特徴がより除去される
2. **クロス言語品質向上**: 英語→日本語でのなまりが軽減
3. **ピッチ精度向上**: F0条件付けによる正確なピッチ再現（F0対応チェックポイントで訓練後）
4. **表現力向上**: ARパスによるアクセント・感情の詳細な再現
5. **柔軟な調整**: CFGレートによる変換品質の調整

---

## 関連ドキュメント

- [JAPANESE_SVC_OPTIMIZATION.md](./JAPANESE_SVC_OPTIMIZATION.md) - 日本語SVC最適化
- [SVC_SOTA_RESEARCH.md](./SVC_SOTA_RESEARCH.md) - SVC最新研究
- [RESEARCH.md](./RESEARCH.md) - 基礎研究
