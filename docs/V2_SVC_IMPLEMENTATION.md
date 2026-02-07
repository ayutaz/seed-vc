# V2 SVC（歌声変換）実装ドキュメント

> **最終更新**: 2026年2月
>
> | 機能 | ステータス |
> |------|----------|
> | 設定ファイル (`configs/v2/vc_wrapper_svc.yaml`) | ✅ 完了 |
> | 推論コード (`modules/v2/vc_wrapper.py`) | ✅ 完了 |
> | Length Regulator F0対応 (`modules/v2/length_regulator.py`) | ✅ 完了 |
> | 訓練コード (`train_v2.py`) | ✅ 完了 |
> | テスト (`tests/test_v2_svc.py`) | ✅ 完了 (12件) |
> | Web UI (`app_svc_v2.py`) | ✅ 完了 |
> | SVC専用チェックポイント | ⏳ 未公開（F0条件付き訓練が必要） |

---

## 目次

1. [アーキテクチャ](#アーキテクチャ)
2. [V2 VC との設定差分](#v2-vc-との設定差分)
3. [F0パイプライン](#f0パイプライン)
4. [推論](#推論)
5. [訓練](#訓練)
6. [テスト](#テスト)
7. [チェックポイントに関する注意](#チェックポイントに関する注意)
8. [ファイル一覧](#ファイル一覧)
9. [関連ドキュメント](#関連ドキュメント)

---

## アーキテクチャ

### V2 SVCパイプライン

```
入力歌声 (44kHz)
  │
  ├─→ 16kHzリサンプリング
  │     ├─→ ASTRAL量子化（Wide: 2048トークン → 話者非依存コンテンツ）
  │     ├─→ ASTRAL量子化（Narrow: 32トークン → ARパス用）
  │     ├─→ RMVPE → F0抽出 → F0調整（自動/ピッチシフト）
  │     └─→ CAMPPlus → スタイルベクトル
  │
  ├─→ Mel spectrogram (128bands, 2048FFT, 512hop)
  │
  └─→ CFMパス: InterpolateRegulator(content + F0) → DiT → Mel生成
       [ARパス: InterpolateRegulator(narrow + F0) → AR → Wide tokens]
            │
            └─→ BigVGAN (44kHz) → 出力歌声
```

### 主要コンポーネント

| コンポーネント | ファイル | 役割 |
|--------------|---------|------|
| VoiceConversionWrapper | `modules/v2/vc_wrapper.py` | 全体のラッパー（推論・訓練） |
| InterpolateRegulator | `modules/v2/length_regulator.py` | 長さ調整 + F0条件付け |
| RMVPE | `modules/rmvpe.py` | F0抽出器（16kHz入力） |
| DiT | `modules/v2/dit_wrapper.py` | 拡散トランスフォーマー |
| CFM | `modules/v2/cfm.py` | Conditional Flow Matching |
| NaiveTransformer | `modules/v2/ar.py` | AutoRegressiveモデル |
| BigVGAN | `modules/bigvgan/` | ボコーダー（44kHz出力） |
| CAMPPlus | `modules/campplus/` | スピーカーエンコーダー |

### V2のV1に対する主な改善点

| 改善点 | 説明 |
|--------|------|
| ASTRAL量子化 | Narrow(32)/Wide(2048)トークンで話者特徴をより完全に分離 |
| CFG | Classifier-Free Guidance で変換品質を調整可能 |
| ARパス | アクセント・感情・抑揚を個別に制御 |
| torch.compile | ARモデルで約6倍の高速化 |

---

## V2 VC との設定差分

| パラメータ | V2 VC (`vc_wrapper.yaml`) | V2 SVC (`vc_wrapper_svc.yaml`) |
|-----------|--------------------------|-------------------------------|
| sr | 22050 | 44100 |
| hop_size | 256 | 512 |
| n_fft | 1024 | 2048 |
| win_size | 1024 | 2048 |
| num_mels | 80 | 128 |
| f0_condition (CFM) | false | true |
| f0_condition (AR) | false | true |
| n_f0_bins | - | 512 |
| in_channels (DiT) | 80 | 128 |
| vocoder | bigvgan_v2_22khz_80band_256x | bigvgan_v2_44khz_128band_512x |

AR、style_encoder、content_extractorの構成は共通です。

---

## F0パイプライン

### データフロー

```
音声 (16kHz numpy)
  │
  ▼
RMVPE.infer_from_audio(audio, thred=0.03)
  │  → numpy array (T,) [Hz単位、無声=0]
  ▼
torch.from_numpy(f0).float()  → Tensor (T,)
  │
  ▼ [推論時のみ]
adjust_f0(f0_source, f0_target, auto_adjust, pitch_shift)
  │  → log F0中央値マッチング + 半音シフト
  ▼
InterpolateRegulator.forward(x, ylens, f0)
  │  1. f0_to_coarse(f0, n_f0_bins=512)  [Hz → 離散ビン]
  │  2. f0_embedding(quantized_f0)        [ビン → 埋め込みベクトル]
  │  3. interpolate → target length
  │  4. content_embedding + f0_embedding  [加算]
  ▼
CFM / AR への入力（F0統合済み）
```

### F0が`None`の場合

`InterpolateRegulator`は`f0_condition=true`かつ`f0=None`のとき、学習可能な`f0_mask`パラメータを代わりに加算します。これにより、推論時にF0なしでも動作します（品質は低下）。

---

## 推論

### Python API

```python
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
import yaml

# モデル初期化
cfg = DictConfig(yaml.safe_load(open("configs/v2/vc_wrapper_svc.yaml")))
wrapper = instantiate(cfg)
wrapper.load_checkpoints()

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
    pitch_shift=0,       # 半音単位（-24〜+24）
    device=device,
)
```

### VoiceConversionWrapper の主要メソッド

| メソッド | 用途 | F0使用 |
|---------|------|--------|
| `convert_singing_voice()` | SVC推論（F0条件付き） | ✅ |
| `convert_timbre()` | 音色のみ変換（CFMパス） | ❌ |
| `convert_voice()` | AR+CFM音声変換 | ❌ |
| `convert_voice_with_streaming()` | ストリーミング音声変換 | ❌ |
| `extract_f0(audio_16k)` | F0抽出（numpy→Tensor） | - |
| `adjust_f0(src, tgt, ...)` | F0自動調整・ピッチシフト | - |
| `forward(waves_16k, mels, ..., f0=None)` | 訓練用forward | ✅ (optional) |

---

## 訓練

### コマンド

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

#### AR訓練（オプション）

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

### CLI引数一覧

| 引数 | デフォルト | 説明 |
|------|----------|------|
| `--config` | `configs/v2/vc_wrapper.yaml` | 設定ファイルパス |
| `--pretrained-cfm-ckpt` | None | CFMチェックポイント |
| `--pretrained-ar-ckpt` | None | ARチェックポイント |
| `--dataset-dir` | (必須) | データセットディレクトリ |
| `--run-name` | (必須) | 実験名（`./runs/<name>/`に保存） |
| `--batch-size` | 2 | バッチサイズ |
| `--max-steps` | 1000 | 最大訓練ステップ数 |
| `--max-epochs` | 1000 | 最大エポック数 |
| `--save-every` | 500 | チェックポイント保存間隔 |
| `--num-workers` | 0 | DataLoaderワーカー数 |
| `--train-cfm` | flag | CFMモデルを訓練 |
| `--train-ar` | flag | ARモデルを訓練 |

### 訓練時のF0処理フロー

`vc_wrapper_svc.yaml`（`f0_condition: true`）を指定すると、以下が自動的に有効化されます:

1. **`_init_main_model()`**: `model.f0_condition`属性を検出し、`self.f0_condition = True`を設定
2. **`_process_batch()`**: 各バッチで`_extract_f0_batch()`を呼び出し
3. **`_extract_f0_batch()`**: バッチ内の各音声に対してRMVPEでF0を抽出し、`pad_sequence`でパディング
4. **`model.forward(..., f0=f0)`**: F0テンソルが`forward_cfm()`と`forward_ar()`に伝搬
5. **`InterpolateRegulator`**: F0を量子化・埋め込みし、コンテンツ特徴に加算

通常VC設定（`f0_condition: false`）では`f0=None`が渡され、従来通りの動作となります。

### データ要件

| 項目 | 要件 |
|------|------|
| 形式 | .wav, .flac, .mp3, .m4a, .opus, .ogg |
| サンプリングレート | 44kHz推奨（自動リサンプリング） |
| ファイル長 | 1〜30秒 |
| 最小データ量 | 10時間 |
| 推奨データ量 | 50時間以上 |

### チェックポイント形式

保存先: `./runs/<run-name>/`

```
CFM_epoch_XXXXX_step_XXXXX.pth
  └─ net:
       ├─ cfm: CFMモデルの状態辞書
       └─ length_regulator: CFM Length Regulatorの状態辞書

AR_epoch_XXXXX_step_XXXXX.pth
  └─ net:
       ├─ ar: ARモデルの状態辞書
       └─ length_regulator: AR Length Regulatorの状態辞書
```

### 推奨リソース

| リソース | 推奨 | 最低 |
|---------|------|------|
| GPU | 8x A100 (80GB) | 1x 24GB VRAM |
| CFM訓練 | 2〜3日 | 1〜2週間 |
| AR訓練 | 1〜2日 | 3〜5日 |

---

## テスト

```bash
uv run pytest tests/test_v2_svc.py -v
```

### テスト一覧 (12件)

| クラス | テスト | 検証内容 |
|--------|--------|---------|
| TestF0Extraction | `test_extract_f0_returns_tensor` | F0抽出の戻り値型 |
| TestF0Extraction | `test_extract_f0_raises_when_rmvpe_not_initialized` | RMVPE未初期化時のエラー |
| TestF0Adjustment | `test_adjust_f0_auto_adjust` | 自動ピッチレンジ調整 |
| TestF0Adjustment | `test_adjust_f0_pitch_shift` | 半音シフト（12半音=1オクターブ） |
| TestF0Adjustment | `test_adjust_f0_no_adjustment` | 無調整時の恒等性 |
| TestLengthRegulatorF0 | `test_length_regulator_receives_f0` | F0パラメータの受け渡し |
| TestLengthRegulatorF0 | `test_length_regulator_f0_none_uses_mask` | F0=None時のf0_mask使用 |
| TestVCWrapperInitialization | `test_init_with_f0_condition_true` | f0_condition=true時の初期化 |
| TestVCWrapperInitialization | `test_init_with_f0_condition_false` | f0_condition=false時の初期化 |
| TestConvertSingingVoice | `test_convert_singing_voice_raises_when_f0_disabled` | F0無効時のエラー |
| TestF0ToCoarse | `test_f0_to_coarse_basic` | F0→離散ビン変換 |
| TestF0ToCoarse | `test_f0_to_coarse_unvoiced` | 無声フレーム（F0=0）の処理 |

---

## チェックポイントに関する注意

> **制約**: デフォルトの事前学習済みチェックポイント（`v2/cfm_small.pth`）は`f0_condition=false`で訓練されています。
>
> SVC設定で読み込むとF0エンベディング層（`f0_embedding`）の重みが欠落しますが、`strict=False`でロードするためエラーにはなりません。ただし**F0条件付けはランダム初期化のまま実質無効**です。
>
> V2 SVCを実用的に使用するには、`vc_wrapper_svc.yaml`でCFMモデルをF0条件付きで再訓練する必要があります。

---

## ファイル一覧

| ファイル | 役割 | ステータス |
|---------|------|----------|
| `configs/v2/vc_wrapper_svc.yaml` | SVC用設定（44kHz, F0条件付け） | ✅ |
| `configs/v2/vc_wrapper.yaml` | 通常VC用設定（22kHz, F0なし） | ✅ |
| `modules/v2/vc_wrapper.py` | 推論・訓練ラッパー（F0対応） | ✅ |
| `modules/v2/length_regulator.py` | Length Regulator（F0埋め込み） | ✅ |
| `modules/rmvpe.py` | RMVPE F0抽出器 | ✅ |
| `train_v2.py` | V2訓練スクリプト（F0対応） | ✅ |
| `tests/test_v2_svc.py` | V2 SVCユニットテスト（12件） | ✅ |
| `app_svc_v2.py` | V2 SVC用Web UI（Gradio） | ✅ |

---

## 関連ドキュメント

- [JAPANESE_SVC_OPTIMIZATION.md](./JAPANESE_SVC_OPTIMIZATION.md) - 日本語SVC最適化
- [SVC_SOTA_RESEARCH.md](./SVC_SOTA_RESEARCH.md) - SVC最新研究
- [RESEARCH.md](./RESEARCH.md) - 基礎研究
