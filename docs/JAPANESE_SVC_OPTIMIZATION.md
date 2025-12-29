# Seed-VC 日本語歌声変換（SVC）最適化ガイド

## 概要

本ドキュメントは、Seed-VCを日本語歌声変換向けに最適化するための調査結果と実装ガイドです。

**対象環境**: T4/A10 GPU (16GB VRAM)

---

## 目次

1. [重要な注意事項](#1-重要な注意事項)
2. [改善オプション一覧](#2-改善オプション一覧)
3. [コンテンツエンコーダ比較](#3-コンテンツエンコーダ比較)
4. [代替アーキテクチャ](#4-代替アーキテクチャ)
5. [日本語歌声データセット](#5-日本語歌声データセット)
6. [大規模データセット収集方法](#6-大規模データセット収集方法)
7. [GPU要件](#7-gpu要件)
8. [ファインチューニング手順](#8-ファインチューニング手順)
9. [日本語HuBERT統合ガイド](#9-日本語hubert統合ガイド)
10. [SVCC 2025知見](#10-svcc-2025知見)
11. [参考リンク集](#11-参考リンク集)

---

## 1. 重要な注意事項

### rinna/japanese-hubert-base は単純な置き換えでは機能しない

**結論**: rinna/japanese-hubert-baseをWhisperの代わりに設定するだけでは精度は向上しません。**再学習が必要です**。

#### 理由

Seed-VCの事前学習済みDiTモデルは、**Whisper-small**または**XLSR**の特徴量で学習されています。

```
事前学習済みチェックポイント
  └── content_in_proj層（Whisper特徴量→内部表現への変換）
      └── Whisper-smallの出力パターンに最適化された重み
```

rinna/japanese-hubert-baseに置き換えると：
- 出力次元は同じ（768次元）だが
- **特徴量の意味が異なる**ため、既存の重みが機能しない

#### 対応方法

| アプローチ | 効果 | 必要なステップ数 | データ量 |
|-----------|------|-----------------|---------|
| 設定だけ変更 | ❌ 動作しない | - | - |
| ファインチューニング | ⚠️ 限定的な効果 | 2,000-5,000 | 2時間+ |
| **スクラッチから再学習** | ✅ 推奨 | 5,000-10,000 | 50時間+ |

#### 推奨アプローチ

**Phase 1（短期）**: 既存のWhisper-smallモデルで日本語データをファインチューニング
- 最も簡単で効果的
- 2,000ステップで十分なデモ品質

**Phase 2（長期）**: 大量の日本語データがある場合
- rinna/japanese-hubert-baseでスクラッチから学習
- 50時間以上のデータ推奨
- 5,000-10,000ステップ

---

## 2. 改善オプション一覧

### 2.1 効果と難易度の比較

| オプション | 効果 | 難易度 | 必要データ | 推奨度 |
|-----------|------|--------|-----------|--------|
| **日本語データでFT** | 高 | 低 | 2時間 | ⭐⭐⭐ |
| ContentVec置換 | 高 | 高 | 50時間+ | ⭐⭐ |
| japanese-hubert置換 | 高 | 高 | 50時間+ | ⭐ |
| スピーカーエンコーダ改善 | 中 | 中 | - | ⭐ |

### 2.2 コンテンツエンコーダーの改善（効果: 高）

| オプション | 説明 | 難易度 | 期待効果 |
|-----------|------|--------|----------|
| **日本語データでFT** | 既存Whisperモデルを日本語データで追加学習 | 低 | 高（推奨） |
| ContentVec置換 | 話者分離性能が最高のエンコーダに変更 | 高 | 高 |
| rinna/japanese-hubert | 19,000時間の日本語音声で学習済み | 高 | 高（要再学習） |

### 2.3 スピーカーエンコーダー改善（効果: 中）

| オプション | 説明 | 日本語対応 |
|-----------|------|------------|
| CAMPPlus（現行） | 中国語・英語中心で学習 | 限定的 |
| OpenVoice V2 SE | 日本語ネイティブサポート | 良好 |
| ECAPA-TDNN | マルチリンガル対応 | 良好 |

**推奨**: 短期的にはCAMPPlusで十分。長期的にはECAPA-TDNNへの置換を検討。

---

## 3. コンテンツエンコーダ比較

### 3.1 SVCC 2023 比較結果

| エンコーダ | 明瞭度 | 話者分離 | Voice ID精度 | 推奨用途 |
|-----------|--------|---------|--------------|---------|
| **ContentVec** | ◎ | ◎ | 37.7% | SVC全般（最推奨） |
| HuBERT | ◎ | △ | 73.7% | 汎用（話者情報残る） |
| HuBERT-soft | ○ | △ | - | So-VITS-SVC |
| Whisper | ○ | ○ | - | 多言語対応（Seed-VC） |
| WavLM | ○ | ○ | - | 大規模モデル |

**注**: Voice ID精度が低いほど話者情報が除去されており、SVCに適している

### 3.2 各エンコーダの特徴

| エンコーダ | 出力次元 | 学習データ | 特徴 |
|-----------|---------|-----------|------|
| ContentVec | 256 | LibriSpeech | HuBERT + speaker-invariant tweaks |
| HuBERT-base | 768 | LibriSpeech | Meta製、広く使用 |
| Whisper-small | 768 | 680k時間 | OpenAI、多言語対応 |
| WavLM-Large | 1024 | 94k時間 | Microsoft、大規模 |
| rinna/japanese-hubert | 768 | 19k時間日本語 | 日本語特化 |

### 3.3 推奨選択

- **Zero-shot SVC**: ContentVec（話者分離性能最高）
- **多言語対応**: Whisper（Seed-VCの現行選択）
- **日本語特化**: rinna/japanese-hubert（要再学習）

---

## 4. 代替アーキテクチャ

### 4.1 主要アーキテクチャ比較（2024-2025）

| モデル | 生成方式 | コンテンツ抽出 | Zero-shot | 公開状況 |
|--------|---------|---------------|-----------|---------|
| **Seed-VC** | Flow Matching + DiT | Whisper/XLSR | ✅ | [OSS (GPL v3)](https://github.com/Plachtaa/seed-vc) |
| **Vevo1.5** | ARLM + Flow Matching | BPE + Chromagram | ✅ | [Amphion](https://github.com/open-mmlab/Amphion) |
| **RIFT-SVC** | Rectified Flow + DiT | ContentVec | ❌ | [OSS](https://github.com/Pur1zumu/RIFT-SVC) |
| **HQ-SVC** | Unified Codec | Multi-feature | ✅ | [推論のみ](https://github.com/ShawnPi233/HQ-SVC) |
| **SaMoye** | VITS | 複数ASR融合 | ✅ | [OSS](https://github.com/CarlWangChina/SaMoye-SVC) |

### 4.2 Seed-VCより優れている可能性があるアーキテクチャ

#### Vevo1.5 (ICLR 2025) - SVCC 2025トップ

```
入力 → BPE + Chromagram Tokenizer
     → Autoregressive LM → Content-Style Tokens
     → Flow Matching Transformer → Mel
     → Vocoder → 出力
```

| 項目 | 詳細 |
|------|------|
| 構成 | ARLM + Flow Matching Transformer |
| 学習データ | 7,000時間の歌声（SingNet） |
| 優位点 | 歌声スタイル変換対応、SVCC 2025でトップ |
| 制限 | 大規模データ必要、Amphion経由 |
| リンク | https://github.com/open-mmlab/Amphion |

#### RIFT-SVC V3 (2025)

| 項目 | 詳細 |
|------|------|
| 構成 | Rectified Flow + DiT + ContentVec |
| 優位点 | Multiple CFGでarticulation/timbre個別制御 |
| 制限 | Zero-shot非対応（ファインチューニング前提） |
| リンク | https://github.com/Pur1zumu/RIFT-SVC |

### 4.3 結論

**Zero-shot SVCではSeed-VCが依然として最良の選択肢。**

理由:
1. 完全なZero-shot対応
2. リアルタイム推論可能（~300ms遅延）
3. GPL v3ライセンス（商用利用可、派生物も同ライセンス）
4. 44kHz歌声変換対応
5. ファインチューニング対応

---

## 5. 日本語歌声データセット

### 5.1 主要データセット一覧

| データセット | 内容 | 時間 | ライセンス | 商用利用 |
|-------------|------|------|-----------|----------|
| **JVS-MuSiC** | 100人の歌手 | ~2時間 | CC BY-NC-SA 4.0 | ❌ |
| **Kiritan** | J-POP 50曲 | ~57分 | 非商用のみ | ❌ |
| **JSUT-Song** | 単一話者歌声 | ~数時間 | CC BY-SA 4.0 | ✅ |
| **GTSinger** | 多言語・多スタイル | 80時間 | CC BY-NC-SA | ❌ |

### 5.2 データセット別入手方法

#### JVS-MuSiC

| 項目 | 詳細 |
|------|------|
| 公式ページ | https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_music |
| 内容 | 100人の歌手による童謡「かたつむり」+ 各自選曲1曲 |
| 形式 | WAV 48kHz 16bit mono |

#### 東北きりたん歌唱データベース

| 項目 | 詳細 |
|------|------|
| ダウンロード | https://zunko.jp/kiridev/login.php |
| アクセス | Facebookログイン必要 |
| ラベルデータ | https://github.com/mmorise/kiritan_singing |
| 利用条件 | 非商用のみ、クレジット「©SSS」表示必須 |

#### JSUT-Song（商用利用可能）

| 項目 | 詳細 |
|------|------|
| リポジトリ | https://www.nii.ac.jp/dsc/idr/speech/ |
| ライセンス | CC BY-SA 4.0（商用利用可） |

### 5.3 商用利用に関する注意事項

ほとんどの日本語歌声データセットは**非商用（研究目的）のみ**です。

BtoB商用デモの場合：
1. **JSUT-Songベース** - CC BY-SA 4.0で商用利用可能
2. **自社録音データ** - 完全な権利を保持
3. **著作権法30条の4** - 学習目的での利用（詳細は次セクション）

---

## 6. 大規模データセット収集方法

### 6.1 収集規模の可能性

| 方法 | 規模 | 難易度 | 商用利用 |
|------|------|--------|---------|
| **SingNet方式（Web収集）** | 数千時間可能 | 高 | 学習のみ可 |
| **パブリックドメイン** | 数十時間 | 低 | ✅ |
| **ボカロ・歌い手許諾** | 数百時間可能 | 中 | 要交渉 |
| **合成音声** | 無制限 | 低 | 要確認 |
| **自社録音** | 制限なし | 高コスト | ✅ |

### 6.2 著作権法30条の4（日本）

| 用途 | 適用可否 | 条件 |
|------|---------|------|
| **研究開発** | ✅ 適用可能 | 「著作物の表現を享受しない」利用 |
| **商用サービス学習** | ⚠️ 条件付き | 学習結果の利用は可能、データ再配布は不可 |
| **データセット公開** | ❌ 困難 | 著作権者の許諾必要 |

**重要**: 歌声には**著作権**だけでなく**著作隣接権**（歌手・レコード会社の権利）も存在

### 6.3 SingNet方式パイプライン

[SingNet](https://arxiv.org/abs/2505.09325)は以下のパイプラインで3,000時間の歌声を収集：

```
1. Web収集（YouTube/ニコ動/音楽ストリーミング）
     ↓
2. 音源分離（Demucs等で歌声抽出）
     ↓
3. 品質スコアリング（MOS予測モデルでフィルタ）
     ↓
4. 歌詞アライメント（Whisper等）
     ↓
5. 言語・ジャンル分類
```

**日本語歌声の期待収集量**: 500-1,000時間（研究目的）

### 6.4 現実的な収集オプション

#### 研究開発の場合
- SingNet方式で日本語500-1,000時間収集可能
- 著作権法30条の4に基づく利用

#### BtoB商用の場合
- JSUT-Song + 自社録音で50-100時間
- 許諾ベースのパートナーシップ

---

## 7. GPU要件

### 7.1 推奨スペック

| GPU | VRAM | 対応状況 | バッチサイズ目安 |
|-----|------|---------|-----------------|
| RTX 3060 | 12GB | 推論可能、学習は厳しい | 1 |
| **T4** | 16GB | 公式確認済み | 1-2 |
| **A10** | 24GB | 推奨 | 2-4 |
| RTX 3090/4090 | 24GB | 最推奨 | 4-8 |

### 7.2 メモリ使用量の目安

| 処理 | T4 (16GB) | RTX 3090 (24GB) |
|------|-----------|-----------------|
| 推論（1サンプル） | ~8GB | ~8GB |
| 学習（batch_size=1） | ~14GB | ~14GB |
| 学習（batch_size=2） | ~16GB（ギリギリ） | ~18GB |

**推奨**: T4/A10では`batch_size=1`を使用

### 7.3 学習時間の見積もり

| GPU | 1000ステップ | 2000ステップ | 5000ステップ |
|-----|-------------|-------------|-------------|
| T4 (16GB) | ~20分 | ~40分 | ~1.5時間 |
| RTX 3090 (24GB) | ~10分 | ~20分 | ~50分 |

---

## 8. ファインチューニング手順

### 8.1 既存設定でのファインチューニング（推奨）

最も簡単で効果的な方法です。

#### データ準備

```bash
japanese_singing_data/
├── song1.wav
├── song2.wav
└── ...
```

**要件**:
- サンプリングレート: 44100Hz
- ファイル長: 1-30秒
- 形式: .wav, .mp3, .flac, .ogg, .m4a, .opus

#### 前処理

```bash
ffmpeg -i input.wav -ar 44100 -ac 1 -acodec pcm_s16le output.wav
```

#### ファインチューニング実行

```bash
python train.py \
  --config configs/presets/config_dit_mel_seed_uvit_whisper_base_f0_44k.yml \
  --dataset-dir ./japanese_singing_data \
  --run-name japanese_svc_ft \
  --batch-size 1 \
  --max-steps 2000 \
  --save-every 500 \
  --num-workers 0
```

| パラメータ | 推奨値 | 説明 |
|-----------|--------|------|
| `--batch-size` | 1 | T4/A10の場合 |
| `--max-steps` | 2000 | デモ品質 |
| `--num-workers` | 0 | Windows必須 |

### 8.2 ファインチューニング済みモデルの使用

```python
# main.pyでの変更
dit_checkpoint_path = "./runs/japanese_svc_ft/ft_model.pth"
dit_config_path = "./configs/presets/config_dit_mel_seed_uvit_whisper_base_f0_44k.yml"
```

---

## 9. 日本語HuBERT統合ガイド

### 9.1 前提条件

**重要**: この方法は**スクラッチからの再学習**が必要です。単純な設定変更では機能しません。

必要なリソース:
- 50時間以上の日本語歌声データ
- 5,000-10,000ステップの学習
- RTX 3090以上のGPU推奨

### 9.2 設定ファイルの作成

`configs/presets/config_dit_mel_seed_uvit_hubert_ja_f0_44k.yml`:

```yaml
log_dir: "./runs"
device: "cuda"
batch_size: 1
pretrained_model: ""  # スクラッチから学習

preprocess_params:
  sr: 44100
  spect_params:
    n_fft: 2048
    win_length: 2048
    hop_length: 512
    n_mels: 128
    fmin: 0
    fmax: "None"

model_params:
  dit_type: "DiT"

  vocoder:
    type: "bigvgan"
    name: "nvidia/bigvgan_v2_44khz_128band_512x"

  speech_tokenizer:
    type: 'xlsr'  # HuBERTはWav2Vec2互換
    name: "rinna/japanese-hubert-base"
    output_layer: 9

  style_encoder:
    dim: 192
    campplus_path: "campplus_cn_common.bin"

  length_regulator:
    channels: 768
    in_channels: 768  # HuBERT出力次元
    f0_condition: true

  DiT:
    hidden_dim: 768
    num_heads: 12
    depth: 17
    in_channels: 128
    content_dim: 768
    f0_condition: true
```

### 9.3 学習コマンド

```bash
python train.py \
  --config configs/presets/config_dit_mel_seed_uvit_hubert_ja_f0_44k.yml \
  --dataset-dir ./japanese_singing_data_large \
  --run-name japanese_svc_hubert_scratch \
  --batch-size 1 \
  --max-steps 10000 \
  --num-workers 0
```

---

## 10. SVCC 2025知見

### 10.1 チャレンジ概要

- **タスク**: 歌唱スタイル変換（SSC）- SVCより難しい
- **データ**: GTSinger（7スタイル）
- **参加**: 26システム、7チーム

### 10.2 トップシステム

| 順位 | システム | ベース | 特徴 |
|------|---------|--------|------|
| 1 | S6 | Vevo1.5 | DPO post-training |
| 上位 | S5 | Seed-VC | Residual Style Adaptor |

### 10.3 主な発見

- **話者類似度**: 5システムがGround Truthと有意差なし
- **スタイル変換**: 最高でも70%程度（GTは90%）
- **難しいスタイル**:
  - Breathy: 37.3%
  - Glissando: 42.6%
  - Vibrato: 43.9%

**示唆**: ビブラートやブレスなどの歌唱技法の変換は現在のSOTAでも難しい

---

## 11. 参考リンク集

### 論文
- [Seed-VC (arXiv 2024)](https://arxiv.org/abs/2411.09943)
- [Vevo: ICLR 2025](https://arxiv.org/abs/2502.07243)
- [SVCC 2025](https://arxiv.org/abs/2509.15629)
- [SingNet Dataset](https://arxiv.org/abs/2505.09325)
- [ContentVec](https://arxiv.org/abs/2204.09224)

### GitHub
- [Seed-VC](https://github.com/Plachtaa/seed-vc)
- [Amphion (Vevo1.5)](https://github.com/open-mmlab/Amphion)
- [RIFT-SVC](https://github.com/Pur1zumu/RIFT-SVC)
- [HQ-SVC](https://github.com/ShawnPi233/HQ-SVC)

### データセット
- [JVS-MuSiC](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_music)
- [GTSinger](https://github.com/AaronZ345/GTSinger)
- [kiritan_singing](https://github.com/mmorise/kiritan_singing)
- [無償音声コーパス一覧 (Qiita)](https://qiita.com/nakakq/items/74fea8b55d08032d25f9)

### コンテンツエンコーダー
- [rinna/japanese-hubert-base](https://huggingface.co/rinna/japanese-hubert-base)
- [ContentVec (HuggingFace)](https://huggingface.co/lengyue233/content-vec-best)

---

## 更新履歴

| 日付 | 内容 |
|------|------|
| 2025-12-29 | 初版作成 |
| 2025-12-29 | 最終版更新: SVCC 2025知見、コンテンツエンコーダ比較、大規模データセット収集方法、代替アーキテクチャ情報を追加 |

---

*本ドキュメントはSeed-VC日本語最適化のための調査結果をまとめたものです。*
