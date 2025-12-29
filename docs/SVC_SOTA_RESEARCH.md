# 歌声変換（SVC）最新研究調査レポート 2024-2025

## 概要

本ドキュメントは、歌声変換（Singing Voice Conversion, SVC）分野の最新研究（2024-2025年）を調査し、アーキテクチャ、学習手法、データセットについてまとめたものです。

### SVCの主な課題

1. **Timbre Leakage（音色漏れ）**: ソース話者の音色が変換後も残る問題
2. **Zero-shot Generalization**: 未見の話者への汎化性能
3. **歌唱スタイルの保持**: ビブラート、ブレス、グリッサンドなどの表現

---

## 1. 主要アーキテクチャ比較

### 1.1 アーキテクチャ一覧

| モデル | 発表 | 生成方式 | コンテンツ抽出 | Zero-shot | 公開状況 |
|--------|------|---------|---------------|-----------|---------|
| **Seed-VC** | 2024 | Flow Matching + DiT | Whisper/XLSR | ✅ | [OSS (GPL v3)](https://github.com/Plachtaa/seed-vc) |
| **Vevo1.5** | ICLR 2025 | ARLM + Flow Matching | BPE + Chromagram | ✅ | [Amphion](https://github.com/open-mmlab/Amphion) |
| **RIFT-SVC** | 2025 | Rectified Flow + DiT | ContentVec | ❌ | [OSS](https://github.com/Pur1zumu/RIFT-SVC) |
| **DAFMSVC** | 2025 | CFM + Dual Attention | WavLM + kNN | ✅ | 論文のみ |
| **SaMoye** | IJCAI 2025 | VITS | 複数ASR融合 | ✅ | [OSS](https://github.com/CarlWangChina/SaMoye-SVC) |
| **HQ-SVC** | AAAI 2026 | Unified Codec | Multi-feature | ✅ | [推論のみ](https://github.com/ShawnPi233/HQ-SVC) |
| **kNN-SVC** | ICASSP 2025 | kNN + Additive Synth | WavLM | ✅ | [OSS](https://github.com/SmoothKen/knn-svc) |
| **FreeSVC** | ICASSP 2025 | VITS + SPIN | ECAPA2 | ✅ | [OSS](https://github.com/freds0/free-svc) |

### 1.2 詳細アーキテクチャ

#### Seed-VC
```
入力音声 → Whisper/XLSR(コンテンツ) → CAMPPlus(話者)
         → Length Regulator → DiT(Flow Matching)
         → BigVGAN → 出力音声
```
- **特徴**: 外部Timbre Shifter（OpenVoice）で学習時の音色摂動
- **推論遅延**: ~300ms（リアルタイム対応）

#### Vevo1.5 (SVCC 2025 Top System)
```
入力 → BPE Tokenizer + Chromagram Melody Tokenizer
     → Autoregressive LM → Content-Style Tokens
     → Flow Matching Transformer → Mel Spectrogram
     → Vocoder → 出力
```
- **特徴**: 2段階生成（ARLM + Flow Matching）
- **学習データ**: SingNet 7,000時間

#### RIFT-SVC V3
```
入力音声 → ContentVec(コンテンツ) + F0 + RMS
         → Rectified Flow DiT
         → NSF-HiFiGAN → 出力音声
```
- **特徴**: Multiple CFG（articulation/timbre個別制御）
- **改善点**: V2のWhisper削除、LogNormスケジューラー

#### DAFMSVC
```
入力音声 → WavLM-Large → kNN Feature Matching (k=4)
         → Dual Cross-Attention (Content + Melody + Speaker)
         → Conditional Flow Matching
         → 出力音声
```
- **特徴**: Learnable gating parameter αで漸進的音色注入
- **結果**: OpenSingerでSSIM 0.754達成

---

## 2. コンテンツエンコーダ比較

### 2.1 SVCC 2023 比較結果

| エンコーダ | 明瞭度 | 話者分離 | Voice ID精度 | 推奨用途 |
|-----------|--------|---------|--------------|---------|
| **ContentVec** | ◎ | ◎ | 37.7% | SVC全般（推奨） |
| HuBERT | ◎ | △ | 73.7% | 汎用（話者情報残る） |
| HuBERT-soft | ○ | △ | - | So-VITS-SVC |
| Whisper | ○ | ○ | - | 多言語対応 |
| WavLM | ○ | ○ | - | 大規模モデル |

**結論**: ContentVecが最も話者情報を除去でき、SVCに適している

### 2.2 各エンコーダの特徴

| エンコーダ | 出力次元 | 学習データ | 備考 |
|-----------|---------|-----------|------|
| ContentVec | 256 | LibriSpeech | HuBERT + speaker-invariant tweaks |
| HuBERT-base | 768 | LibriSpeech | Meta製、広く使用 |
| Whisper-small | 768 | 680k時間 | OpenAI、多言語対応 |
| WavLM-Large | 1024 | 94k時間 | Microsoft、大規模 |
| rinna/japanese-hubert | 768 | 19k時間日本語 | 日本語特化 |

### 2.3 推奨選択

- **Zero-shot SVC**: ContentVec（話者分離性能最高）
- **多言語対応**: Whisper（Seed-VCの選択）
- **日本語特化**: rinna/japanese-hubert（要再学習）

---

## 3. 生成モデルの進化

### 3.1 世代別比較

| 世代 | 代表モデル | 特徴 | 問題点 |
|------|-----------|------|--------|
| **GAN** | HiFi-GAN, BigVGAN | 高速推論 | mode collapse |
| **VAE-GAN** | VITS, So-VITS-SVC | End-to-end | 音質限界 |
| **Diffusion** | DiffSVC, Grad-SVC | 高音質 | 推論速度 |
| **Flow Matching** | Seed-VC, RIFT-SVC | 高音質+高速 | 複雑な学習 |
| **ARLM + Diffusion** | Vevo1.5 | SOTA | 大規模データ必要 |

### 3.2 Flow Matching vs Diffusion

| 項目 | Diffusion | Flow Matching |
|------|-----------|---------------|
| サンプリング | 多ステップ必要 | 少ステップで可能 |
| 学習安定性 | 安定 | より安定 |
| 推論速度 | 遅い | 速い |
| 採用例 | DiffSVC | Seed-VC, Vevo1.5 |

---

## 4. データセット

### 4.1 大規模データセット

| データセット | 規模 | 言語 | 話者数 | ライセンス | 入手方法 |
|-------------|------|------|--------|-----------|---------|
| **SingNet** | 3,000時間 | 多言語 | 多数 | 未公開 | 非公開 |
| **GTSinger** | 80時間 | 多言語 | 多数 | CC BY-NC-SA | [GitHub](https://github.com/AaronZ345/GTSinger) |
| SaMoye Dataset | 1,815時間 | 中国語 | 6,367 | 非公開 | 論文参照 |
| OpenSinger | 50時間 | 中国語 | 76 | 研究用 | 公開 |

### 4.2 日本語歌声データセット

| データセット | 規模 | 特徴 | 商用利用 |
|-------------|------|------|---------|
| JVS-MuSiC | ~2時間 | 100人歌手 | ❌ |
| Kiritan | ~57分 | プロ歌手 | ❌ |
| JSUT-Song | 数時間 | 単一話者 | ✅ (CC BY-SA) |

### 4.3 学習データ要件

| 目的 | 最小データ量 | 推奨データ量 | 学習ステップ |
|------|-------------|-------------|-------------|
| PoC/デモ | 30分 | 2時間 | 1,000-2,000 |
| 商用品質 | 4時間 | 10時間+ | 5,000-10,000 |
| SOTA | 50時間+ | 500時間+ | 10,000+ |

---

## 5. SVCC 2025 知見

### 5.1 チャレンジ概要

- **タスク**: 歌唱スタイル変換（SSC）- SVCより難しい
- **データ**: GTSinger（7スタイル: breathy, falsetto, mixed, pharyngeal, glissando, vibrato, control）
- **参加**: 26システム、7チーム

### 5.2 トップシステム

| 順位 | システム | ベース | 特徴 |
|------|---------|--------|------|
| 1 | S6 | Vevo1.5 | DPO post-training |
| 2 | S7 | Vevo1.5 | Qwen 2.5-0.5B ARLM |
| 上位 | S5 | Seed-VC | Residual Style Adaptor |

### 5.3 主な発見

- **話者類似度**: 5システムがGround Truthと有意差なし
- **スタイル変換**: 最高でも70%程度（GTは90%）
- **難しいスタイル**:
  - Breathy: 37.3%
  - Glissando: 42.6%
  - Vibrato: 43.9%

---

## 6. Seed-VCとの比較・改善オプション

### 6.1 Seed-VCの優位点

- ✅ Zero-shot対応（1-30秒の参照音声のみ）
- ✅ リアルタイム推論（~300ms遅延）
- ✅ MITライセンス（商用利用可）
- ✅ 44kHz出力（歌声対応）
- ✅ 活発なメンテナンス

### 6.2 改善の可能性がある領域

| 領域 | 現状 | 改善案 | 難易度 |
|------|------|--------|--------|
| コンテンツエンコーダ | Whisper | ContentVec | 高（再学習必要） |
| 話者エンコーダ | CAMPPlus | ECAPA2 | 中 |
| 生成モデル | DiT | ARLM + DiT | 高 |
| データ | 汎用 | 日本語FT | 低 |

### 6.3 推奨アプローチ

#### 短期（1-2週間）
```bash
# 既存モデルで日本語ファインチューニング
python train.py \
  --config configs/presets/config_dit_mel_seed_uvit_whisper_base_f0_44k.yml \
  --dataset-dir <japanese_singing_data> \
  --batch-size 1 --max-steps 2000
```

#### 中期（1-2ヶ月）
- ContentVec統合を検討
- `train.py`の`build_semantic_fn`を修正
- スクラッチから再学習（5,000+ステップ）

#### 長期
- Vevo1.5/Amphionへの移行検討
- 大規模日本語歌声データセット構築

---

## 7. 参考リンク

### 論文
- [Seed-VC Paper](https://arxiv.org/abs/2411.09943)
- [Vevo: ICLR 2025](https://arxiv.org/abs/2502.07243)
- [SVCC 2025](https://arxiv.org/abs/2509.15629)
- [DAFMSVC](https://arxiv.org/abs/2508.05978)
- [SaMoye](https://arxiv.org/abs/2407.07728)

### GitHub
- [Seed-VC](https://github.com/Plachtaa/seed-vc)
- [Amphion (Vevo1.5)](https://github.com/open-mmlab/Amphion)
- [RIFT-SVC](https://github.com/Pur1zumu/RIFT-SVC)
- [HQ-SVC](https://github.com/ShawnPi233/HQ-SVC)
- [SaMoye-SVC](https://github.com/CarlWangChina/SaMoye-SVC)

### データセット
- [GTSinger](https://github.com/AaronZ345/GTSinger)
- [JVS-MuSiC](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_music)

---

## 8. 結論

**Zero-shot SVCにおいて、Seed-VCは依然として最良の選択肢です。**

理由:
1. 完全なZero-shot対応
2. リアルタイム推論可能
3. GPL v3ライセンス（派生物も同ライセンス）
4. 44kHz歌声変換対応
5. ファインチューニング可能

**より高品質を目指す場合:**
1. 日本語データでのファインチューニング（最も費用対効果が高い）
2. ContentVecへの置換（再学習必要）
3. Vevo1.5への移行（大規模データがあれば）

---

*最終更新: 2025年12月*
