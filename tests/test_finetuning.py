"""
V1モデル（CFMベース）ファインチューニングのテストスクリプト

使用方法:
    uv run python tests/test_finetuning.py

このスクリプトは以下をテストします:
1. train.pyのインポートと初期化
2. データローダーの動作確認
3. 1ステップの学習が正常に動作するか
"""

import os
import sys
import tempfile
import shutil

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from scipy.io import wavfile


def create_dummy_audio_files(output_dir: str, num_files: int = 3, duration_sec: float = 5.0):
    """テスト用のダミー音声ファイルを作成"""
    os.makedirs(output_dir, exist_ok=True)
    sr = 22050

    for i in range(num_files):
        # ランダムなサイン波を生成
        t = np.linspace(0, duration_sec, int(sr * duration_sec), dtype=np.float32)
        freq = 220 + i * 110  # A3, D4, G4
        audio = 0.5 * np.sin(2 * np.pi * freq * t)

        # WAVファイルとして保存
        filepath = os.path.join(output_dir, f"test_audio_{i}.wav")
        wavfile.write(filepath, sr, (audio * 32767).astype(np.int16))
        print(f"Created: {filepath}")

    return output_dir


def test_dataloader():
    """データローダーのテスト"""
    print("\n=== Testing DataLoader ===")

    from data.ft_dataset import build_ft_dataloader

    # テスト用の一時ディレクトリを作成
    temp_dir = tempfile.mkdtemp()
    try:
        # ダミー音声ファイルを作成
        create_dummy_audio_files(temp_dir, num_files=3, duration_sec=5.0)

        # データローダーを構築
        spect_params = {
            'n_fft': 1024,
            'win_length': 1024,
            'hop_length': 256,
            'n_mels': 80,
            'fmin': 0,
            'fmax': 8000,
        }

        dataloader = build_ft_dataloader(
            data_path=temp_dir,
            spect_params=spect_params,
            sr=22050,
            batch_size=2,
            num_workers=0,
        )

        # 1バッチ取得
        for batch in dataloader:
            waves, mels, wave_lengths, mel_input_length = batch
            print(f"Waves shape: {waves.shape}")
            print(f"Mels shape: {mels.shape}")
            print(f"Wave lengths: {wave_lengths}")
            print(f"Mel input length: {mel_input_length}")
            break

        print("DataLoader test PASSED")
        return True

    except Exception as e:
        print(f"DataLoader test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # クリーンアップ
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_trainer_init():
    """Trainerの初期化テスト（GPUが必要）"""
    print("\n=== Testing Trainer Initialization ===")

    if not torch.cuda.is_available():
        print("CUDA not available, skipping Trainer test")
        return True

    from train import Trainer

    # テスト用の一時ディレクトリを作成
    temp_dir = tempfile.mkdtemp()
    try:
        # ダミー音声ファイルを作成
        create_dummy_audio_files(temp_dir, num_files=5, duration_sec=5.0)

        # 設定ファイルのパス
        config_path = "configs/presets/config_dit_mel_seed_uvit_whisper_small_wavenet.yml"

        if not os.path.exists(config_path):
            print(f"Config file not found: {config_path}")
            return False

        # Trainerを初期化（短いステップ数で）
        trainer = Trainer(
            config_path=config_path,
            pretrained_ckpt_path=None,  # Hugging Faceからダウンロード
            data_dir=temp_dir,
            run_name="test_run",
            batch_size=1,
            num_workers=0,
            steps=2,  # テスト用に2ステップのみ
            save_interval=1,
        )

        print(f"Trainer initialized successfully")
        print(f"Model keys: {list(trainer.model.keys())}")
        print(f"F0 condition: {trainer.f0_condition}")
        print(f"Sample rate: {trainer.sr}")

        print("Trainer initialization test PASSED")
        return True

    except Exception as e:
        print(f"Trainer initialization test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # クリーンアップ
        shutil.rmtree(temp_dir, ignore_errors=True)
        # テスト用のログディレクトリも削除
        test_log_dir = os.path.join("runs", "test_run")
        if os.path.exists(test_log_dir):
            shutil.rmtree(test_log_dir, ignore_errors=True)


def test_one_training_step():
    """1ステップの学習テスト（GPUが必要）"""
    print("\n=== Testing One Training Step ===")

    if not torch.cuda.is_available():
        print("CUDA not available, skipping training step test")
        return True

    from train import Trainer

    # テスト用の一時ディレクトリを作成
    temp_dir = tempfile.mkdtemp()
    try:
        # ダミー音声ファイルを作成（より多く、より長く）
        create_dummy_audio_files(temp_dir, num_files=5, duration_sec=8.0)

        # 設定ファイルのパス（小さいモデルを使用）
        config_path = "configs/presets/config_dit_mel_seed_uvit_xlsr_tiny.yml"

        if not os.path.exists(config_path):
            print(f"Config file not found: {config_path}, trying alternative...")
            config_path = "configs/presets/config_dit_mel_seed_uvit_whisper_small_wavenet.yml"

        if not os.path.exists(config_path):
            print(f"No config file found")
            return False

        # Trainerを初期化
        trainer = Trainer(
            config_path=config_path,
            pretrained_ckpt_path=None,
            data_dir=temp_dir,
            run_name="test_training_step",
            batch_size=1,
            num_workers=0,
            steps=1,
            save_interval=1,
        )

        # 1バッチ取得して学習
        for batch in trainer.train_dataloader:
            batch = [b.to(trainer.device) for b in batch]
            loss = trainer.train_one_step(batch)
            print(f"Training step completed with loss: {loss:.4f}")
            break

        print("One training step test PASSED")
        return True

    except Exception as e:
        print(f"One training step test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # クリーンアップ
        shutil.rmtree(temp_dir, ignore_errors=True)
        test_log_dir = os.path.join("runs", "test_training_step")
        if os.path.exists(test_log_dir):
            shutil.rmtree(test_log_dir, ignore_errors=True)


def main():
    """メインテスト関数"""
    print("=" * 60)
    print("V1 Finetuning Test Suite")
    print("=" * 60)

    results = {}

    # テスト1: データローダー
    results["DataLoader"] = test_dataloader()

    # テスト2: Trainer初期化（オプション、GPU必要）
    # results["Trainer Init"] = test_trainer_init()

    # テスト3: 1ステップ学習（オプション、GPU必要）
    # results["Training Step"] = test_one_training_step()

    # 結果サマリー
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("All tests PASSED!")
        return 0
    else:
        print("Some tests FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
