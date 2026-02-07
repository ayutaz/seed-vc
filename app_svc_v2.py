import os
os.environ['HF_HUB_CACHE'] = './checkpoints/hf_cache'
import gradio as gr
import torch
import yaml
import argparse

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

dtype = torch.float16 if device.type == "cuda" else torch.float32

vc_wrapper = None
sr = 44100


def load_models(args):
    global vc_wrapper, sr
    from hydra.utils import instantiate
    from omegaconf import DictConfig

    config_path = args.config or "configs/v2/vc_wrapper_svc.yaml"
    cfg = DictConfig(yaml.safe_load(open(config_path, "r")))
    vc_wrapper = instantiate(cfg)
    sr = cfg.sr

    vc_wrapper.load_checkpoints(
        cfm_checkpoint_path=args.cfm_checkpoint_path,
        ar_checkpoint_path=args.ar_checkpoint_path,
    )
    vc_wrapper.to(device)
    vc_wrapper.eval()

    if args.compile:
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.triton.unique_kernel_names = True
        if hasattr(torch._inductor.config, "fx_graph_cache"):
            torch._inductor.config.fx_graph_cache = True
        vc_wrapper.compile_cfm()


@torch.no_grad()
@torch.inference_mode()
def convert_singing_voice(
    source, target,
    diffusion_steps, length_adjust, inference_cfg_rate,
    auto_f0_adjust, pitch_shift,
):
    if source is None or target is None:
        return None
    result = vc_wrapper.convert_singing_voice(
        source_audio_path=source,
        target_audio_path=target,
        diffusion_steps=int(diffusion_steps),
        length_adjust=length_adjust,
        inference_cfg_rate=inference_cfg_rate,
        auto_f0_adjust=auto_f0_adjust,
        pitch_shift=int(pitch_shift),
        device=device,
        dtype=dtype,
    )
    return (sr, result.squeeze(0) if result.ndim > 1 else result)


def main(args):
    load_models(args)

    description = (
        "V2モデルによる歌声変換（Singing Voice Conversion）<br>"
        "ASTRAL量子化による話者分離 + F0条件付けによるピッチ制御<br>"
        "参照音声が25秒を超える場合、自動的にカットされます。"
    )

    inputs = [
        gr.Audio(type="filepath", label="ソース音声（歌声）"),
        gr.Audio(type="filepath", label="参照音声（ターゲット話者）"),
        gr.Slider(minimum=1, maximum=200, value=30, step=1,
                  label="拡散ステップ数",
                  info="デフォルト30、高品質には50〜100を推奨"),
        gr.Slider(minimum=0.5, maximum=2.0, step=0.1, value=1.0,
                  label="長さ調整",
                  info="1.0未満で速度アップ、1.0超で速度ダウン"),
        gr.Slider(minimum=0.0, maximum=1.0, step=0.1, value=0.5,
                  label="推論CFGレート",
                  info="変換品質の調整（0.0〜1.0）"),
        gr.Checkbox(label="自動F0調整", value=True,
                    info="ピッチレンジを参照音声に自動マッチング"),
        gr.Slider(minimum=-24, maximum=24, step=1, value=0,
                  label="ピッチシフト（半音）",
                  info="-12で1オクターブ下、+12で1オクターブ上"),
    ]

    outputs = gr.Audio(label="変換結果", format='wav')

    gr.Interface(
        fn=convert_singing_voice,
        inputs=inputs,
        outputs=outputs,
        title="Seed-VC V2 歌声変換",
        description=description,
        cache_examples=False,
    ).launch(share=args.share)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config file (default: configs/v2/vc_wrapper_svc.yaml)")
    parser.add_argument("--cfm-checkpoint-path", type=str, default=None,
                        help="Path to custom CFM checkpoint")
    parser.add_argument("--ar-checkpoint-path", type=str, default=None,
                        help="Path to custom AR checkpoint")
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio link")
    parser.add_argument("--compile", action="store_true",
                        help="Compile CFM model with torch.compile")
    args = parser.parse_args()
    main(args)
