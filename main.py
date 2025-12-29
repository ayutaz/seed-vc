"""
Simplified singing voice conversion UI for testing
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import os
os.environ['HF_HUB_CACHE'] = './checkpoints/hf_cache'

import gradio as gr
import torch
import torchaudio
import librosa
from modules.commons import build_model, load_checkpoint, recursive_munch
import yaml
from hf_utils import load_custom_model_from_hf
import numpy as np
from pydub import AudioSegment

# Global variables
device = None
model = None
semantic_fn = None
vocoder_fn = None
campplus_model = None
to_mel = None
f0_fn = None
sr = None
hop_length = None

def load_models(fp16=True):
    global sr, hop_length, model, semantic_fn, vocoder_fn, campplus_model, to_mel, f0_fn

    print(f"Using device: {device}")
    print(f"Using fp16: {fp16}")

    # Load DiT model
    print("Loading DiT model...")
    dit_checkpoint_path, dit_config_path = load_custom_model_from_hf(
        "Plachta/Seed-VC",
        "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth",
        "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml"
    )

    config = yaml.safe_load(open(dit_config_path, "r"))
    model_params = recursive_munch(config["model_params"])
    model_params.dit_type = 'DiT'
    model = build_model(model_params, stage="DiT")
    hop_length = config["preprocess_params"]["spect_params"]["hop_length"]
    sr = config["preprocess_params"]["sr"]

    model, _, _, _ = load_checkpoint(
        model, None, dit_checkpoint_path,
        load_only_params=True, ignore_modules=[], is_distributed=False
    )
    for key in model:
        model[key].eval()
        model[key].to(device)
    model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)
    print("DiT model loaded")

    # Load CAMPPlus
    print("Loading CAMPPlus...")
    from modules.campplus.DTDNN import CAMPPlus
    campplus_ckpt_path = load_custom_model_from_hf("funasr/campplus", "campplus_cn_common.bin", config_filename=None)
    campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
    campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
    campplus_model.eval()
    campplus_model.to(device)
    print("CAMPPlus loaded")

    # Load BigVGAN
    print("Loading BigVGAN...")
    from modules.bigvgan import bigvgan
    bigvgan_name = model_params.vocoder.name
    bigvgan_model = bigvgan.BigVGAN.from_pretrained(bigvgan_name, use_cuda_kernel=False)
    bigvgan_model.remove_weight_norm()
    vocoder_fn = bigvgan_model.eval().to(device)
    print("BigVGAN loaded")

    # Load Whisper
    print("Loading Whisper...")
    from transformers import AutoFeatureExtractor, WhisperModel
    whisper_name = model_params.speech_tokenizer.name
    whisper_model = WhisperModel.from_pretrained(whisper_name, torch_dtype=torch.float16).to(device)
    del whisper_model.decoder
    whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_name)

    def _semantic_fn(waves_16k):
        ori_inputs = whisper_feature_extractor(
            [waves_16k.squeeze(0).cpu().numpy()],
            return_tensors="pt",
            return_attention_mask=True
        )
        ori_input_features = whisper_model._mask_input_features(
            ori_inputs.input_features, attention_mask=ori_inputs.attention_mask
        ).to(device)
        with torch.no_grad():
            ori_outputs = whisper_model.encoder(
                ori_input_features.to(whisper_model.encoder.dtype),
                head_mask=None, output_attentions=False, output_hidden_states=False, return_dict=True
            )
        S_ori = ori_outputs.last_hidden_state.to(torch.float32)
        S_ori = S_ori[:, :waves_16k.size(-1) // 320 + 1]
        return S_ori

    semantic_fn = _semantic_fn
    print("Whisper loaded")

    # Mel spectrogram
    mel_fn_args = {
        "n_fft": config['preprocess_params']['spect_params']['n_fft'],
        "win_size": config['preprocess_params']['spect_params']['win_length'],
        "hop_size": config['preprocess_params']['spect_params']['hop_length'],
        "num_mels": config['preprocess_params']['spect_params']['n_mels'],
        "sampling_rate": sr,
        "fmin": config['preprocess_params']['spect_params'].get('fmin', 0),
        "fmax": None if config['preprocess_params']['spect_params'].get('fmax', "None") == "None" else 8000,
        "center": False
    }
    from modules.audio import mel_spectrogram
    to_mel = lambda x: mel_spectrogram(x, **mel_fn_args)

    # Load RMVPE
    print("Loading RMVPE...")
    from modules.rmvpe import RMVPE
    model_path = load_custom_model_from_hf("lj1995/VoiceConversionWebUI", "rmvpe.pt", None)
    rmvpe = RMVPE(model_path, is_half=False, device=device)
    f0_fn = rmvpe.infer_from_audio
    print("RMVPE loaded")

    print("All models loaded!")

@torch.no_grad()
@torch.inference_mode()
def voice_conversion(source, target, diffusion_steps, length_adjust, inference_cfg_rate, auto_f0_adjust, pitch_shift):
    """Simplified voice conversion for testing"""
    if source is None or target is None:
        return None

    # Load audio
    source_audio = librosa.load(source, sr=sr)[0]
    ref_audio = librosa.load(target, sr=sr)[0]

    source_audio = torch.tensor(source_audio).unsqueeze(0).float().to(device)
    ref_audio = torch.tensor(ref_audio[:sr * 25]).unsqueeze(0).float().to(device)

    # Resample to 16k
    ref_waves_16k = torchaudio.functional.resample(ref_audio, sr, 16000)
    converted_waves_16k = torchaudio.functional.resample(source_audio, sr, 16000)

    # Extract content features
    S_alt = semantic_fn(converted_waves_16k)
    S_ori = semantic_fn(ref_waves_16k)

    # Extract style
    feat2 = torchaudio.compliance.kaldi.fbank(ref_waves_16k, num_mel_bins=80, dither=0, sample_frequency=16000)
    feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
    style2 = campplus_model(feat2.unsqueeze(0))

    # Extract F0
    F0_ori = f0_fn(ref_waves_16k[0], thred=0.03)
    F0_alt = f0_fn(converted_waves_16k[0], thred=0.03)

    # Generate mel spectrograms
    mel = to_mel(source_audio.to(device).float())  # source mel
    mel2 = to_mel(ref_audio.to(device).float())    # reference mel

    # Length targets
    target_lengths = torch.LongTensor([int(mel.size(2) * length_adjust)]).to(mel.device)
    target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)

    # Convert F0 to tensor (F0_alt and F0_ori are numpy arrays from RMVPE)
    F0_ori = torch.from_numpy(F0_ori).to(device)[None]
    F0_alt = torch.from_numpy(F0_alt).to(device)[None]

    # Calculate F0 adjustment
    voiced_F0_ori = F0_ori[F0_ori > 1]
    voiced_F0_alt = F0_alt[F0_alt > 1]

    log_f0_alt = torch.log(F0_alt + 1e-5)
    voiced_log_f0_ori = torch.log(voiced_F0_ori + 1e-5)
    voiced_log_f0_alt = torch.log(voiced_F0_alt + 1e-5)
    median_log_f0_ori = torch.median(voiced_log_f0_ori)
    median_log_f0_alt = torch.median(voiced_log_f0_alt)

    # Shift alt log f0 level to ori log f0 level
    shifted_log_f0_alt = log_f0_alt.clone()
    if auto_f0_adjust:
        shifted_log_f0_alt[F0_alt > 1] = log_f0_alt[F0_alt > 1] - median_log_f0_alt + median_log_f0_ori
    shifted_f0_alt = torch.exp(shifted_log_f0_alt)
    if pitch_shift != 0:
        factor = 2 ** (pitch_shift / 12)
        shifted_f0_alt[F0_alt > 1] = shifted_f0_alt[F0_alt > 1] * factor

    # Length regulation - get cond from source and prompt_condition from reference
    cond, _, codes, _, _ = model.length_regulator(
        S_alt, ylens=target_lengths, n_quantizers=3, f0=shifted_f0_alt
    )
    prompt_condition, _, _, _, _ = model.length_regulator(
        S_ori, ylens=target2_lengths, n_quantizers=3, f0=F0_ori
    )

    # Concatenate prompt condition with source condition along time dimension
    cat_condition = torch.cat([prompt_condition, cond], dim=1)

    # Run CFM inference
    vc_target = model.cfm.inference(
        cat_condition, torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
        mel2, style2, None, diffusion_steps, inference_cfg_rate=inference_cfg_rate
    )
    vc_target = vc_target[:, :, mel2.size(-1):]

    # Vocoder
    vc_wave = vocoder_fn(vc_target.float())[0]

    output = vc_wave[0].cpu().numpy()
    output = (output * 32768.0).clip(-32768, 32767).astype(np.int16)

    return (sr, output)


def main():
    global device

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")

    # Load models
    load_models(fp16=True)

    # Create UI
    print("Creating Gradio interface...")
    with gr.Blocks(title="Seed Voice Conversion") as demo:
        gr.Markdown("# Seed Voice Conversion - Singing Voice")

        with gr.Row():
            source = gr.Audio(type="filepath", label="Source Audio (English song)")
            target = gr.Audio(type="filepath", label="Reference Audio (Japanese voice)")

        with gr.Row():
            diffusion_steps = gr.Slider(1, 100, value=50, step=1, label="Diffusion Steps")
            length_adjust = gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="Length Adjust")

        with gr.Row():
            cfg_rate = gr.Slider(0.0, 1.0, value=0.7, step=0.1, label="CFG Rate")
            auto_f0 = gr.Checkbox(value=True, label="Auto F0 Adjust")
            pitch_shift = gr.Slider(-24, 24, value=0, step=1, label="Pitch Shift (semitones)")

        convert_btn = gr.Button("Convert", variant="primary")
        output = gr.Audio(label="Output")

        convert_btn.click(
            voice_conversion,
            inputs=[source, target, diffusion_steps, length_adjust, cfg_rate, auto_f0, pitch_shift],
            outputs=output
        )

    print("Launching Gradio...")
    demo.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    main()
