#!/usr/bin/env python3
"""
Inference test script for ERNP + LIG (Emotion-Rectified Noise Prior + Likelihood-Inverse Guidance)
from the paper "Rectifying the Emotional Flow: Aligning Priors and Dynamic Guidance for High-Arousal TTS"

All hyperparameters are loaded from a YAML config file (default: src/f5_tts/configs/emo_infer.yaml).
CLI arguments override config values when provided.

This script compares:
  1. Baseline F5-TTS (standard CFG)
  2. F5-TTS + ERNP only
  3. F5-TTS + LIG only
  4. F5-TTS + ERNP + LIG (full method)

Usage:
    python infer_emo_test.py
    python infer_emo_test.py --config path/to/emo_infer.yaml
    python infer_emo_test.py --ernp_lambda_init 50.0 --lig_pi 0.99
    python infer_emo_test.py --ref_audio /path/to/emotional_ref.wav --ref_text "transcript"
"""

import argparse
import json
import os
import time
from importlib.resources import files

import numpy as np
import soundfile as sf
import torch

from f5_tts.api import F5TTS
from f5_tts.configs.emo_config import get_test_sentences, load_emo_config
from f5_tts.infer.utils_infer import (
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
)
from f5_tts.model.utils import convert_char_to_pinyin, seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="ERNP + LIG Inference Test for F5-TTS")

    # Config file
    parser.add_argument("--config", type=str, default=None,
                        help="Path to emo_infer.yaml config file (default: built-in)")

    # Model (override config)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--ckpt_file", type=str, default=None)
    parser.add_argument("--vocab_file", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)

    # Reference audio (override config)
    parser.add_argument("--ref_audio", type=str, default=None)
    parser.add_argument("--ref_text", type=str, default=None)
    parser.add_argument("--lang", type=str, default=None, choices=["en", "zh"])

    # Generation text (override config)
    parser.add_argument("--gen_text", type=str, default=None)

    # Inference params (override config)
    parser.add_argument("--nfe_step", type=int, default=None)
    parser.add_argument("--cfg_strength", type=float, default=None)
    parser.add_argument("--speed", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)

    # ERNP params (override config)
    parser.add_argument("--ernp_lambda_init", type=float, default=None,
                        help="ERNP lookahead guidance strength")
    parser.add_argument("--ernp_lambda_base", type=float, default=None,
                        help="ERNP calibration guidance strength")
    parser.add_argument("--ernp_tau", type=float, default=None,
                        help="ERNP lookahead step size (None=auto)")

    # LIG params (override config)
    parser.add_argument("--lig_pi", type=float, default=None,
                        help="LIG purity coefficient")
    parser.add_argument("--lig_lambda_max", type=float, default=None,
                        help="LIG max guidance clamp")
    parser.add_argument("--lig_sigma", type=float, default=None,
                        help="LIG noise scale for log-R estimation")

    # Output (override config)
    parser.add_argument("--output_dir", type=str, default=None)

    return parser.parse_args()


# Determine available configurations based on config completeness
def get_available_configs(cfg):
    """Return only the configs that have valid parameters."""
    configs = [
        {
            "name": "baseline",
            "desc": "Standard CFG (baseline)",
            "use_ernp": False,
            "use_lig": False,
        },
    ]

    ernp_available = (
        cfg.ernp.lambda_init is not None
        and cfg.ernp.lambda_base is not None
    )
    lig_available = (
        cfg.lig.pi is not None
        and cfg.lig.lambda_max is not None
        and cfg.lig.sigma is not None
    )

    if ernp_available:
        configs.append({
            "name": "ernp_only",
            "desc": "ERNP only (emotion-rectified noise prior)",
            "use_ernp": True,
            "use_lig": False,
        })

    if lig_available:
        configs.append({
            "name": "lig_only",
            "desc": "LIG only (likelihood-inverse guidance)",
            "use_ernp": False,
            "use_lig": True,
        })

    if ernp_available and lig_available:
        configs.append({
            "name": "ernp_lig",
            "desc": "ERNP + LIG (full method)",
            "use_ernp": True,
            "use_lig": True,
        })

    return configs, ernp_available, lig_available


def run_single_inference(
    tts_model,
    vocoder,
    audio_tensor,
    ref_text,
    gen_text,
    duration,
    cfg,
    use_ernp=False,
    use_lig=False,
):
    """Run a single inference with specified ERNP/LIG settings."""
    seed_everything(cfg.inference.seed)

    hop_length = cfg.inference.hop_length
    ref_audio_len = audio_tensor.shape[-1] // hop_length

    text_list = [ref_text + gen_text]
    final_text_list = convert_char_to_pinyin(text_list)

    with torch.inference_mode():
        generated, trajectory = tts_model.sample(
            cond=audio_tensor,
            text=final_text_list,
            duration=duration,
            steps=cfg.inference.nfe_step,
            cfg_strength=cfg.inference.cfg_strength,
            sway_sampling_coef=cfg.inference.sway_sampling_coef,
            seed=cfg.inference.seed,
            # ERNP params from config
            use_ernp=use_ernp,
            ernp_lambda_init=cfg.ernp.lambda_init,
            ernp_lambda_base=cfg.ernp.lambda_base,
            ernp_tau=cfg.ernp.tau,
            # LIG params from config
            use_lig=use_lig,
            lig_pi=cfg.lig.pi,
            lig_lambda_max=cfg.lig.lambda_max,
            lig_sigma=cfg.lig.sigma,
        )

        generated = generated.to(torch.float32)
        generated = generated[:, ref_audio_len:, :]
        generated_mel = generated.permute(0, 2, 1)

        generated_wave = vocoder.decode(generated_mel)
        generated_wave = generated_wave.squeeze().cpu().numpy()

    return generated_wave


def main():
    args = parse_args()

    # Load config from YAML, with CLI overrides
    cli_overrides = {k: v for k, v in vars(args).items() if v is not None and k != "config"}
    cfg = load_emo_config(config_path=args.config, overrides=cli_overrides)

    # Resolve output dir
    output_dir = cfg.output.dir
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("ERNP + LIG Inference Test for F5-TTS")
    print("(Paper: Rectifying the Emotional Flow)")
    print("=" * 70)
    print(f"Config: {args.config or 'default (emo_infer.yaml)'}")

    # Device
    device = cfg.model.device
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load model
    print("\n[1/4] Loading F5-TTS model...")
    t0 = time.time()
    tts = F5TTS(
        model=cfg.model.name,
        ckpt_file=cfg.model.ckpt_file,
        vocab_file=cfg.model.vocab_file,
        device=device,
    )
    model_load_time = time.time() - t0
    print(f"  Model loaded in {model_load_time:.2f}s")

    # Prepare reference audio
    print("\n[2/4] Preparing reference audio...")
    if cfg.ref_audio.path:
        ref_audio_path = cfg.ref_audio.path
        ref_text = cfg.ref_audio.text
    else:
        if cfg.ref_audio.lang == "zh":
            ref_audio_path = str(files("f5_tts").joinpath("infer/examples/basic/basic_ref_zh.wav"))
            ref_text = cfg.ref_audio.text or ""
        else:
            ref_audio_path = str(files("f5_tts").joinpath("infer/examples/basic/basic_ref_en.wav"))
            ref_text = cfg.ref_audio.text or "Some call me nature, others call me mother nature."

    ref_audio_path, ref_text = preprocess_ref_audio_text(ref_audio_path, ref_text)
    print(f"  Ref audio: {ref_audio_path}")
    print(f"  Ref text : {ref_text}")

    # Load and preprocess reference audio tensor
    import torchaudio
    audio, sr = torchaudio.load(ref_audio_path)
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    target_sample_rate = cfg.inference.target_sample_rate
    hop_length = cfg.inference.hop_length
    target_rms = cfg.inference.target_rms
    rms = torch.sqrt(torch.mean(torch.square(audio)))
    if rms < target_rms:
        audio = audio * target_rms / rms
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        audio = resampler(audio)
    audio = audio.to(device)

    ref_audio_len = audio.shape[-1] // hop_length

    # Determine available configs
    CONFIGS, ernp_available, lig_available = get_available_configs(cfg)

    # Get test sentences from config
    test_sentences = get_test_sentences(cfg, cfg.ref_audio.lang)
    if args.gen_text:
        test_sentences = [{"text": args.gen_text, "label": "custom", "intensity": "unknown"}]

    print(f"\n[3/4] Running inference: {len(test_sentences)} sentences x {len(CONFIGS)} configs = {len(test_sentences) * len(CONFIGS)} total")
    print(f"  NFE Steps      : {cfg.inference.nfe_step}")
    print(f"  CFG Strength   : {cfg.inference.cfg_strength}")
    if ernp_available:
        print(f"  ERNP           : enabled")
    else:
        print(f"  ERNP           : disabled (params not in config)")
    if lig_available:
        print(f"  LIG            : enabled")
    else:
        print(f"  LIG            : disabled (params not in config)")
    print(f"  Seed           : {cfg.inference.seed}")
    print("-" * 70)

    all_results = []

    for si, sentence in enumerate(test_sentences):
        gen_text = sentence["text"]
        label = sentence["label"]

        # Calculate duration
        ref_text_len = len(ref_text.encode("utf-8"))
        gen_text_len = len(gen_text.encode("utf-8"))
        duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / cfg.inference.speed)

        print(f"\n{'='*50}")
        print(f"Sentence {si+1}/{len(test_sentences)} [{label}]")
        print(f"  Text: {gen_text[:80]}{'...' if len(gen_text) > 80 else ''}")
        print(f"  Duration: {duration} frames")

        for config in CONFIGS:
            config_name = config["name"]
            print(f"\n  --- Config: {config['desc']} ---")

            t1 = time.time()
            wav = run_single_inference(
                tts_model=tts.ema_model,
                vocoder=tts.vocoder,
                audio_tensor=audio,
                ref_text=ref_text,
                gen_text=gen_text,
                duration=duration,
                cfg=cfg,
                use_ernp=config["use_ernp"],
                use_lig=config["use_lig"],
            )
            infer_time = time.time() - t1

            # Save audio
            out_filename = f"{label}_{config_name}.wav"
            out_path = os.path.join(output_dir, out_filename)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            sf.write(out_path, wav, target_sample_rate)

            audio_duration = len(wav) / target_sample_rate
            rtf = infer_time / audio_duration if audio_duration > 0 else float("inf")

            # Compute basic audio statistics
            wav_abs = np.abs(wav)
            energy = np.sqrt(np.mean(wav ** 2))
            peak = np.max(wav_abs)
            dynamic_range = 20 * np.log10(peak / (energy + 1e-10)) if energy > 0 else 0

            result = {
                "sentence_idx": si + 1,
                "label": label,
                "config": config_name,
                "config_desc": config["desc"],
                "text": gen_text,
                "output_file": out_path,
                "audio_duration_s": round(audio_duration, 3),
                "infer_time_s": round(infer_time, 3),
                "rtf": round(rtf, 4),
                "rms_energy": round(float(energy), 6),
                "peak_amplitude": round(float(peak), 6),
                "dynamic_range_db": round(float(dynamic_range), 2),
            }
            all_results.append(result)

            print(f"    Output   : {out_path}")
            print(f"    Duration : {audio_duration:.2f}s | Infer: {infer_time:.2f}s | RTF: {rtf:.4f}")
            print(f"    Energy   : {energy:.4f} | Peak: {peak:.4f} | DR: {dynamic_range:.1f}dB")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("[4/4] SUMMARY")
    print("=" * 70)

    # Group by config
    for config in CONFIGS:
        config_results = [r for r in all_results if r["config"] == config["name"]]
        avg_rtf = np.mean([r["rtf"] for r in config_results])
        avg_energy = np.mean([r["rms_energy"] for r in config_results])
        avg_dr = np.mean([r["dynamic_range_db"] for r in config_results])
        print(f"\n  [{config['desc']}]")
        print(f"    Avg RTF: {avg_rtf:.4f} | Avg Energy: {avg_energy:.4f} | Avg DR: {avg_dr:.1f}dB")

    # Save detailed results
    results_jsonl = os.path.join(output_dir, "emo_infer_results.jsonl")
    with open(results_jsonl, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nDetailed results: {results_jsonl}")

    # Save comparison table as JSON
    comparison = {}
    for config in CONFIGS:
        config_results = [r for r in all_results if r["config"] == config["name"]]
        comparison[config["name"]] = {
            "description": config["desc"],
            "avg_rtf": round(float(np.mean([r["rtf"] for r in config_results])), 4),
            "avg_rms_energy": round(float(np.mean([r["rms_energy"] for r in config_results])), 6),
            "avg_dynamic_range_db": round(float(np.mean([r["dynamic_range_db"] for r in config_results])), 2),
            "avg_infer_time_s": round(float(np.mean([r["infer_time_s"] for r in config_results])), 3),
            "num_samples": len(config_results),
        }
    comparison_file = os.path.join(output_dir, "comparison_summary.json")
    with open(comparison_file, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    print(f"Comparison summary: {comparison_file}")

    # Save the config metadata for this run (without sensitive ERNP/LIG values)
    config_dump_file = os.path.join(output_dir, "config_used.yaml")
    import yaml
    safe_dump = {
        "model": {"name": cfg.model.name},
        "ref_audio": {"lang": cfg.ref_audio.lang},
        "inference": {
            "nfe_step": cfg.inference.nfe_step,
            "cfg_strength": cfg.inference.cfg_strength,
            "speed": cfg.inference.speed,
            "seed": cfg.inference.seed,
        },
        "ernp_enabled": ernp_available,
        "lig_enabled": lig_available,
    }
    with open(config_dump_file, "w", encoding="utf-8") as f:
        yaml.dump(safe_dump, f, default_flow_style=False, allow_unicode=True)
    print(f"Config used: {config_dump_file}")

    print(f"\nAll audio files saved to: {os.path.abspath(output_dir)}")
    print("Done!")


if __name__ == "__main__":
    main()
