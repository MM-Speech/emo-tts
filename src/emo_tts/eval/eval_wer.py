"""
WER Evaluation for HIED benchmark.

Uses Whisper (Radford et al., 2023) for English and FunASR Paraformer (Gao et al., 2023) for Chinese.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def load_asr_model(lang, device="cuda"):
    if lang == "zh":
        from funasr import AutoModel

        model = AutoModel(
            model="iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            disable_update=True,
        )
    elif lang == "en":
        from faster_whisper import WhisperModel

        model = WhisperModel("large-v3", device=device, compute_type="float16")
    else:
        raise ValueError(f"Unsupported language: {lang}. Use 'zh' or 'en'.")
    return model


def transcribe(model, wav_path, lang):
    if lang == "zh":
        import zhconv

        res = model.generate(input=str(wav_path), batch_size_s=300, disable_pbar=True)
        hypo = res[0]["text"]
        hypo = zhconv.convert(hypo, "zh-cn")
    elif lang == "en":
        segments, _ = model.transcribe(str(wav_path), beam_size=5, language="en")
        hypo = " ".join(segment.text for segment in segments)
    return hypo.strip()


def main():
    parser = argparse.ArgumentParser(description="WER Evaluation on HIED benchmark")
    parser.add_argument("--gen_wav_dir", type=str, required=True, help="Directory containing generated wav files.")
    parser.add_argument("--manifest", type=str, default=None, help="Path to HIED manifest.jsonl for ground truth text.")
    parser.add_argument("--lang", type=str, default="en", choices=["zh", "en"], help="Language for ASR model.")
    parser.add_argument("--gpu_nums", type=int, default=1, help="Number of GPUs to use.")
    parser.add_argument("--ext", type=str, default="wav", help="Audio file extension.")
    args = parser.parse_args()

    from jiwer import process_words

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading ASR model ({args.lang})...")
    asr_model = load_asr_model(args.lang, device=device)

    gen_wav_dir = Path(args.gen_wav_dir)
    audio_paths = sorted(gen_wav_dir.rglob(f"*.{args.ext}"))
    print(f"Found {len(audio_paths)} audio files in {gen_wav_dir}")

    # Load ground truth from manifest if provided
    gt_texts = {}
    if args.manifest:
        with open(args.manifest) as f:
            for line in f:
                item = json.loads(line.strip())
                gt_texts[item["id"]] = item.get("text", item.get("gen_text", ""))

    import string

    try:
        from zhon.hanzi import punctuation as zh_punctuation

        all_punctuation = zh_punctuation + string.punctuation
    except ImportError:
        all_punctuation = string.punctuation

    wer_results = []
    for wav_path in tqdm(audio_paths, desc="Evaluating WER"):
        utt_id = wav_path.stem
        hypo = transcribe(asr_model, wav_path, args.lang)

        truth = gt_texts.get(utt_id, "")
        if not truth:
            continue

        raw_truth, raw_hypo = truth, hypo

        for p in all_punctuation:
            truth = truth.replace(p, "")
            hypo = hypo.replace(p, "")
        truth = truth.replace("  ", " ").strip()
        hypo = hypo.replace("  ", " ").strip()

        if args.lang == "zh":
            truth = " ".join(list(truth))
            hypo = " ".join(list(hypo))
        else:
            truth = truth.lower()
            hypo = hypo.lower()

        if not truth:
            continue

        measures = process_words(truth, hypo)
        wer_results.append({
            "wav": utt_id,
            "truth": raw_truth,
            "hypo": raw_hypo,
            "wer": measures.wer,
        })

    result_path = gen_wav_dir / "_wer_results.jsonl"
    with open(result_path, "w", encoding="utf-8") as f:
        for line in wer_results:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
        avg_wer = np.mean([r["wer"] for r in wer_results]) if wer_results else 0
        f.write(f"\nWER: {avg_wer:.5f}\n")

    print(f"\nTotal {len(wer_results)} samples")
    print(f"WER: {avg_wer:.5f} ({avg_wer * 100:.2f}%)")
    print(f"Results saved to {result_path}")


if __name__ == "__main__":
    main()
