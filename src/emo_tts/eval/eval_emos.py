"""
EMOS (Emotion Score) Evaluation for HIED benchmark.

Uses emotion2vec+ (Ma et al., 2024) via FunASR for speech emotion recognition.
Computes the average confidence score of the target emotion class.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm


# emotion2vec label mapping (9 classes)
EMOTION2VEC_LABELS = {
    0: "angry",
    1: "disgusted",
    2: "fearful",
    3: "happy",
    4: "neutral",
    5: "other",
    6: "sad",
    7: "surprised",
    8: "unknown",
}


def main():
    parser = argparse.ArgumentParser(description="EMOS Evaluation using emotion2vec+")
    parser.add_argument("--gen_wav_dir", type=str, required=True, help="Directory containing generated wav files.")
    parser.add_argument("--manifest", type=str, default=None, help="Path to HIED manifest.jsonl for target emotions.")
    parser.add_argument(
        "--model",
        type=str,
        default="iic/emotion2vec_plus_large",
        help="emotion2vec model ID (default: iic/emotion2vec_plus_large)",
    )
    parser.add_argument("--ext", type=str, default="wav", help="Audio file extension.")
    args = parser.parse_args()

    from funasr import AutoModel

    print(f"Loading emotion2vec model: {args.model}")
    model = AutoModel(model=args.model, hub="hf")

    gen_wav_dir = Path(args.gen_wav_dir)
    audio_paths = sorted(gen_wav_dir.rglob(f"*.{args.ext}"))
    print(f"Found {len(audio_paths)} audio files in {gen_wav_dir}")

    # Load target emotions from manifest if provided
    target_emotions = {}
    if args.manifest:
        with open(args.manifest) as f:
            for line in f:
                item = json.loads(line.strip())
                target_emotions[item["id"]] = item.get("emotion_class", item.get("emotion", "")).lower()

    emos_results = []
    for wav_path in tqdm(audio_paths, desc="Evaluating EMOS"):
        utt_id = wav_path.stem

        rec_result = model.generate(
            str(wav_path),
            granularity="utterance",
            extract_embedding=False,
        )

        if not rec_result or len(rec_result) == 0:
            continue

        result = rec_result[0]
        labels = result.get("labels", [])
        scores = result.get("scores", [])

        if not labels or not scores:
            continue

        # Build emotion score dict
        emotion_scores = {}
        for label, score in zip(labels, scores):
            emotion_scores[label.lower()] = score

        # Predicted emotion (highest score)
        predicted_emotion = labels[int(np.argmax(scores))].lower()
        predicted_score = float(np.max(scores))

        # Target emotion match score
        target_emo = target_emotions.get(utt_id, "")
        target_score = emotion_scores.get(target_emo, 0.0) if target_emo else predicted_score

        emos_results.append({
            "wav": utt_id,
            "predicted_emotion": predicted_emotion,
            "predicted_score": round(predicted_score, 4),
            "target_emotion": target_emo,
            "target_score": round(float(target_score), 4),
            "all_scores": {k: round(v, 4) for k, v in emotion_scores.items()},
        })

    result_path = gen_wav_dir / "_emos_results.jsonl"
    with open(result_path, "w", encoding="utf-8") as f:
        for line in emos_results:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

        # Compute average scores
        avg_target_score = np.mean([r["target_score"] for r in emos_results]) if emos_results else 0
        avg_predicted_score = np.mean([r["predicted_score"] for r in emos_results]) if emos_results else 0

        # Accuracy: predicted == target
        if target_emotions:
            correct = sum(1 for r in emos_results if r["predicted_emotion"] == r["target_emotion"])
            accuracy = correct / len(emos_results) if emos_results else 0
        else:
            accuracy = None

        summary = {
            "avg_target_score": round(float(avg_target_score), 4),
            "avg_predicted_score": round(float(avg_predicted_score), 4),
        }
        if accuracy is not None:
            summary["emotion_accuracy"] = round(accuracy, 4)

        f.write(f"\nEMOS: {json.dumps(summary)}\n")

    print(f"\nTotal {len(emos_results)} samples")
    print(f"Avg target emotion score (EMOS): {avg_target_score:.4f}")
    print(f"Avg predicted emotion score:     {avg_predicted_score:.4f}")
    if accuracy is not None:
        print(f"Emotion accuracy:                {accuracy:.4f} ({accuracy * 100:.1f}%)")
    print(f"Results saved to {result_path}")


if __name__ == "__main__":
    main()
