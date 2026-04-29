"""
UTMOSv2 Evaluation — Speech Quality (MOS prediction).

Uses UTMOSv2 (Baba et al., IEEE SLT 2024) from sarulab-speech.
Install: pip install git+https://github.com/sarulab-speech/UTMOSv2.git
"""

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="UTMOSv2 Evaluation")
    parser.add_argument("--audio_dir", type=str, required=True, help="Directory containing audio files.")
    parser.add_argument("--ext", type=str, default="wav", help="Audio extension.")
    args = parser.parse_args()

    import utmosv2

    print("Loading UTMOSv2 model...")
    model = utmosv2.create_model(pretrained=True)

    audio_dir = Path(args.audio_dir)
    audio_paths = sorted(audio_dir.rglob(f"*.{args.ext}"))
    print(f"Found {len(audio_paths)} audio files in {audio_dir}")

    utmos_results = []
    for audio_path in tqdm(audio_paths, desc="Evaluating UTMOSv2"):
        score = model.predict(input_path=str(audio_path))
        score_val = float(score)
        utmos_results.append({
            "wav": audio_path.stem,
            "utmos": round(score_val, 4),
        })

    result_path = audio_dir / "_utmos_results.jsonl"
    with open(result_path, "w", encoding="utf-8") as f:
        for line in utmos_results:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
        avg_score = np.mean([r["utmos"] for r in utmos_results]) if utmos_results else 0
        f.write(f"\nUTMOSv2: {avg_score:.4f}\n")

    print(f"\nUTMOSv2: {avg_score:.4f}")
    print(f"Results saved to {result_path}")


if __name__ == "__main__":
    main()
