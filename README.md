# 🎭 Rectifying the Emotional Flow

<p align="center">
  <img src="https://img.shields.io/badge/🐍 Python-3.10+-blue" alt="python">
  <img src="https://img.shields.io/badge/🔥 PyTorch-2.x-orange" alt="pytorch">
  <img src="https://img.shields.io/badge/Speech-Synthesis-green" alt="speech">
  <img src="https://img.shields.io/badge/Venue-ACL 2026-purple" alt="venue">
  <a href="#"><img src="https://img.shields.io/badge/📄 arXiv-paper-b31b1b" alt="arxiv"></a>
  <a href="https://huggingface.co/datasets/erminga/emo-tts"><img src="https://img.shields.io/badge/🤗 Dataset-erminga/emo--tts-yellow" alt="dataset"></a>
</p>

## 📝 Abstract

While diffusion and flow-matching models have advanced TTS, generating high-arousal emotions remains a persistent challenge due to the trade-off between stability and expressiveness. Existing systems often suffer from linguistic collapse when pursuing high intensity or fail to meet target emotional levels under stable settings. In this work, we identify that standard Gaussian initialization inevitably introduces a neutral prosody bias, while uniform Classifier-Free Guidance often distorts the acoustic manifold, leading to artifacts. To address this, we propose an inference framework that rectifies the emotional trajectory. An **Emotion-Rectified Noise Prior** injects a semantic gradient at initialization to align sampling with the target emotional manifold, and **Likelihood-Inverse Guidance** adaptively schedules guidance via a conditional/unconditional likelihood ratio, strengthening guidance only when the trajectory drifts toward a neutral fallback. Extensive experiments demonstrate that our method effectively resolves the stability bottleneck in high-intensity scenarios, achieving superior linguistic accuracy and emotional fidelity without model retraining. 

## 🔥 Highlights

- 🎯 **Zero retraining** — pure inference-time enhancement, works on any Flow-Matching TTS
- 🧭 **ERNP** (Emotion-Rectified Noise Prior) — steers initial noise toward emotional manifold via `lookahead → calibration → re-normalization`
- 📈 **LIG** (Likelihood-Inverse Guidance) — replaces constant CFG with `dynamic λ(t)` derived from recursive likelihood-ratio estimation
- ⚡ **No extra model calls** — LIG reuses existing conditional/unconditional velocity fields
- 🏆 **SOTA results** — WER 4.41% → **2.53%**, EMOS 3.63 → **3.89** on HIED benchmark
- 🔌 **Plug-and-play** — validated on CosyVoice2, IndexTTS2, and F5-TTS architectures

## 🏗️ Method

<p align="center">
  <img src="method.png" width="90%" alt="Method Overview">
</p>

Our framework operates entirely at **inference time** with zero retraining, consisting of two complementary components:

### ERNP (Emotion-Rectified Noise Prior)

Rectifies the initial Gaussian noise *before* the ODE solve via a two-step lookahead–calibration cycle:

1. **Lookahead** — forward one step from $x_0$ with high guidance strength $\lambda_{\text{init}}$:  $\quad x_\tau = x_0 + \tau \cdot \tilde{v}_{\lambda_{\text{init}}}(x_0, 0)$
2. **Calibration** — backward one step with base guidance $\lambda_{\text{base}}$:  $\quad x_0^* = x_\tau - \tau \cdot \tilde{v}_{\lambda_{\text{base}}}(x_\tau, \tau)$
3. **Re-normalization** — strictly standardize $x_0^*$ back to $\mathcal{N}(0, I)$

The net effect is a controlled displacement along the **emotional semantic gradient**, steering the starting point toward the target emotional manifold.

### LIG (Likelihood-Inverse Guidance)

Replaces constant CFG with a **dynamic, trajectory-aware** guidance schedule. We model the learned conditional distribution as an additive mixture of neutral and emotional components, and derive the per-step guidance strength:

$$\lambda(x_t, t) = \frac{R_t}{R_t - (1-\pi)}$$

where $R_t$ is the **likelihood ratio** estimated recursively from the conditional/unconditional velocity field divergence — no additional model calls required. When the trajectory is already in the emotional region ($R_t \gg 1$), guidance stays minimal; when it drifts toward neutral ($R_t \to 1-\pi$), guidance increases sharply to correct the course.

```
x₀ ~ N(0,I) ──[ERNP]──▶ Rectified x₀* ──[LIG: dynamic λ(t)]──▶ Emotional speech x₁
```

## 📦 Installation

```bash
# Create environment
conda create -n emo-tts python=3.11
conda activate emo-tts
conda install ffmpeg

# Install PyTorch (match your CUDA version)
pip install torch==2.8.0+cu128 torchaudio==2.8.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128

# Install from source
cd Emo-TTS
pip install -e .
```

## 📁 Code Structure

```
Emo-TTS/
├── src/emo_tts/
│   ├── model/
│   │   ├── cfm.py              # Core: CFM sampling with ERNP + LIG
│   │   ├── backbones/          # DiT, MMDiT, UNet-T
│   │   ├── modules.py          # Mel spectrogram, attention, etc.
│   │   ├── trainer.py          # Training loop
│   │   └── utils.py            # Utilities
│   ├── configs/                # Model architecture YAML configs
│   ├── infer/
│   │   ├── infer_cli.py        # CLI inference
│   │   ├── infer_gradio.py     # Gradio web UI
│   │   ├── infer_emo_test.py   # ERNP + LIG experiment inference
│   │   └── utils_infer.py      # Inference utilities
│   ├── train/                  # Training & finetuning scripts
│   ├── eval/                   # Evaluation tools (WER, EMOS, UTMOS)
│   └── runtime/                # Triton + TensorRT-LLM deployment
├── method.png
├── pyproject.toml
└── README.md
```

> **Key file:** `src/emo_tts/model/cfm.py` — the `CFM.sample()` method integrates both ERNP (noise rectification before ODE) and LIG (dynamic guidance inside ODE).

## 🚀 Inference

```bash
# CLI inference
emo-tts_infer-cli --model EmoTTS_v1_Base \
  --ref_audio "path/to/reference.wav" \
  --ref_text "Transcription of the reference audio." \
  --gen_text "Text you want to synthesize."
```

```bash
# Gradio web UI
emo-tts_infer-gradio
```

```python
# Python API
from emo_tts.api import EmoTTS

tts = EmoTTS(model="EmoTTS_v1_Base")
wav, sr, spec = tts.infer(
    ref_file="path/to/reference.wav",
    ref_text="Transcription of the reference audio.",
    gen_text="Text you want to synthesize.",
    file_wave="output.wav",
)
```

## 🧪 Experiments

We evaluate on the **HIED benchmark** (400 high-arousal emotional samples) across three TTS architectures: **F5-TTS**, **CosyVoice2**, and **IndexTTS2**. The two core metrics are:

- **WER** (↓) — Word Error Rate via ASR, measuring linguistic accuracy
- **EMOS** (↑) — Emotion Score via emotion classifier, measuring emotional fidelity

### 📊 Datasets

All datasets used in this work are publicly available at 🤗 [erminga/emo-tts](https://huggingface.co/datasets/erminga/emo-tts).

#### HIED Benchmark (ours)

**HIED** (High-Intensity Emotional Dataset) is our curated evaluation benchmark specifically designed to stress-test TTS systems under high-arousal emotional conditions.

| | Details |
|---|---|
| **Total samples** | 400 (100 per emotion) |
| **Emotions** | Angry, Happy, Sad, Surprise |
| **Sources** | ESD (354 samples), EmoV-DB (46 samples) |
| **Avg duration** | 3.85 s |
| **Total duration** | ~0.43 h |
| **Acoustic features** | RMS energy, F0 mean/std/range, speaking rate |

```python
# Load HIED directly
from datasets import load_dataset
hied = load_dataset("erminga/emo-tts", "HIED", split="test")
```

<details>
<summary><b>HIED sample fields</b></summary>

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique ID (`HIED_0000` … `HIED_0399`) |
| `audio` | audio | Speech waveform |
| `emotion` | string | Emotion class (`Angry` / `Happy` / `Sad` / `Surprise`) |
| `source_dataset` | string | Origin (`ESD` / `EmoV-DB`) |
| `speaker` | string | Speaker identifier |
| `rms_energy` | float | RMS energy |
| `f0_mean` | float | Mean fundamental frequency (Hz) |
| `f0_std` | float | F0 standard deviation |
| `f0_range` | float | F0 range (Hz) |
| `speaking_rate` | float | Speaking rate (phonemes/s) |
| `duration` | float | Duration (seconds) |

</details>

#### Source Datasets

| Dataset | Emotions | Speakers | Language | Reference |
|---------|----------|----------|----------|-----------|
| **ESD** | Neutral, Happy, Sad, Angry, Surprise | 10 EN + 10 ZH | EN / ZH | [Zhou et al., 2022](https://github.com/HLTSingapore/Emotional-Speech-Data) |
| **EmoV-DB** | Neutral, Amused, Angry, Sleepy, Disgusted | 4 (bea, jenie, josh, sam) | EN / FR | [OpenSLR-115](https://www.openslr.org/115/) · [Adigwe et al., 2018](https://arxiv.org/abs/1806.09514) |
| **Expresso** | 8 read + 26 improvised styles | 4 (2M, 2F), 48kHz | EN | [ylacombe/expresso](https://huggingface.co/datasets/ylacombe/expresso) · [Nguyen et al., 2023](https://arxiv.org/abs/2308.05725) |

> All three source datasets, along with the HIED benchmark, are mirrored in our HuggingFace repository for one-stop download:
> ```bash
> # Download everything (~10 GB)
> huggingface-cli download erminga/emo-tts --repo-type dataset --local-dir ./emo-tts-data
> ```

### Reproduce Results

**Step 1.** Download the HIED dataset:

```python
from datasets import load_dataset
hied = load_dataset("erminga/emo-tts", "HIED", split="test")
```

**Step 2.** Run inference (Baseline vs. ERNP + LIG ablations):

```bash
# Baseline — standard CFG
python src/emo_tts/infer/infer_emo_test.py \
    --config configs/emo_infer.yaml \
    --output_dir results/baseline

# ERNP only — emotion-rectified noise prior
python src/emo_tts/infer/infer_emo_test.py \
    --config configs/emo_infer.yaml \
    --ernp_lambda_init 50.0 --ernp_lambda_base 2.0 \
    --output_dir results/ernp_only

# LIG only — likelihood-inverse guidance
python src/emo_tts/infer/infer_emo_test.py \
    --config configs/emo_infer.yaml \
    --lig_pi 0.99 --lig_lambda_max 15.0 --lig_sigma 0.5 \
    --output_dir results/lig_only

# Full method: ERNP + LIG
python src/emo_tts/infer/infer_emo_test.py \
    --config configs/emo_infer.yaml \
    --ernp_lambda_init 50.0 --ernp_lambda_base 2.0 \
    --lig_pi 0.99 --lig_lambda_max 15.0 --lig_sigma 0.5 \
    --output_dir results/ernp_lig
```

**Step 3.** Evaluate (WER + EMOS + UTMOSv2):

```bash
pip install -e .[eval]

# WER — Word Error Rate
#   EN: Whisper (Radford et al., 2023)
#   ZH: FunASR (Gao et al., 2023)
python src/emo_tts/eval/eval_wer.py \
    --gen_wav_dir results/ernp_lig \
    --gpu_nums 8

# EMOS — Emotion Score via emotion2vec (Ma et al., 2024)
python src/emo_tts/eval/eval_emos.py \
    --gen_wav_dir results/ernp_lig

# UTMOSv2 — Speech Quality (MOS prediction)
python src/emo_tts/eval/eval_utmos.py \
    --audio_dir results/ernp_lig --ext wav
```

<summary><b>Evaluation tools & checkpoints</b></summary>

| Tool | Purpose | Source |
|------|---------|--------|
| **Whisper** | English ASR (WER) | [openai/whisper](https://github.com/openai/whisper) · [Radford et al., 2023](https://arxiv.org/abs/2212.04356) |
| **FunASR** | Chinese ASR (WER) | [modelscope/FunASR](https://github.com/modelscope/FunASR) · [Gao et al., 2023](https://arxiv.org/abs/2305.11013) |
| **emotion2vec** | Emotion Score (EMOS) | [ddlBoJack/emotion2vec](https://github.com/ddlBoJack/emotion2vec) · [Ma et al., 2024](https://arxiv.org/abs/2312.15185) |
| **UTMOSv2** | Speech Quality (MOS) | [sarulab-speech/UTMOSv2](https://github.com/sarulab-speech/UTMOSv2) · [HF](https://huggingface.co/sarulab-speech/UTMOSv2) |


## 📜 License

Code is released under the [MIT License](LICENSE).

