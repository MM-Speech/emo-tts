<!-- omit in toc -->
# Shared Model Cards

<!-- omit in toc -->
### **Prerequisites of using**
- This document is serving as a quick lookup table for the community training/finetuning result, with various language support.
- The models in this repository are open source and are based on voluntary contributions from contributors.
- The use of models must be conditioned on respect for the respective creators. The convenience brought comes from their efforts.

<!-- omit in toc -->
### **Welcome to share here**
- Have a pretrained/finetuned result: model checkpoint (pruned best to facilitate inference, i.e. leave only `ema_model_state_dict`) and corresponding vocab file (for tokenization).
- Host a public [huggingface model repository](https://huggingface.co/new) and upload the model related files.
- Make a pull request adding a model card to the current page, i.e. `src\emo_tts\infer\SHARED.md`.

<!-- omit in toc -->
### Supported Languages
- [Multilingual](#multilingual)
    - [Emo-TTS v1 v0 Base @ zh \& en @ Emo-TTS](#emo-tts-v1-v0-base--zh--en--emo-tts)
- [Arabic](#arabic)
    - [Emo-TTS Small @ ar & en @ SILMA AI](#emo-tts-small--ar--en--silma-ai)
- [English](#english)
- [Finnish](#finnish)
    - [Emo-TTS Base @ fi @ AsmoKoskinen](#emo-tts-base--fi--asmokoskinen)
- [French](#french)
    - [Emo-TTS Base @ fr @ RASPIAUDIO](#emo-tts-base--fr--raspiaudio)
- [German](#german)
    - [Emo-TTS Base @ de @ hvoss-techfak](#emo-tts-base--de--hvoss-techfak)
- [Hindi](#hindi)
    - [Emo-TTS Small @ hi @ SPRINGLab](#emo-tts-small--hi--springlab)
- [Italian](#italian)
    - [Emo-TTS Base @ it @ alien79](#emo-tts-base--it--alien79)
- [Japanese](#japanese)
    - [Emo-TTS Base @ ja @ Jmica](#emo-tts-base--ja--jmica)
- [Latvian](#latvian)
    - [Emo-TTS Base @ lv @ RaivisDejus](#emo-tts-base--lv--raivisdejus)
- [Mandarin](#mandarin)
- [Russian](#russian)
    - [Emo-TTS Base @ ru @ HotDro4illa](#emo-tts-base--ru--hotdro4illa)
- [Spanish](#spanish)
    - [Emo-TTS Base @ es @ jpgallegoar](#emo-tts-base--es--jpgallegoar)


## Multilingual

#### Emo-TTS v1 v0 Base @ zh & en @ Emo-TTS
|Model|🤗Hugging Face|Data (Hours)|Model License|
|:---:|:------------:|:-----------:|:-------------:|
|Emo-TTS v1 Base|[ckpt & vocab](https://huggingface.co/SWivid/Emo-TTS/tree/main/EmoTTS_v1_Base)|[Emilia 95K zh&en](https://huggingface.co/datasets/amphion/Emilia-Dataset/tree/fc71e07)|cc-by-nc-4.0|

```bash
Model: hf://SWivid/Emo-TTS/EmoTTS_v1_Base/model_1250000.safetensors
# A Variant Model: hf://SWivid/Emo-TTS/EmoTTS_v1_Base_no_zero_init/model_1250000.safetensors
Vocab: hf://SWivid/Emo-TTS/EmoTTS_v1_Base/vocab.txt
Config: {"dim": 1024, "depth": 22, "heads": 16, "ff_mult": 2, "text_dim": 512, "conv_layers": 4}
```

|Model|🤗Hugging Face|Data (Hours)|Model License|
|:---:|:------------:|:-----------:|:-------------:|
|Emo-TTS Base|[ckpt & vocab](https://huggingface.co/SWivid/Emo-TTS/tree/main/EmoTTS_Base)|[Emilia 95K zh&en](https://huggingface.co/datasets/amphion/Emilia-Dataset/tree/fc71e07)|cc-by-nc-4.0|

```bash
Model: hf://SWivid/Emo-TTS/EmoTTS_Base/model_1200000.safetensors
Vocab: hf://SWivid/Emo-TTS/EmoTTS_Base/vocab.txt
Config: {"dim": 1024, "depth": 22, "heads": 16, "ff_mult": 2, "text_dim": 512, "text_mask_padding": False, "conv_layers": 4, "pe_attn_head": 1}
```

*Other infos, e.g. Author info, Github repo, Link to some sampled results, Usage instruction, Tutorial (Blog, Video, etc.) ...*


## Arabic

#### Emo-TTS Small @ ar & en @ SILMA AI
|Model|🤗Hugging Face|Data (Hours)|Model License|
|:---:|:------------:|:-----------:|:-------------:|
|Emo-TTS Small|[ckpt & vocab](https://huggingface.co/silma-ai/silma-tts)| Tens of thousands EN/AR |Apache-2.0|

- Pretrained by [SILMA.AI](https://silma.ai)
- [GitHub repo](https://github.com/SILMA-AI/silma-tts), Inference code


## English


## Finnish

#### Emo-TTS Base @ fi @ AsmoKoskinen
|Model|🤗Hugging Face|Data|Model License|
|:---:|:------------:|:-----------:|:-------------:|
|Emo-TTS Base|[ckpt & vocab](https://huggingface.co/AsmoKoskinen/Emo-TTS_Finnish_Model)|[Common Voice](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0), [Vox Populi](https://huggingface.co/datasets/facebook/voxpopuli)|cc-by-nc-4.0|

```bash
Model: hf://AsmoKoskinen/Emo-TTS_Finnish_Model/model_common_voice_fi_vox_populi_fi_20241206.safetensors
Vocab: hf://AsmoKoskinen/Emo-TTS_Finnish_Model/vocab.txt
Config: {"dim": 1024, "depth": 22, "heads": 16, "ff_mult": 2, "text_dim": 512, "text_mask_padding": False, "conv_layers": 4, "pe_attn_head": 1}
```


## French

#### Emo-TTS Base @ fr @ RASPIAUDIO
|Model|🤗Hugging Face|Data (Hours)|Model License|
|:---:|:------------:|:-----------:|:-------------:|
|Emo-TTS Base|[ckpt & vocab](https://huggingface.co/RASPIAUDIO/F5-French-MixedSpeakers-reduced)|[LibriVox](https://librivox.org/)|cc-by-nc-4.0|

```bash
Model: hf://RASPIAUDIO/F5-French-MixedSpeakers-reduced/model_last_reduced.pt
Vocab: hf://RASPIAUDIO/F5-French-MixedSpeakers-reduced/vocab.txt
Config: {"dim": 1024, "depth": 22, "heads": 16, "ff_mult": 2, "text_dim": 512, "text_mask_padding": False, "conv_layers": 4, "pe_attn_head": 1}
```

- [Online Inference with Hugging Face Space](https://huggingface.co/spaces/RASPIAUDIO/emo-tts_french).
- [Tutorial video to train a new language model](https://www.youtube.com/watch?v=UO4usaOojys).
- [Discussion about this training can be found here](https://github.com/SWivid/Emo-TTS/issues/434).


## German

#### Emo-TTS Base @ de @ hvoss-techfak
|Model|🤗Hugging Face|Data (Hours)|Model License|
|:---:|:------------:|:-----------:|:-------------:|
|Emo-TTS Base|[ckpt & vocab](https://huggingface.co/hvoss-techfak/Emo-TTS-German)|[Mozilla Common Voice 19.0](https://commonvoice.mozilla.org/en/datasets) & 800 hours Crowdsourced |cc-by-nc-4.0|

```bash
Model: hf://hvoss-techfak/Emo-TTS-German/model_emotts_german.pt
Vocab: hf://hvoss-techfak/Emo-TTS-German/vocab.txt
Config: {"dim": 1024, "depth": 22, "heads": 16, "ff_mult": 2, "text_dim": 512, "text_mask_padding": False, "conv_layers": 4, "pe_attn_head": 1}
```

- Finetuned by [@hvoss-techfak](https://github.com/hvoss-techfak)


## Hindi

#### Emo-TTS Small @ hi @ SPRINGLab
|Model|🤗Hugging Face|Data (Hours)|Model License|
|:---:|:------------:|:-----------:|:-------------:|
|Emo-TTS Small|[ckpt & vocab](https://huggingface.co/SPRINGLab/F5-Hindi-24KHz)|[IndicTTS Hi](https://huggingface.co/datasets/SPRINGLab/IndicTTS-Hindi) & [IndicVoices-R Hi](https://huggingface.co/datasets/SPRINGLab/IndicVoices-R_Hindi) |cc-by-4.0|

```bash
Model: hf://SPRINGLab/F5-Hindi-24KHz/model_2500000.safetensors
Vocab: hf://SPRINGLab/F5-Hindi-24KHz/vocab.txt
Config: {"dim": 768, "depth": 18, "heads": 12, "ff_mult": 2, "text_dim": 512, "text_mask_padding": False, "conv_layers": 4, "pe_attn_head": 1}
```

- Authors: SPRING Lab, Indian Institute of Technology, Madras
- Website: https://asr.iitm.ac.in/


## Italian

#### Emo-TTS Base @ it @ alien79
|Model|🤗Hugging Face|Data|Model License|
|:---:|:------------:|:-----------:|:-------------:|
|Emo-TTS Base|[ckpt & vocab](https://huggingface.co/alien79/Emo-TTS-italian)|[ylacombe/cml-tts](https://huggingface.co/datasets/ylacombe/cml-tts) |cc-by-nc-4.0|

```bash
Model: hf://alien79/Emo-TTS-italian/model_159600.safetensors
Vocab: hf://alien79/Emo-TTS-italian/vocab.txt
Config: {"dim": 1024, "depth": 22, "heads": 16, "ff_mult": 2, "text_dim": 512, "text_mask_padding": False, "conv_layers": 4, "pe_attn_head": 1}
```

- Trained by [Mithril Man](https://github.com/MithrilMan)
- Model details on [hf project home](https://huggingface.co/alien79/Emo-TTS-italian)
- Open to collaborations to further improve the model


## Japanese

#### Emo-TTS Base @ ja @ Jmica
|Model|🤗Hugging Face|Data (Hours)|Model License|
|:---:|:------------:|:-----------:|:-------------:|
|Emo-TTS Base|[ckpt & vocab](https://huggingface.co/Jmica/EmoTTS/tree/main/JA_21999120)|[Emilia 1.7k JA](https://huggingface.co/datasets/amphion/Emilia-Dataset/tree/fc71e07) & [Galgame Dataset 5.4k](https://huggingface.co/datasets/OOPPEENN/Galgame_Dataset)|cc-by-nc-4.0|

```bash
Model: hf://Jmica/EmoTTS/JA_21999120/model_21999120.pt
Vocab: hf://Jmica/EmoTTS/JA_21999120/vocab_japanese.txt
Config: {"dim": 1024, "depth": 22, "heads": 16, "ff_mult": 2, "text_dim": 512, "text_mask_padding": False, "conv_layers": 4, "pe_attn_head": 1}
```


## Latvian

#### Emo-TTS Base @ lv @ RaivisDejus
|Model|🤗Hugging Face|Data (Hours)|Model License|
|:---:|:------------:|:-----------:|:-------------:|
|Emo-TTS Base|[ckpt & vocab](https://huggingface.co/RaivisDejus/Emo-TTS-Latvian)|[Common voice](https://datacollective.mozillafoundation.org/datasets/cmj8u3pec00flnxxbntvfb4as)|cc0-1.0|

```bash
Model: hf://RaivisDejus/Emo-TTS-Latvian/model.safetensors
Vocab: hf://RaivisDejus/Emo-TTS-Latvian/vocab.txt
Config: {"dim": 1024, "depth": 22, "heads": 16, "ff_mult": 2, "text_dim": 512, "text_mask_padding": False, "conv_layers": 4, "pe_attn_head": 1}
```


## Mandarin


## Russian

#### Emo-TTS Base @ ru @ HotDro4illa
|Model|🤗Hugging Face|Data (Hours)|Model License|
|:---:|:------------:|:-----------:|:-------------:|
|Emo-TTS Base|[ckpt & vocab](https://huggingface.co/hotstone228/Emo-TTS-Russian)|[Common voice](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0)|cc-by-nc-4.0|

```bash
Model: hf://hotstone228/Emo-TTS-Russian/model_last.safetensors
Vocab: hf://hotstone228/Emo-TTS-Russian/vocab.txt
Config: {"dim": 1024, "depth": 22, "heads": 16, "ff_mult": 2, "text_dim": 512, "text_mask_padding": False, "conv_layers": 4, "pe_attn_head": 1}
```
- Finetuned by [HotDro4illa](https://github.com/HotDro4illa)
- Any improvements are welcome


## Spanish

#### Emo-TTS Base @ es @ jpgallegoar
|Model|🤗Hugging Face|Data (Hours)|Model License|
|:---:|:------------:|:-----------:|:-------------:|
|Emo-TTS Base|[ckpt & vocab](https://huggingface.co/jpgallegoar/F5-Spanish)|[Voxpopuli](https://huggingface.co/datasets/facebook/voxpopuli) & Crowdsourced & TEDx, 218 hours|cc0-1.0|

- @jpgallegoar [GitHub repo](https://github.com/jpgallegoar/Spanish-F5), Jupyter Notebook and Gradio usage for Spanish model.
