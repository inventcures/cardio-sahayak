# Cardio-Sahayak India 🇮🇳 🫀

[![DeepWiki](https://img.shields.io/badge/DeepWiki-Indexed-blue.svg)](https://deepwiki.com)
[![CodeWiki](https://img.shields.io/badge/CodeWiki-Indexed-green.svg)](https://codewiki.com)

**Cardio-Sahayak India** is a specialized Large Language Model (LLM) and Vision-Language Model (VLM) designed for complex cardiology care, heavily optimized for the Indian population and its specific demographic factors (e.g., South Asian cardiac mutation markers).

## Project Overview

This project builds upon the state-of-the-art **MedGemma-27B** backbone, adapting it through a rigorous two-phase training methodology:
1. **Phase 1 (General Medical Reasoning):** Text and Multimodal SFT integrating Indian National Consensus on Cardiology guidelines and MedSigLIP vision encoders for 12-lead ECG analysis.
2. **Phase 2 (South Asian Contextualization):** Resumed fine-tuning on a curated 166-record *V2 Dataset* integrating real-world Indian clinical notes (EkaCare), Gemini-driven synthetic phenotype shifts, and multimodal scanned ECG references to deeply embed the "South Asian Phenotype" (e.g., lower BMI thresholds, MYBPC3 Δ25bp genetic markers).

## Architecture & Deployment

- **Base Model:** `google/medgemma-27b-it` (Gemma3 VLM Architecture)
- **Vision Encoder:** `google/medsiglip-448`
- **Fine-Tuning Strategy:** Two-Phase QLoRA (4-bit NormalFloat via bitsandbytes)
- **Compute:** Modal.com (`A100-80GB` GPUs)
- **Edge Deployment:** We developed a runtime architectural patch to convert the complex Gemma3 language model into a highly compressed **Q4_K_M GGUF (16.6 GB)** format using `llama.cpp`, enabling fully offline subspecialist AI on standard clinical laptops.

## Repository Structure

- `data_prep_indian_cardio.py`: Generates the Phase 1 Hugging Face dataset.
- `ingest_eka_notes.py`, `synthetic_phenotype_shifter.py`, `compile_v2_dataset.py`: Scripts used to generate the expanded Phase 2 V2 dataset.
- `finetune_cardio_sahayak.py` / `modal_train_cardio_sahayak.py`: Scripts for Phase 1 fine-tuning.
- `modal_train_cardio_sahayak_v2.py`: Phase 2 training script, resuming from Phase 1 adapters.
- `modal_gguf_convert.py`: Automated Modal pipeline for patching and generating GGUF files.
- `vlm_ecg_inference.py`: Inference script for generating clinical reports from ECG images.

## Datasets & Weights

- **Instruction Datasets:** 
  - Phase 1: `tp53/cardio-sahayak-india-instruct-v0`
  - Phase 2: `tp53/cardio-sahayak-india-instruct-v2`
- **Vision Dataset:** `PULSE-ECG/ECGBench` (ptb-test-report subset)
- **Model Weights (Public - CC-BY-4.0 with Attribution):**
  - Text Adapters (Phase 2): `tp53/cardio-sahayak` (in the `v2_weights/` subfolder)
  - VLM Adapters: `tp53/cardio-sahayak-vlm`
  - GGUF Offline Weights: `tp53/cardio-sahayak-gguf`

## Quick Start

### 1. Set Up Environment
```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt # Ensure Modal, Transformers, TRL, PEFT are installed
```

### 2. Configure Secrets
Ensure your Hugging Face token and Modal credentials are in `.env` or set up in Modal:
```bash
modal secret create huggingface-secret HF_TOKEN=your_token_here
```

### 3. Run Inference
To test the multimodal ECG capabilities:
```bash
python vlm_ecg_inference.py
```

## Future Roadmap

1. **GGUF Conversion:** Export adapters and base model to GGUF format for low-resource deployment in rural Indian clinics.
2. **Validation:** Benchmark the fine-tuned models against PubMedQA, MedQA, and standard cardiology board exam datasets.
3. **UI Integration:** Build a Gradio/Streamlit interface for direct physician interaction.

---
*Developed with ❤️ for better cardiac care in India.*