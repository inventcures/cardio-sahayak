# Cardio-Sahayak India 🇮🇳 🫀

[![DeepWiki](https://img.shields.io/badge/DeepWiki-Indexed-blue.svg)](https://deepwiki.com)
[![CodeWiki](https://img.shields.io/badge/CodeWiki-Indexed-green.svg)](https://codewiki.com)

**Cardio-Sahayak India** is a specialized Large Language Model (LLM) and Vision-Language Model (VLM) designed for complex cardiology care, heavily optimized for the Indian population and its specific demographic factors (e.g., South Asian cardiac mutation markers).

## Project Overview

This project builds upon the state-of-the-art **MedGemma-27B** backbone, adapting it through:
1. **Text Supervised Fine-Tuning (SFT):** Fine-tuning on specialized instruction sets integrating Indian National Consensus on Cardiology guidelines and South Asian genetic contexts.
2. **Multimodal VLM Training:** Adapting the MedGemma architecture with **MedSigLIP** vision encoders to analyze 12-lead ECG images and output rigorous clinical reports.

## Architecture

- **Base Model:** `google/medgemma-27b-it`
- **Vision Encoder:** `google/medsiglip-448`
- **Quantization:** 4-bit NormalFloat (NF4) via bitsandbytes
- **Fine-Tuning Strategy:** QLoRA
- **Compute:** Modal.com (`A100-80GB` GPUs)

## Repository Structure

- `data_prep_indian_cardio.py`: Generates the Hugging Face dataset containing cardiology instruction pairs tailored for the Indian demographic.
- `finetune_cardio_sahayak.py` / `modal_train_cardio_sahayak.py`: Scripts for fine-tuning the text-only MedGemma backbone.
- `modal_train_vlm_cardio_sahayak.py`: Multimodal instruction tuning script leveraging `PULSE-ECG/ECGBench` for ECG image understanding.
- `vlm_ecg_inference.py`: Inference script for generating clinical reports from ECG images using the fine-tuned VLM.
- `medsiglip_ecg_classifier.py`: Zero-shot ECG classification using MedSigLIP.

## Datasets & Weights

- **Instruction Dataset:** `tp53/cardio-sahayak-india-instruct-v0`
- **Vision Dataset:** `PULSE-ECG/ECGBench` (ptb-test-report subset)
- **Model Weights (Public - CC-BY-4.0 with Attribution):**
  - Text Adapters: `tp53/cardio-sahayak`
  - VLM Adapters: `tp53/cardio-sahayak-vlm`

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