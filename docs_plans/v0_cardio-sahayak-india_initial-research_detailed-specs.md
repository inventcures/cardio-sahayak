# Detailed Specifications: Cardio-Sahayak India (v0)

## 1. Project Overview
**Cardio-Sahayak India** is a specialized Large Language Model (LLM) designed for cardiology, specifically optimized for the Indian population. It accounts for unique genetic predispositions, clinical profiles, and socio-economic contexts prevalent in South Asia.

## 2. Research Foundation
Based on recent advancements:
- **AMIE (Articulate Medical Intelligence Explorer):** Demonstrates that LLMs (Gemini 2.0 Flash) can outperform generalists in complex cardiology diagnosis (e.g., HCM).
- **EchoJEPA & ECG-JEPA:** Establish that latent predictive architectures are superior for signal-based (ECG/Echo) foundation models.
- **South Asian Specificity:** Research indicates specific risk factors (lower BMI thresholds for MI, MYBPC3 Δ25bp genetic variants) that general models often overlook.

## 3. Technical Strategy
### 3.1 Model Selection (Dual-Architecture)
To handle both text-based reasoning and raw ECG image interpretation, we will utilize a dual-architecture approach:
- **Text & Reasoning Model:** `google/medgemma-27b-it` (27B Instruction-Tuned). Provides the deep reasoning required for complex cardiology care based on extracted text and patient history.
- **Multimodal ECG-Vision Model:** A Vision-Language Model (VLM) fine-tuned for ECGs (e.g., fine-tuning `llava-med-v1.5` or `Qwen2-VL` on the ECGInstruct dataset or using the MEIT framework). This component will ingest raw 12-lead ECG images/graphs, extract structural features (e.g., ST-elevation, T-wave inversion), and output a structured preliminary text interpretation.

### 3.2 Fine-Tuning Methodology
- **Text Model (MedGemma):** QLoRA (4-bit quantization with LoRA adapters) via Hugging Face TRL (Supervised Fine-Tuning).
- **Vision Model (VLM):** Multimodal Instruction Tuning (e.g., following the MEIT: Multimodal Electrocardiogram Instruction Tuning framework). We will freeze the vision encoder and fine-tune the projection layer and LLM backbone on ECG image-report pairs.
- **Hardware:** Modal.com (GPU: `A100:80GB` or `H100`).

### 3.3 Data Ingestion & Preprocessing
- **Clinical Guidelines:** Indian National Consensus on Cardiology, ICMR guidelines.
- **Genetic Data:** Focus on South Asian specific cardiac mutations.
- **Multimodal ECG Datasets:** 
    - PTB-XL (converted to image representations).
    - ECGInstruct dataset for visual-text alignment.
    - Indian hospital collected ECG image datasets (from GitHub/Mendeley).
- **Formatting:** 
    - *Text:* Convert clinical notes into instruction-response pairs.
    - *Multimodal:* Pair generated 12-lead ECG images with their corresponding rigorous clinical reports (capturing morphological details like PR interval, QRS duration, and diagnostic conclusions).

## 4. Implementation Roadmap
### Phase 1: Data Preparation
- Script: `data_prep_indian_cardio.py`
- Goal: Create a Hugging Face Dataset with ~10k-50k high-quality cardiology instruction pairs.

### Phase 2: Fine-Tuning (The "Sahayak" Adaptation)
- Script: `modal_train_cardio_sahayak.py`
- Platform: Modal.com
- Target HF Repo: `tp53/cardio-sahayak` (Private)
- Uses: `SFTTrainer` from TRL.
- Monitoring: Trackio integration for real-time loss/metric visualization.

### Phase 3: Validation
- Benchmark against:
    - General MedGemma performance on South Asian cases.
    - Clinician-validated rubrics (based on the 10-domain rubric in the AMIE paper).

## 5. Deployment
- Target: Hugging Face Inference Endpoints or local deployment via GGUF (for low-resource clinical settings in India).

## 6. Credentials & Environment
- **HF Token:** YOUR_HF_TOKEN_HERE (HF Pro/Zero GPU access).
- **Project Directory:** `cardio-sahayak/`
