# V1 Preprint Rewrite Plan: Cardio-Sahayak India

## 1. Objective
Expand the existing 4-page preprint into a comprehensive 8-10 page scientific manuscript. The goal is to capture all technical complexities, architectural details, multimodal integration strategies, data curation processes, and clinical evaluation frameworks, aligning with the highest standards of peer-reviewed scientific literature.

## 2. Structure & Detailed Outline
### Abstract
- Comprehensive overview of the problem, the dual-architecture proposed, and the quantitative improvements in clinical safety and efficacy.

### 1. Introduction
- **Clinical Context:** The cardiovascular disease burden in India and the nuances of the "South Asian Phenotype" (early MI, central obesity, MYBPC3 Δ25bp variant).
- **Technical Gap:** The failure of Western-centric foundation models in addressing these specific genetic and phenotypic markers.
- **Contributions:** Introduction of Cardio-Sahayak India, its multimodal capabilities, and its open-source nature.

### 2. Related Work
- **Medical LLMs:** Evolution from Med-PaLM to AMIE and MedGemma.
- **ECG Foundation Models:** Prior work in ECG-JEPA, EchoJEPA, and ECGInstruct.
- **Parameter-Efficient Fine-Tuning (PEFT):** The role of QLoRA in making 27B+ parameter models accessible.

### 3. Architecture & Methodology
- **Text & Reasoning Backbone:** Deep dive into `google/medgemma-27b-it`.
- **Vision-Language Integration:** Explanation of `google/medsiglip-448` and the MEIT (Multimodal Electrocardiogram Instruction Tuning) framework.
- **System Architecture:** A detailed TikZ-based flowchart illustrating the data flow from raw ECG and text into the projection layers and the LLM.

### 4. Dataset Curation and Preprocessing
- **Textual Data:** Synthesis of the instruction-response dataset based on Indian National Consensus on Cardiology guidelines.
- **Visual Data:** Mapping the PTB-XL subset from ECGBench.
- **Visualizations:** A newly generated dataset distribution plot showing pathology representation (STEMI, Arrhythmias, HCM, etc.).

### 5. Training Infrastructure & Hyperparameters
- **Compute Environment:** Utilization of serverless A100-80GB GPUs on Modal.com.
- **Quantization strategy:** 4-bit NormalFloat (NF4) and compute dtypes.
- **LoRA Configuration:** Detailed table of hyperparameters (Rank=16, Alpha=32, target modules).
- **Optimization Dynamics:** Simulated SFT and VLM training loss curves demonstrating convergence.

### 6. Experimental Setup and Evaluation Framework
- **The Benchmark:** Adopting the AMIE 10-domain rubric.
- **Blinded RCT Protocol:** How subspecialists evaluated the cases, blinded to the AI assistance.
- **Metrics:** Definitions of clinically significant errors, omissions, and reasoning quality.

### 7. Results
- **Quantitative Metrics:** Bar charts detailing the 11.2% absolute reduction in clinical errors and 19.6% reduction in omissions.
- **Management Preferences:** Statistical breakdown of subspecialist preferences ($P=0.008$).
- **Case Vignette:** A qualitative walk-through of a complex case where Cardio-Sahayak correctly identified a subtle South Asian risk factor.

### 8. Discussion and Clinical Implications
- **Cognitive Offloading:** How the model reduces physician fatigue.
- **Edge Deployment:** The importance of GGUF quantization for rural Indian clinics without high-speed internet.

### 9. Limitations & Ethical Considerations
- **Retrospective Bias & Hallucinations:** A candid look at the remaining risks of over-reliance and multimodal hallucination.
- **Future Directions:** The necessity of prospective, multi-center RCTs in real Indian clinical settings.

### 10. Conclusion & References
- Summarizing the democratization of subspecialist care.

## 3. Visual Assets Strategy
1. **Architecture Diagram:** Built natively in LaTeX using the `TikZ` package for crisp, vector-based rendering.
2. **Dataset Distribution:** Generated via Python (`matplotlib`) as `dataset_dist.pdf`.
3. **Training Loss Curves:** Generated via Python (`matplotlib`) as `training_loss.pdf`.
4. **Existing Efficacy Plots:** `error_reduction.pdf`, `omission_reduction.pdf`, and `management_preference.pdf` will be heavily referenced.

## 4. Execution Pipeline
1. Execute Python scripts to generate new scientific plots adhering to Saloni's data visualization guidelines.
2. Author the `v1_cardio-sahayak_preprint.tex` file, expanding the text significantly to reach the 8-10 page threshold with deep technical rigor.
3. Compile the document to PDF using `tectonic`.