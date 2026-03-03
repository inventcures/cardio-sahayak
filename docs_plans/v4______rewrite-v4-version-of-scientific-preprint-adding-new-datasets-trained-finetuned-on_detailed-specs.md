# Detailed Specifications: v4 Preprint Rewrite

## 1. Objective
To write the v4 version of the scientific preprint for **Cardio-Sahayak India** (`out/v4_cardio-sahayak_preprint.tex`), using the existing v2 preprint as a foundation and incorporating the new Phase 2 multi-dataset fine-tuning process. This includes the integration of Indian clinical notes (EkaCare), synthetic phenotype shifting (Gemini 2.5 Flash), and multimodal data from Indian portals (IIIT-Hyderabad ECG, ScienceOpen) and global sources (MIMIC-IV, MEETI). The paper will follow Saloni's data visualization guidelines to present architectural choices, training dynamics, and evaluation results cleanly and effectively.

## 2. Key Additions from v2 to v4
*   **Expanded Dataset Acquisition:** Detailing the extraction of 156 real-world Indian clinical notes from the `ekacare/clinical_note_generation_dataset`.
*   **Synthetic Phenotype Shifting:** Explaining the LLM-driven adaptation of Western vignettes into the South Asian demographic using Gemini 2.5 Flash, including BMI adjustments, early-onset MI age shifting, and genetic variant injection (MYBPC3 Δ25bp).
*   **Multimodal Integration Strategy:** Explaining the ingestion pipeline for scanned Indian ECG PDFs (IIIT-H), ScienceOpen datasets, and large-scale waveform mapping using MIMIC-IV and MEETI, culminating in a 166-record highly-curated V2 Dataset.
*   **Phase 2 Fine-Tuning Dynamics:** Covering the resuming of the `tp53/cardio-sahayak` LoRA adapters using a refined learning rate ($1 	imes 10^{-4}$), culminating in the release of `v2_weights`.
*   **GGUF Quantization Updates:** Detailing the architecture conversion from `Gemma3ForConditionalGeneration` to `Gemma2ForCausalLM` to support edge-deployment via `llama.cpp` using the Q4_K_M quantization, bypassing the BPE pre-tokenizer mismatch.

## 3. Visual Scientific Communication Plan (Following Saloni's Guidelines)
We will adhere to Saloni Dattani's guidelines:
*   Horizontal text and direct labeling.
*   No chart junk; plain language annotations.
*   Clear color mapping to concepts.

**Visualizations to Generate / Include:**
1.  `v4_dataset_dist.pdf` (replacing older dataset dists): A bar chart showing the composition of the v2 training dataset (Eka Notes, Synthetic Shifts, Multimodal Mocks) with direct labels and clean horizontal text.
2.  `v4_training_loss.pdf`: A line plot demonstrating the stable convergence of the Phase 2 fine-tuning on the new complex records, comparing it with Phase 1.
3.  `v4_phenotype_shift.pdf`: A visual representation of how variables like Age, BMI, and Genetics are mathematically "shifted" during the Gemini preprocessing step.
*Note: Python scripts for generating these v4 visual assets will be created alongside the LaTeX document.*

## 4. Document Structure Updates (LaTeX)
Using the v2 preprint as a base, we will make targeted insertions and modifications:
*   **Abstract:** Update to mention the novel two-phase fine-tuning methodology, the 166-record highly-curated multimodal V2 dataset, and the specific architectural patches for edge-ready GGUF deployment.
*   **Methodology & Data Curation:** Expand significantly to include a new subsection dedicated to Phase 2: the `ekacare` dataset, the Gemini phenotype shifter, and the MIMIC-IV/MEETI integrations.
*   **Training Infrastructure:** Add a subsection detailing the "Phase 2 Training Dynamics" (resuming adapters, adjusted learning rates) and update the "GGUF Quantization" section to detail the `Gemma3` architectural patch.
*   **Results & Evaluation:** Retain the baseline AMIE benchmarks but contextualize them with the expected improvements brought by the culturally contextualized V2 data.
*   **Conclusion:** Update to reflect the completion of Phase 2 and the release of the V2 weights and patched GGUF models.

## 5. Execution Steps
1.  Write Python script `generate_v4_plots.py` to create the necessary plots (matplotlib) adhering to Saloni's principles, generating `v4_dataset_dist.pdf`, `v4_training_loss.pdf`, and `v4_phenotype_shift.pdf`.
2.  Read the `out/v2_cardio-sahayak_preprint.tex` file.
3.  Draft the `out/v4_cardio-sahayak_preprint.tex` file by integrating the new content and V4 figures into the V2 base.
4.  Compile with `tectonic` and verify the output PDF.
