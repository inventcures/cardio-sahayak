# Detailed Specifications: v3 Preprint Rewrite

## 1. Objective
To write the v3 version of the scientific preprint for **Cardio-Sahayak India** (`out/v3_cardio-sahayak_preprint.tex`), incorporating the Phase 2 multi-dataset fine-tuning process. This includes the integration of Indian clinical notes (Eka), synthetic phenotype shifting (Gemini 2.5 Flash), and multimodal data from Indian portals (IIIT-Hyderabad ECG, ScienceOpen) and global sources (MIMIC-IV, MEETI). The paper will follow Saloni's data visualization guidelines to present architectural choices, training dynamics, and evaluation results cleanly and effectively.

## 2. Key Additions from v2 to v3
*   **Expanded Dataset Acquisition:** Detailing the extraction of 156 real-world Indian clinical notes from the `ekacare/clinical_note_generation_dataset`.
*   **Synthetic Phenotype Shifting:** Explaining the LLM-driven adaptation of Western vignettes into the South Asian demographic using Gemini, including BMI adjustments, early-onset MI age shifting, and genetic variant injection (MYBPC3 Δ25bp).
*   **Multimodal Integration Strategy:** Explaining the ingestion pipeline for scanned Indian ECG PDFs (IIIT-H), ScienceOpen datasets, and large-scale waveform mapping using MIMIC-IV and MEETI.
*   **Phase 2 Fine-Tuning Dynamics:** Covering the resuming of the `tp53/cardio-sahayak` LoRA adapters using a refined learning rate ($1 	imes 10^{-4}$), culminating in the release of `v2_weights`.
*   **GGUF Quantization Updates:** Detailing the architecture conversion from `Gemma3ForConditionalGeneration` to `Gemma2ForCausalLM` to support edge-deployment via `llama.cpp` using the Q4_K_M quantization.

## 3. Visual Scientific Communication Plan (Following Saloni's Guidelines)
We will adhere to Saloni Dattani's guidelines:
*   Horizontal text and direct labeling.
*   No chart junk; plain language annotations.
*   Clear color mapping to concepts.

**Visualizations to Generate:**
1.  `v3_dataset_dist.pdf`: A bar chart showing the composition of the v2 training dataset (Eka Notes, Synthetic Shifts, Multimodal Mocks) with direct labels and clean horizontal text.
2.  `v3_training_loss.pdf`: A line plot demonstrating the stable convergence of the Phase 2 fine-tuning on the new complex records, comparing it with Phase 1.
3.  `v3_architecture.pdf` (or TikZ diagram): An updated dual-architecture flow specifically highlighting the new data pipelines and adapter resuming process.
4.  `v3_phenotype_shift.pdf`: A visual representation of how variables like Age, BMI, and Genetics are mathematically "shifted" during the Gemini preprocessing step.

## 4. Document Structure (LaTeX)
*   **Abstract:** Updated to mention the 166-record highly-curated multimodal V2 dataset and the edge-ready GGUF deployment.
*   **Introduction:** Emphasizing the data drought for South Asian cardiology and our multi-pronged approach to solving it.
*   **Methodology & Data Curation:** A new section dedicated to the `ekacare` dataset, the Gemini phenotype shifter, and the MIMIC-IV/MEETI integrations.
*   **Training & GGUF Quantization:** Detailing the Phase 2 training and the specific architectural patch for `llama.cpp` compatibility.
*   **Results & Evaluation:** Comparing baseline AMIE benchmarks against the expected improvements brought by the culturally contextualized data.
*   **Conclusion & Open Source Impact.**

## 5. Execution Steps
1.  Generate Python scripts to create the necessary plots (matplotlib/seaborn) adhering to Saloni's principles.
2.  Draft the `v3_cardio-sahayak_preprint.tex` file incorporating the new content and figures.
3.  Compile with `tectonic` and verify the output.
