# Detailed Specifications: v2 - Expanded Dataset Acquisition & Integration for Cardio-Sahayak India

## 1. Objective
To significantly improve the generalizability, robustness, and clinical safety of **Cardio-Sahayak India**, we must expand the volume and diversity of our fine-tuning datasets. The primary focus remains on **Indian and South Asian populations** (capturing unique metabolic profiles, early-onset risks, and genetic variants like MYBPC3 Δ25bp). We need to aggregate multimodal data encompassing raw ECG waveforms, 12-lead images, unstructured clinical notes, and structured diagnostic reports. Where South Asian data is insufficient, we will fallback to high-quality global datasets (e.g., MIMIC-IV) and contextually adapt them.

## 2. Target Datasets: Primary (South Asian Focus)

### 2.1 Eka Structured Clinical Note Generation Dataset
- **Source:** Hugging Face (`ekacare/clinical_note_generation_dataset`)
- **Modality:** Text (Clinical Notes)
- **Relevance:** Sourced directly from the Indian healthcare ecosystem. It provides the necessary vernacular, phrasing, and structural nuances of Indian clinical documentation.
- **Action:** Direct download via `datasets` library.

### 2.2 ECG Dataset 1.0.2 (IIIT-Hyderabad)
- **Source:** India Data Portal (IHub-Data)
- **Modality:** ECG Images/PDFs + Labels (Normal, AFib, MI)
- **Relevance:** Directly addresses the digitization of scanned ECGs common in Indian clinical settings.
- **Action:** Automated scraping/downloading and PDF-to-image conversion pipeline.

### 2.3 ECG Images Dataset of Cardiac and COVID-19 Patients
- **Source:** ScienceOpen / Data in Brief (South Asian/Pakistan Institutes)
- **Modality:** 12-lead ECG Images + Clinical Reviews
- **Relevance:** Contains 1,937 distinct records from South Asian patients, covering MI, abnormal heartbeats, and normal profiles.
- **Action:** Scripted download and alignment of images to their corresponding clinical reviews.

### 2.4 SAGE (South Asian Genomes and Exomes)
- **Source:** CSIR-IGIB (clingen.igib.res.in/sage/)
- **Modality:** Genomic variant frequencies.
- **Relevance:** Crucial for grounding the LLM's reasoning regarding familial risk and specific variants (e.g., MYBPC3 Δ25bp) within the Indian population.
- **Action:** API querying / bulk download of summary statistics to generate synthetic clinical vignettes.

## 3. Target Datasets: Fallback & Global Baseline (For Generalization)

To ensure the foundation model has a sufficiently broad baseline understanding of cardiac pathology, we must integrate massive, high-quality global datasets.

### 3.1 MIMIC-IV-ECG & MIMIC-IV-Note
- **Source:** PhysioNet
- **Modality:** 800,000+ 12-lead ECG waveforms + De-identified free-text clinical notes.
- **Relevance:** The gold standard for linking physiological signals to clinical reasoning.
- **Action:** Requires PhysioNet credentialed access. We need a script (`download_mimic_ecg.py`) using `wget` or the PhysioNet API, followed by waveform-to-image plotting scripts to match the MedSigLIP input format.

### 3.2 MEETI (MIMIC-IV-Ext ECG-Text-Image)
- **Source:** GitHub (PKUDigitalHealth)
- **Modality:** Synchronized ECG waveforms, high-res images, and detailed textual interpretations.
- **Relevance:** Already bridges the gap between raw data and LLM-ready text.
- **Action:** Git clone and data extraction pipeline.

### 3.3 Kaggle / Mendeley ECG & CHD Datasets
- **Source:** Kaggle (ECG-Clinical Dataset for CHD), Mendeley (Heart Condition Classification via IoT).
- **Action:** Use `kaggle` CLI tool for automated ingestion.

## 4. Data Processing & Integration Strategy

### 4.1 Harmonization Pipeline
All disparate data sources must be funneled into a unified Hugging Face `Dataset` structure to be compatible with our existing `modal_train_vlm_cardio_sahayak.py` and SFT frameworks.
- **Text:** Synthesize instruction-response pairs (e.g., `<instruction> Analyze this patient's clinical note... <response> Based on the Indian guidelines...`).
- **Vision:** Standardize all ECGs to 896x896 (MedSigLIP's expected resolution), normalizing grid lines and lead layouts.

### 4.2 Synthetic Contextualization (The "Phenotype Shift")
For datasets like MIMIC-IV (which are Western-centric), we will use an LLM (e.g., Gemini Flash) in a preprocessing step to intelligently "shift" the clinical vignettes:
- Lowering the age of MI onset by 5-10 years.
- Adjusting BMI descriptors (e.g., labeling a BMI of 24 as a potential risk for central adiposity in a South Asian context).
- Injecting family history of specific genetic variants.

## 5. Implementation Roadmap (New Scripts Needed)

1. `scrape_indian_ecg_portals.py`: Selenium/BeautifulSoup script to navigate and extract PDFs/Images from IIIT-Hyderabad and ScienceOpen.
2. `ingest_eka_notes.py`: Simple script to download and reformat the `ekacare` dataset into the Cardio-Sahayak instruction format.
3. `mimic_waveform_to_image.py`: Utilizes `wfdb` (Waveform Database Software Package) to read MIMIC-IV `.dat` files and plot standard 12-lead ECG images.
4. `synthetic_phenotype_shifter.py`: Prompt-based script to rewrite Western clinical notes into South Asian specific vignettes.
5. `compile_v2_dataset.py`: The final aggregator that merges the scraped, downloaded, and synthesized data into `tp53/cardio-sahayak-india-instruct-v2`.

## 6. Infrastructure
- All heavy scraping and processing will be containerized on **Modal.com** to leverage fast network I/O and parallel processing for the 800k+ MIMIC records.
