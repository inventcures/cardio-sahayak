import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs('examples', exist_ok=True)

def generate_mock_ecg(filename, title, condition="normal"):
    fig, ax = plt.subplots(figsize=(10, 4))
    
    t = np.linspace(0, 10, 1000)
    
    if condition == "normal":
        y = np.sin(2 * np.pi * 1.0 * t) + 0.1 * np.random.randn(len(t))
    elif condition == "stemi":
        y = np.sin(2 * np.pi * 1.0 * t) + 0.5 * np.sin(2 * np.pi * 0.5 * t) + 0.1 * np.random.randn(len(t))
    elif condition == "afib":
        y = 0.5 * np.sin(2 * np.pi * 3.0 * t) + 0.8 * np.random.randn(len(t))
    elif condition == "st_depression": # CAD/Ischemia
        y = np.sin(2 * np.pi * 1.0 * t) - 0.4 * np.sin(2 * np.pi * 1.0 * t + np.pi/4) + 0.1 * np.random.randn(len(t))
    elif condition == "lvh": # Left Ventricular Hypertrophy
        y = 2.5 * np.sin(2 * np.pi * 1.0 * t) + 0.1 * np.random.randn(len(t))
    elif condition == "dcm": # Dilated Cardiomyopathy (often low voltage / conduction delays)
        y = 0.4 * np.sin(2 * np.pi * 0.8 * t) + 0.1 * np.random.randn(len(t))
    else:
        y = np.sin(2 * np.pi * 1.0 * t) + 0.1 * np.random.randn(len(t))
        
    ax.plot(t, y, color='black', linewidth=1)
    ax.set_title(title)
    ax.grid(True, color='red', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Hide axes
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(f'examples/{filename}', dpi=150, bbox_inches='tight')
    plt.close()

# The Big 6 for South Asian Clinics
generate_mock_ecg('cad_stemi_ecg.jpg', 'Acute STEMI (Early Onset Phenotype)', condition='stemi')
generate_mock_ecg('cad_ischemia_ecg.jpg', 'Ischemic Heart Disease (ST Depression)', condition='st_depression')
generate_mock_ecg('lvh_ecg.jpg', 'Left Ventricular Hypertrophy (Hypertension)', condition='lvh')
generate_mock_ecg('afib_ecg.jpg', 'Atrial Fibrillation', condition='afib')
generate_mock_ecg('normal_variant_ecg.jpg', 'Normal Sinus Rhythm (Early Repolarization Variant)', condition='normal')
generate_mock_ecg('hcm_dcm_ecg.jpg', 'Cardiomyopathy (Low Voltage / Conduction Delay)', condition='dcm')
