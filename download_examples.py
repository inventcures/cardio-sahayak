import urllib.request
import os

os.makedirs('examples', exist_ok=True)

images = {
    'normal_ecg.jpg': 'https://upload.wikimedia.org/wikipedia/commons/9/9e/12_lead_ECG_of_26_year_old_male.jpg',
    'stemi_ecg.jpg': 'https://upload.wikimedia.org/wikipedia/commons/e/e4/12_Lead_ECG_showing_acute_inferior_myocardial_infarction.jpg',
    'afib_ecg.png': 'https://upload.wikimedia.org/wikipedia/commons/4/41/Atrial_fibrillation_%281%29.png'
}

for name, url in images.items():
    try:
        print(f"Downloading {name}...")
        urllib.request.urlretrieve(url, f"examples/{name}")
    except Exception as e:
        print(f"Failed to download {name}: {e}")
