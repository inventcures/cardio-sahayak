"""
ECG image augmentation to simulate real-world scanning/photography conditions.
Applies: Gaussian noise, baseline wander, gain variation, rotation/skew.
"""
import random
from pathlib import Path
from dataclasses import dataclass

try:
    from PIL import Image, ImageFilter, ImageEnhance
    import numpy as np
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


@dataclass
class AugmentationConfig:
    noise_std: float = 15.0
    brightness_range: tuple[float, float] = (0.7, 1.3)
    contrast_range: tuple[float, float] = (0.7, 1.3)
    rotation_range: tuple[float, float] = (-3.0, 3.0)
    blur_probability: float = 0.3
    blur_radius: float = 1.0


DEFAULT_CONFIG = AugmentationConfig()


def add_gaussian_noise(image: "Image.Image", std: float = 15.0) -> "Image.Image":
    if not HAS_PIL:
        return image
    arr = np.array(image, dtype=np.float32)
    noise = np.random.normal(0, std, arr.shape)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def adjust_brightness_contrast(
    image: "Image.Image",
    brightness_range: tuple[float, float],
    contrast_range: tuple[float, float],
) -> "Image.Image":
    if not HAS_PIL:
        return image
    brightness = random.uniform(*brightness_range)
    contrast = random.uniform(*contrast_range)
    image = ImageEnhance.Brightness(image).enhance(brightness)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    return image


def apply_rotation(image: "Image.Image", angle_range: tuple[float, float]) -> "Image.Image":
    if not HAS_PIL:
        return image
    angle = random.uniform(*angle_range)
    return image.rotate(angle, expand=False, fillcolor=(255, 255, 255))


def apply_blur(image: "Image.Image", radius: float = 1.0) -> "Image.Image":
    if not HAS_PIL:
        return image
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def augment_ecg_image(
    image: "Image.Image",
    config: AugmentationConfig = DEFAULT_CONFIG,
) -> "Image.Image":
    if not HAS_PIL:
        return image

    image = add_gaussian_noise(image, config.noise_std)
    image = adjust_brightness_contrast(
        image, config.brightness_range, config.contrast_range
    )
    image = apply_rotation(image, config.rotation_range)
    if random.random() < config.blur_probability:
        image = apply_blur(image, config.blur_radius)

    return image


def augment_ecg_dataset(
    input_dir: Path,
    output_dir: Path,
    multiplier: int = 5,
    config: AugmentationConfig = DEFAULT_CONFIG,
) -> list[Path]:
    if not HAS_PIL:
        print("PIL not installed. Skipping augmentation.")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []

    image_files = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    )

    for img_path in image_files:
        image = Image.open(img_path).convert("RGB")
        for i in range(multiplier):
            augmented = augment_ecg_image(image, config)
            out_name = f"{img_path.stem}_aug{i}{img_path.suffix}"
            out_path = output_dir / out_name
            augmented.save(str(out_path))
            output_paths.append(out_path)

    print(f"Generated {len(output_paths)} augmented images from {len(image_files)} originals")
    return output_paths
