from dataclasses import dataclass
from typing import Any


@dataclass
class MultimodalCollator:
    """Collates image-text pairs for VLM training (ECG/Echo experts)."""
    processor: Any
    max_length: int = 2048

    def __call__(self, examples: list[dict]) -> dict:
        texts = []
        images = []

        for ex in examples:
            instruction = ex.get("instruction", "")
            output = ex.get("output", "")

            messages = [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": output},
            ]
            text = self.processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)

            img = ex.get("image")
            images.append(img)

        has_images = any(img is not None for img in images)

        if has_images:
            batch = self.processor(
                text=texts,
                images=[img for img in images if img is not None],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
        else:
            batch = self.processor.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )

        batch["labels"] = batch["input_ids"].clone()
        return batch


@dataclass
class TextOnlyCollator:
    """Collates text-only instruction/output pairs (clinical expert)."""
    tokenizer: Any
    max_length: int = 4096

    def __call__(self, examples: list[dict]) -> dict:
        texts = []
        for ex in examples:
            instruction = ex.get("instruction", "")
            output = ex.get("output", "")
            messages = [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": output},
            ]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)

        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        batch["labels"] = batch["input_ids"].clone()
        return batch


@dataclass
class GRPOCollator:
    """Collates MCQ prompts for GRPO training. Returns prompt + correct answer."""
    tokenizer: Any
    max_prompt_length: int = 2048

    def __call__(self, examples: list[dict]) -> dict:
        prompts = []
        answers = []

        for ex in examples:
            question = ex.get("question", "")
            options = ex.get("options", [])
            correct = ex.get("correct_answer", "")

            formatted_options = "\n".join(
                f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)
            )
            prompt = f"{question}\n\n{formatted_options}\n\nAnswer:"
            prompts.append(prompt)
            answers.append(correct)

        batch = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_prompt_length,
        )
        batch["answers"] = answers
        return batch
