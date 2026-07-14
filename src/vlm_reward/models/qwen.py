"""
Shared loading helpers for Qwen vision-language fine-tuning.
"""
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import torch

from .vision import freeze_vision_encoder


LoraTargets = Union[str, Sequence[str]]


def resolve_qlora(qlora: Optional[bool]) -> bool:
    """
    Use the explicit QLoRA choice, or enable it automatically on CUDA.
    """
    return torch.cuda.is_available() if qlora is None else qlora


def load_qwen_image_text_model(
    model_id: str,
    use_qlora: bool,
    bf16: bool,
) -> Any:
    """
    Load Qwen with the quantization settings used by the completed runs.
    """
    from transformers import AutoModelForImageTextToText, BitsAndBytesConfig

    dtype = torch.bfloat16 if bf16 else torch.float16
    model_kwargs = {"attn_implementation": "sdpa"}
    if torch.cuda.is_available():
        model_kwargs["dtype"] = dtype

    if use_qlora:
        if not torch.cuda.is_available():
            raise ValueError("QLoRA requires CUDA. Use --no-qlora for normal loading.")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        model_kwargs["device_map"] = "auto"

    return AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)


def add_lora_adapter(
    model: Any,
    *,
    use_qlora: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: LoraTargets,
    adapter_checkpoint: Optional[str] = None,
) -> Any:
    """
    Attach a new LoRA adapter or resume a trainable existing adapter.
    """
    from peft import (
        LoraConfig,
        PeftModel,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )

    if use_qlora:
        model = prepare_model_for_kbit_training(model)

    if adapter_checkpoint is not None:
        checkpoint_path = Path(adapter_checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)
        print(f"Loading existing LoRA adapter from: {checkpoint_path}")
        return PeftModel.from_pretrained(model, checkpoint_path, is_trainable=True)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
    return get_peft_model(model, lora_config)


def freeze_qwen_vision_encoder(model: Any) -> int:
    """
    Compatibility-named wrapper around shared vision-module discovery.
    """
    return freeze_vision_encoder(model)


def load_merged_lora_for_inference(
    model_id: str,
    adapter_dir: str,
    bf16: bool,
) -> Any:
    """
    Load, merge, and freeze a LoRA model for autoregressive evaluation.
    """
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText

    adapter_path = Path(adapter_dir)
    if not adapter_path.exists():
        raise FileNotFoundError(adapter_path)
    dtype = torch.bfloat16 if bf16 else torch.float16
    model_kwargs = {"device_map": "auto"}
    if torch.cuda.is_available():
        model_kwargs["dtype"] = dtype
    base_model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()
    model.eval()
    return model
