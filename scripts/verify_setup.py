"""Minimal verification for the MiniMind medical QA release.

Checks:
1. tokenizer loads (vocab 6400);
2. a 26M MiniMind model can be instantiated and forward-passed;
3. the official full_sft_512.pth loads (if present);
4. the included medical LoRA adapter loads and forward-passes (if present).

Run from the repository root:
    python scripts/verify_setup.py --weight_path out/full_sft_512.pth
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.model_lora import apply_lora, load_lora  # noqa: E402
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the minimal MiniMind medical QA setup")
    parser.add_argument("--weight_path", type=str, default=str(ROOT / "out/full_sft_512.pth"))
    parser.add_argument(
        "--lora_path",
        type=str,
        default=str(ROOT / "evidence/lora_medical_official_512.pth"),
    )
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / "model"))
    print(f"[ok] tokenizer loaded: vocab_size={len(tokenizer)}")

    config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
    )
    model = MiniMindForCausalLM(config).eval().to(args.device)
    num_parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"[ok] model instantiated: {num_parameters / 1e6:.2f}M parameters")

    input_ids = torch.randint(0, config.vocab_size, (1, 8), device=args.device)
    with torch.no_grad():
        output = model(input_ids)
    print(f"[ok] forward pass: logits={tuple(output.logits.shape)}")

    weight_path = Path(args.weight_path)
    if weight_path.exists():
        state_dict = torch.load(weight_path, map_location=args.device)
        model.load_state_dict(state_dict, strict=True)
        print(f"[ok] loaded base checkpoint: {weight_path.name}")
        with torch.no_grad():
            output = model(input_ids)
        print(f"[ok] forward with base checkpoint: logits={tuple(output.logits.shape)}")
    else:
        print(f"[skip] base checkpoint not found: {weight_path}")
        print("       download it with:")
        print(
            "       curl -L -o out/full_sft_512.pth "
            "https://www.modelscope.cn/models/gongjy/MiniMind2-PyTorch/resolve/master/full_sft_512.pth"
        )

    lora_path = Path(args.lora_path)
    if lora_path.exists():
        apply_lora(model)
        load_lora(model, str(lora_path))
        lora_parameters = sum(
            parameter.numel()
            for module in model.modules()
            if hasattr(module, "lora")
            for parameter in module.lora.parameters()
        )
        print(f"[ok] loaded LoRA adapter: {lora_path.name}")
        print(f"[ok] LoRA parameters: {lora_parameters / 1e6:.3f}M")
        with torch.no_grad():
            output = model(input_ids)
        print(f"[ok] forward with LoRA: logits={tuple(output.logits.shape)}")
    else:
        print(f"[skip] LoRA adapter not found: {lora_path}")

    print("[done] minimal setup is runnable")


if __name__ == "__main__":
    main()
