"""Shared single-image backend for APBench and external benchmark adapters."""

from __future__ import annotations

import os

from metrics import generate as core


def get_device() -> str:
    return core.get_device()


def load_model(model_name: str, device: str):
    loaders = {
        "sd21": core.load_sd21,
        "sd3": core.load_sd3,
        "flux1": core.load_flux1,
        "pixart-sigma": core.load_pixart_sigma,
        "comat21": core.load_comat21,
        "infi": core.load_infi,
        "showo2": core.load_showo2,
    }
    try:
        return loaders[model_name](device)
    except KeyError as exc:
        raise ValueError(f"Unsupported generator: {model_name}") from exc


def generate_image(model, model_name: str, prompt: str, device: str, seed: int):
    import torch

    if model_name in {"sd21", "sd3", "pixart-sigma"}:
        return model(prompt, generator=core.generator_for_device(device, seed)).images[0]
    if model_name == "flux1":
        return model(
            prompt,
            guidance_scale=0.0,
            num_inference_steps=4,
            max_sequence_length=256,
            generator=core.generator_for_device(device, seed),
        ).images[0]
    if model_name == "comat21":
        return model.predict(
            prompt,
            height=768,
            width=768,
            generator=core.generator_for_device(device, seed),
        ).images[0]
    if model_name == "infi":
        return core.generate_infi_batch(model, [prompt], seed)[0]
    if model_name == "showo2":
        return core.generate_showo2_batch(model, [prompt], seed)[0]
    raise ValueError(f"Unsupported generator: {model_name}")

