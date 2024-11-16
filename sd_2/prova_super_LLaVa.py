import os
import pipeline
import torch
import model_loader
from PIL import Image
import logging
import numpy as np
from network_rrdbnet import RRDBNet as net  # Importa la rete per BSRGAN
import utils_image as util  # Funzioni utili per la gestione delle immagini
import random
import matplotlib.pyplot as plt
from transformers import AutoProcessor, AutoModelForImageTextToText
from encoder import VAE_Encoder
from decoder import VAE_Decoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ALLOW_CUDA = False
ALLOW_MPS = True

# Carica il modello e il processor di LLaVA
model_name = "llava-hf/llava-v1.6-mistral-7b-hf"
processor = AutoProcessor.from_pretrained(model_name)
llava_model = AutoModelForImageTextToText.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

def get_views(image, window_size=128, stride=64):
    height, width = image.shape[:2]
    num_blocks_height = (height - window_size) // stride + 1
    num_blocks_width = (width - window_size) // stride + 1
    views = []

    for h in range(num_blocks_height):
        for w in range(num_blocks_width):
            h_start = h * stride
            w_start = w * stride
            patch = image[h_start:h_start + window_size, w_start:w_start + window_size]
            views.append(patch)
    return views

def assemble_patches(patches, image_shape, window_size=128, stride=64):
    height, width, channels = image_shape
    reassembled_image = np.zeros((height, width, channels), dtype=np.float32)
    weight_matrix = np.zeros((height, width, channels), dtype=np.float32)

    num_blocks_height = (height - window_size) // stride + 1
    num_blocks_width = (width - window_size) // stride + 1

    for i, patch in enumerate(patches):
        h_start = (i // num_blocks_width) * stride
        h_end = h_start + window_size
        w_start = (i % num_blocks_width) * stride
        w_end = w_start + window_size

        reassembled_image[h_start:h_end, w_start:w_end] += patch
        weight_matrix[h_start:h_end, w_start:w_end] += 1

    weight_matrix = np.maximum(weight_matrix, 1)
    reassembled_image /= weight_matrix

    return reassembled_image.astype(np.uint8)

def generate_caption(image_patch):
    inputs = processor(images=Image.fromarray(image_patch), text="Describe the image", return_tensors="pt").to(DEVICE)
    outputs = llava_model.generate(**inputs)
    caption = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
    return caption

if torch.cuda.is_available() and ALLOW_CUDA:
    DEVICE = "cuda"
elif (torch.has_mps or torch.backends.mps.is_available()) and ALLOW_MPS:
    DEVICE = "mps"
print(f"Using device: {DEVICE}")

# Carica il modello BSRGAN
bsrgan_model_path = "BSRGANx2.pth"
sf = 2
bsrgan_model = net(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=sf)
bsrgan_model.load_state_dict(torch.load(bsrgan_model_path, map_location=DEVICE), strict=True)
bsrgan_model.eval().to(DEVICE)

# Carica l'immagine di input
input_image_path = "siura.png"
img = Image.open(input_image_path).convert("RGB")
img_np = np.array(img, dtype=np.uint8)
img_L = util.uint2tensor4(img_np).to(DEVICE)

# Applica BSRGAN
with torch.no_grad():
    img_E = bsrgan_model(img_L)
img_E = util.tensor2uint(img_E)

# Suddividi in patch
patches = get_views(img_E, window_size=512)

# Genera descrizioni per ogni patch
assembly = []
prompt = "image with a better quality of "

for idx, patch in enumerate(patches):
    print(f"Patch {idx + 1} dimensioni: {patch.shape}")
    caption = generate_caption(patch)
    print(f"Caption for Patch {idx + 1}: {caption}")

    # Simula la pipeline di miglioramento immagine
    img_pil = Image.fromarray(patch)
    sampler = "ddpm"
    num_inference_steps = 50
    seed = 42

    output_image = pipeline.generate(
        prompt=prompt + caption,
        uncond_prompt="",
        input_image=img_pil,
        strength=0.25,
        do_cfg=True,
        cfg_scale=14,
        sampler_name=sampler,
        n_inference_steps=num_inference_steps,
        seed=seed,
        models=model_loader.preload_models_from_standard_weights("v1-5-pruned-emaonly.ckpt", DEVICE),
        device=DEVICE,
        idle_device="cpu",
        tokenizer=processor.tokenizer,
    )

    img = Image.fromarray(output_image)
    assembly.append(img)

# Ricomponi l'immagine
image_shape = img_E.shape
output_image = assemble_patches(assembly, image_shape, window_size=512, stride=256)

# Salva l'immagine finale
output_reassembled_path = "immagine_ricomposta.png"
Image.fromarray(output_image).save(output_reassembled_path)

plt.imshow(output_image)
plt.title("Immagine Ricomposta")
plt.axis("off")
plt.show()
