import os
import model_loader
import pipeline
from PIL import Image
from pathlib import Path
from transformers import CLIPTokenizer
import torch
import logging

from network_rrdbnet import RRDBNet as net  # Importa la rete per BSRGAN
import utils_image as util  # Funzioni utili per la gestione delle immagini

DEVICE = "cpu"

ALLOW_CUDA = False
ALLOW_MPS = True

if torch.cuda.is_available() and ALLOW_CUDA:
    DEVICE = "cuda"
elif (torch.has_mps or torch.backends.mps.is_available()) and ALLOW_MPS:
    DEVICE = "mps"
print(f"Using device: {DEVICE}")

tokenizer = CLIPTokenizer("../data/tokenizer_vocab.json", merges_file="../data/tokenizer_merges.txt")
model_file = "../data/v1-5-pruned-emaonly.ckpt"
models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)

# Carica il modello BSRGAN
sf = 2  # Fattore di super-risoluzione (BSRGAN di solito usa x4)
bsrgan_model_path = "../data/BSRGANx2.pth"  # Assicurati che il modello sia in questa directory

# Definisci la rete BSRGAN
bsrgan_model = net(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=sf)
bsrgan_model.load_state_dict(torch.load(bsrgan_model_path, map_location=DEVICE), strict=True)
bsrgan_model.eval()
for k, v in bsrgan_model.named_parameters():
    v.requires_grad = False
bsrgan_model = bsrgan_model.to(DEVICE)

# TEXT TO IMAGE
prompt = "A graceful red fox with vivid orange fur, a fluffy white-tipped tail, and sharp, attentive eyes standing on a soft mossy forest floor. Surrounding the fox are tall, lush green trees with sunlight filtering through the leaves, casting dappled light onto the ground. In the background, a gentle stream flows through the scene, with a serene natural ambiance of wildflowers and ferns. The setting exudes a peaceful, vibrant, and untouched wilderness"
uncond_prompt = ""  # Prompt negativo
do_cfg = True
cfg_scale = 8  # Scala di configurazione
# IMAGE TO IMAGE
input_image = None
image_path = "immagini/foxy.jpg"
strength = 0.9

# SAMPLER
sampler = "ddpm"
num_inference_steps = 50
seed = 42

output_image = pipeline.generate(
    prompt=prompt,
    uncond_prompt=uncond_prompt,
    input_image=input_image,
    strength=strength,
    do_cfg=do_cfg,
    cfg_scale=cfg_scale,
    sampler_name=sampler,
    n_inference_steps=num_inference_steps,
    seed=seed,
    models=models,
    device=DEVICE,
    idle_device="cpu",
    tokenizer=tokenizer,
)
import numpy as np
# Converti l'output generato in un'immagine PIL
img = Image.fromarray(output_image).convert('RGB')

# Prepara l'immagine per il modello BSRGAN
img_L = util.uint2tensor4(np.array(img, dtype=np.uint8)).to(DEVICE)

# Applica BSRGAN per la super-risoluzione
with torch.no_grad():
    img_E = bsrgan_model(img_L)
img_E = util.tensor2uint(img_E)

# Salva l'immagine finale con BSRGAN applicato
util.imsave(img_E, "foxy_out_carlotta.png")


