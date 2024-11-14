import os
import pipeline
import torch
import model_loader
from PIL import Image
from transformers import CLIPTokenizer
import logging
import numpy as np
from network_rrdbnet import RRDBNet as net  # Importa la rete per BSRGAN
import utils_image as util  # Funzioni utili per la gestione delle immagini
import random
from PIL import Image
import matplotlib.pyplot as plt
from openai import OpenAI
from PIL import Image
import torchvision.transforms as transforms
import model_converter
from encoder import VAE_Encoder
from decoder import VAE_Decoder
DEVICE = "cpu"

ALLOW_CUDA = False
ALLOW_MPS = True



def get_views(image, window_size=128, stride=64, random_jitter=False, vae_scale_factor=1):
    """
    Suddivide un'immagine in patch sovrapposti di dimensione `window_size` e passo `stride`.
    La funzione supporta il jitter casuale per un leggero spostamento dei patch.
    
    Parametri:
    - image: immagine di input come array NumPy (H, W, C)
    - window_size: dimensione di ciascun patch
    - stride: passo tra i patch
    - random_jitter: se True, aggiunge un offset casuale alla posizione dei patch
    - vae_scale_factor: fattore di scala per altezza e larghezza dell'immagine
    
    Ritorna:
    - views: lista di patch come array NumPy
    """
    height, width = image.shape[:2]
    height //= vae_scale_factor
    width //= vae_scale_factor
    num_blocks_height = int((height - window_size) / stride - 1e-6) + 2 if height > window_size else 1
    num_blocks_width = int((width - window_size) / stride - 1e-6) + 2 if width > window_size else 1
    total_num_blocks = int(num_blocks_height * num_blocks_width)
    views = []
    
    for i in range(total_num_blocks):
        h_start = int((i // num_blocks_width) * stride)
        h_end = h_start + window_size
        w_start = int((i % num_blocks_width) * stride)
        w_end = w_start + window_size

        # Correzione dei bordi per evitare out-of-bounds
        if h_end > height:
            h_start = int(h_start + height - h_end)
            h_end = int(height)
        if w_end > width:
            w_start = int(w_start + width - w_end)
            w_end = int(width)
        if h_start < 0:
            h_end = int(h_end - h_start)
            h_start = 0
        if w_start < 0:
            w_end = int(w_end - w_start)
            w_start = 0

        # Applica jitter casuale se abilitato
        if random_jitter:
            jitter_range = (window_size - stride) // 4
            w_jitter = 0
            h_jitter = 0
            if (w_start != 0) and (w_end != width):
                w_jitter = random.randint(-jitter_range, jitter_range)
            elif (w_start == 0) and (w_end != width):
                w_jitter = random.randint(-jitter_range, 0)
            elif (w_start != 0) and (w_end == width):
                w_jitter = random.randint(0, jitter_range)
            if (h_start != 0) and (h_end != height):
                h_jitter = random.randint(-jitter_range, jitter_range)
            elif (h_start == 0) and (h_end != height):
                h_jitter = random.randint(-jitter_range, 0)
            elif (h_start != 0) and (h_end == height):
                h_jitter = random.randint(0, jitter_range)
            h_start += (h_jitter + jitter_range)
            h_end += (h_jitter + jitter_range)
            w_start += (w_jitter + jitter_range)
            w_end += (w_jitter + jitter_range)

        # Estrai il patch e aggiungilo alla lista
        patch = image[h_start:h_end, w_start:w_end]
        views.append(patch)

    return views


def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x

def load_and_preprocess_image(image_path, width, height, DEVICE):
    image = Image.open(image_path).convert("RGB")
    return image


if torch.cuda.is_available() and ALLOW_CUDA:
    DEVICE = "cuda"
elif (torch.has_mps or torch.backends.mps.is_available()) and ALLOW_MPS:
    DEVICE = "mps"
print(f"Using device: {DEVICE}")

# Parametri e caricamento dei modelli
ckpt_path = "../data/v1-5-pruned-emaonly.ckpt"

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8
seed = 1239876543



# Carica il modello BSRGAN con fattore di super-risoluzione 2x
sf = 2  # Imposta il fattore di super-risoluzione a 2x
bsrgan_model_path = "../data/BSRGANx2.pth"  # Assicurati che il modello sia in questa directory

# Definisci la rete BSRGAN
bsrgan_model = net(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=sf)
bsrgan_model.load_state_dict(torch.load(bsrgan_model_path, map_location=DEVICE), strict=True)
bsrgan_model.eval()
for k, v in bsrgan_model.named_parameters():
    v.requires_grad = False
bsrgan_model = bsrgan_model.to(DEVICE)

# Carica l'immagine PNG esistente
input_image_path = "siura.png" # Sostituisci con il percorso della tua immagine PNG
img = load_and_preprocess_image(input_image_path, WIDTH, HEIGHT, DEVICE)

# Converti l'immagine PIL in un array NumPy con valori uint8
img_np = np.array(img, dtype=np.uint8)
img_L = util.uint2tensor4(img_np).to(DEVICE)  # Converte l'array NumPy in un tensore 4D

# Applica BSRGAN per la super-risoluzione
with torch.no_grad():
    img_E = bsrgan_model(img_L)
img_E = util.tensor2uint(img_E)  # Converte il tensore di output in uint8


bsrgan_model.to("cpu")

# Salva l'immagine finale con BSRGAN applicato
output_image_path = "nome_del_file_finale.png"  # Nome del file di output
util.imsave(img_E, output_image_path)

print(np.array(img).shape)
print(np.array(img_E).shape)

key = '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' 
model_GPT = 'gpt-4o-mini'

prompt = 'image with a better quality of '


views = get_views(img_E, window_size=512)
#Verifica delle dimensioni di ciascun patch generato
for idx, patch in enumerate(views):
    print(f"Patch {idx + 1} dimensioni: {patch.shape}")
    
    messaggi = [ 
        { 
            "role": "system",  
            "content": ( 
                "You are an expert in concise visual image description."
                "For each image, provide a brief, structured description that includes only essential details such as colors, shapes, objects, atmosphere, and visual composition."
                "Focus on key elements for a generative model, using a simple structure of '[adjective] [subject] [material], [color scheme], [photo location], detailed'."
            ) 
        }, 
        { 
            "role": "user",  
            "content": ( 
                f"This is the image patch: {patch}. "
                f"This patch is part of a larger image that represents {prompt}."
                "Provide a structured, concise description that captures the essential features of the image for a generative model. "
                "Use the format '[adjective] [subject] [material], [color scheme], [photo location], detailed'. "
                
                            ) 
        } 
    ]

    
    risposta = OpenAI(api_key=key).chat.completions.create( 
        model= model_GPT, 
        messages=messaggi, 
        temperature=0.7 
    ) 
    print("\nRisposta:",risposta.choices[0].message.content.strip())

    # Parametri per il rumore
    mean = 0           # Media del rumore
    stddev = 25       # Deviazione standard del rumore

    # Crea il rumore gaussiano
    gaussian_noise = np.random.normal(mean, stddev, patch.shape)

    # Aggiungi il rumore all'immagine e clippa i valori tra 0 e 255
    noisy_patch = patch + gaussian_noise
    noisy_patch = np.clip(noisy_patch, 0, 255).astype(np.uint8)

    img_pil = Image.fromarray(patch)
    
    # SAMPLER
    sampler = "ddpm"
    num_inference_steps = 50
    seed = 42

    output_image = pipeline.generate(
        prompt=prompt+risposta.choices[0].message.content.strip(),
        uncond_prompt="",
        input_image=img_pil,
        strength=0.25,
        do_cfg=True,
        cfg_scale=14,
        sampler_name=sampler,
        n_inference_steps=num_inference_steps,
        seed=seed,
        models=model_loader.preload_models_from_standard_weights("../data/v1-5-pruned-emaonly.ckpt", DEVICE),
        device=DEVICE,
        idle_device="cpu",
        tokenizer=CLIPTokenizer("../data/tokenizer_vocab.json", merges_file="../data/tokenizer_merges.txt"),
        )
    
    # Converti l'output generato in un'immagine PIL
    img = Image.fromarray(output_image)
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    axes = axes.ravel()  # Appiattisce l'array per accedere agli assi più facilmente
    axes[0].imshow(patch)
    axes[0].set_title(f"patch1")
    axes[0].axis('off')
    axes[1].imshow(noisy_patch)
    axes[1].set_title(f"patch1 noise")
    axes[1].axis('off')
    axes[2].imshow(img)
    axes[2].set_title(f"patch migliorato")
    axes[2].axis('off')
    plt.tight_layout()
    plt.savefig(f"output_plot_{idx}.png", format="PNG", dpi=300)
    plt.show()

    


    

