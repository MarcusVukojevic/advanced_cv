import os
import torch
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from network_rrdbnet import RRDBNet as net
import utils_image as util
from transformers import CLIPTokenizer
from openai import OpenAI
import pipeline
import model_loader
from PromptBSRGAN import ImageProcessor

# Esempio di utilizzo della classe
if __name__ == "__main__":
    processor = ImageProcessor(device="mps")

    window_size_param = 512
    stride_param = 362#256
    # Parametri
    input_image_path = "siura.png"
    bsrgan_model_path = "../data/BSRGANx2.pth"
    model_gpt_key = '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'
    model_gpt_name = "gpt-4o-mini"

    # Caricamento immagine
    img_iniziale = processor.load_and_preprocess_image(input_image_path, 512, 512)
    img_np = np.array(img_iniziale, dtype=np.uint8)
    
    # Applica BSRGAN
    img_bsrgan = processor.apply_bsrgan(img_np, bsrgan_model_path)

    # Suddividi in patch
    patches, positions  = processor.get_views(img_bsrgan, window_size=window_size_param, stride=stride_param, random_jitter=False, vae_scale_factor=1)
    print(len(patches))
    # Processa le patch
    
    processed_patches = processor.process_patches_maky(patches)
  
    # Ricomponi l'immagine
    reassembled_image = processor.assemble_patches(processed_patches, positions, img_bsrgan.shape, window_size_param)

    # Salva e mostra
    img = Image.fromarray(reassembled_image)
    img.save("reassembled_image.png")

    fig, axes = plt.subplots(1, 3, figsize=(10, 10))
    axes = axes.ravel()  # Appiattisce l'array per accedere agli assi più facilmente
    axes[0].imshow(img_iniziale)
    axes[0].set_title(f"Immagine Iniziale Piccola")
    axes[0].axis('off')
    axes[1].imshow(img_bsrgan)
    axes[1].set_title(f"Immagine Iniziale Ingrandita")
    axes[1].axis('off')
    axes[2].imshow(img)
    axes[2].set_title(f"Immagine Ingrandita Migliorata")
    axes[2].axis('off')
    plt.tight_layout()
    plt.show() 

