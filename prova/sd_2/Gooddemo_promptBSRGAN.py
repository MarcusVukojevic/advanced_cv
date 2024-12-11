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
from GoodPromptBSRGAN import ImageProcessor

# Esempio di utilizzo della classe
if __name__ == "__main__":

    processor = ImageProcessor(device='cuda', allow_cuda=True, allow_mps=False)

    window_size_param = 512
    stride_param = 300#256#362#
    # Parametri
    input_image_path = "immagini/foxy.png"
    bsrgan_model_path = "../data/BSRGANx2.pth"
    
    # Caricamento immagine
    img_iniziale = processor.load_and_preprocess_image(input_image_path)#, 512, 512
    
    img_np = np.array(img_iniziale, dtype=np.uint8)
    print("dimensione dell'immagine iniziale:",img_np.shape)
    
    # Applica BSRGAN
    img_bsrgan1 = processor.apply_bsrgan(img_np, bsrgan_model_path)
    img_bsrgan = processor.apply_bsrgan(img_bsrgan1, bsrgan_model_path)
    print("dimensione dell'immagine ingrandita:",img_bsrgan1.shape)
    

    # Suddividi in patch
    #patches, positions  = processor.get_views1(img_bsrgan, window_size=window_size_param, stride=stride_param, random_jitter=False, vae_scale_factor=1)
    patches  = processor.get_views(img_bsrgan, window_size=window_size_param, stride=stride_param)
    print('numero patches: ',len(patches))
  
    # Processa le patch
    
    processed_patches = processor.process_patches_maky(patches)
  
    # Ricomponi l'immagine
    reassembled_image = processor.assemble_patches(patches=processed_patches, image_shape=img_bsrgan.shape)
    #reassembled_image = processor.assemble_patches1(processed_patches, positions, img_bsrgan.shape, window_size_param)

    # Salva e mostra
    img = Image.fromarray(reassembled_image)
    img.save("immagini/foxy_out_carlotta.png")

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

