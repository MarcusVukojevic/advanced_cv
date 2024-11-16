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
    processor = ImageProcessor()

    # Parametri
    input_image_path = "siura.png"
    bsrgan_model_path = "../data/BSRGANx2.pth"
    model_gpt_key = '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'
    model_gpt_name = "gpt-4o-mini"

    # Caricamento immagine
    img = processor.load_and_preprocess_image(input_image_path, 512, 512)
    img_np = np.array(img)

    # Applica BSRGAN
    img_bsrgan = processor.apply_bsrgan(img_np, bsrgan_model_path)

    # Suddividi in patch
    patches = processor.get_views(img_bsrgan, window_size=128, stride=64)

    # Processa le patch
    processed_patches = processor.process_patches(patches, model_gpt_key, model_gpt_name, prompt="image quality")

    # Ricomponi l'immagine
    reassembled_image = processor.assemble_patches(processed_patches, img_bsrgan.shape, window_size=128, stride=64)

    # Salva e mostra
    Image.fromarray(reassembled_image).save("reassembled_image.png")
    plt.imshow(reassembled_image)
    plt.axis("off")
    plt.show()
