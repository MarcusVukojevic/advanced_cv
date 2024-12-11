import os
import torch
from PIL import Image
from transformers import CLIPTokenizer
import logging
import numpy as np
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
input_image_path = "nome_del_file.png" # Sostituisci con il percorso della tua immagine PNG
img = Image.open(input_image_path).convert('RGB')  # Assicurati che l'immagine sia in RGB


# Converti l'immagine PIL in un array NumPy con valori uint8
img_np = np.array(img, dtype=np.uint8)
img_L = util.uint2tensor4(img_np).to(DEVICE)  # Converte l'array NumPy in un tensore 4D

# Applica BSRGAN per la super-risoluzione
with torch.no_grad():
    img_E = bsrgan_model(img_L)
img_E = util.tensor2uint(img_E)  # Converte il tensore di output in uint8

# Salva l'immagine finale con BSRGAN applicato
output_image_path = "nome_del_file_finale.png"  # Nome del file di output
util.imsave(img_E, output_image_path)
