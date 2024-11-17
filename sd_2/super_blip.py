import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoProcessor, AutoModelForVisualQuestionAnswering
from network_rrdbnet import RRDBNet as net  # Importa la rete per BSRGAN
import utils_image as util  # Funzioni utili per la gestione delle immagini

import model_loader  # Modulo personalizzato per gestire i modelli
import pipeline  # Modulo personalizzato per la generazione delle immagini

# Configurazione dispositivo
ALLOW_MPS = True
ALLOW_CUDA = True
DEVICE = "cuda" if torch.cuda.is_available() and ALLOW_CUDA else \
         "mps" if torch.backends.mps.is_built() and ALLOW_MPS else "cpu"
print(f"Using device: {DEVICE}")

# Carica il processore e il modello BLIP
processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = AutoModelForVisualQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(DEVICE)

# Funzioni di supporto
def get_views(image, window_size=128, stride=64):
    height, width = image.shape[:2]
    views = []

    for h in range(0, height, stride):
        for w in range(0, width, stride):
            h_end = min(h + window_size, height)
            w_end = min(w + window_size, width)
            patch = image[h:h_end, w:w_end]

            # Se la patch è più piccola del `window_size`, pad con zeri
            if patch.shape[0] < window_size or patch.shape[1] < window_size:
                padded_patch = np.zeros((window_size, window_size, 3), dtype=image.dtype)
                padded_patch[:patch.shape[0], :patch.shape[1], :] = patch
                patch = padded_patch

            views.append(patch)
    return views

def assemble_patches(patches, image_shape, window_size=128, stride=64):
    height, width, channels = image_shape
    reassembled_image = np.zeros((height, width, channels), dtype=np.float32)
    weight_matrix = np.zeros((height, width, channels), dtype=np.float32)

    num_blocks_width = (width - 1) // stride + 1

    for i, patch in enumerate(patches):
        h_start = (i // num_blocks_width) * stride
        w_start = (i % num_blocks_width) * stride
        h_end = min(h_start + window_size, height)
        w_end = min(w_start + window_size, width)

        reassembled_image[h_start:h_end, w_start:w_end] += patch[: h_end - h_start, : w_end - w_start]
        weight_matrix[h_start:h_end, w_start:w_end] += 1

    weight_matrix = np.maximum(weight_matrix, 1)
    reassembled_image /= weight_matrix

    return reassembled_image.astype(np.uint8)

def generate_caption(image_patch):
    inputs = processor(images=Image.fromarray(image_patch), text="Describe the image", return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(**inputs)
    caption = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
    return caption

# Carica il modello BSRGAN
bsrgan_model_path = "BSRGANx2.pth"
sf = 2
bsrgan_model = net(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=sf)

# Verifica la struttura del checkpoint e carica i pesi
checkpoint = torch.load(bsrgan_model_path, map_location=DEVICE)
if "state_dict" in checkpoint:
    print("Caricamento pesi da 'state_dict'")
    bsrgan_model.load_state_dict(checkpoint["state_dict"], strict=False)
else:
    print("Caricamento pesi diretti")
    bsrgan_model.load_state_dict(checkpoint, strict=False)

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
patches = get_views(img_E, window_size=512, stride=256)

# Carica il modello necessario per la pipeline
models = model_loader.preload_models_from_standard_weights("v1-5-pruned-emaonly.ckpt", DEVICE)

# Genera descrizioni per ogni patch e migliora l'immagine
assembly = []
prompt = "image with a better quality of "

for idx, patch in enumerate(patches):
    print(f"Patch {idx + 1} dimensioni: {patch.shape}")
    caption = generate_caption(patch)
    print(f"Caption for Patch {idx + 1}: {caption}")

    # Simula la pipeline di miglioramento immagine
    img_pil = Image.fromarray(patch)
    output_image = pipeline.generate(
        prompt=prompt + caption,
        uncond_prompt="",
        input_image=img_pil,
        strength=0.25,
        do_cfg=True,
        cfg_scale=14,
        sampler_name="ddpm",
        n_inference_steps=50,
        seed=42,
        models=models,
        device=DEVICE,
        idle_device="cpu",
        tokenizer=processor.tokenizer,
    )
    assembly.append(np.array(output_image))

# Ricomponi l'immagine
image_shape = img_E.shape
output_image = assemble_patches(assembly, image_shape, window_size=512, stride=256)

# Salva e mostra l'immagine finale
output_reassembled_path = "immagine_ricomposta.png"
Image.fromarray(output_image.astype(np.uint8)).save(output_reassembled_path)

plt.imshow(output_image)
plt.title("Immagine Ricomposta")
plt.axis("off")
plt.show()
