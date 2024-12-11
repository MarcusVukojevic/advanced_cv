from encoder import VAE_Encoder
from decoder import VAE_Decoder
import torch
import model_converter
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x

def load_and_preprocess_image(image_path, width, height, device):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalizza tra -1 e 1
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
    return image_tensor

# Parametri e caricamento dei modelli
ckpt_path = "../data/v1-5-pruned-emaonly.ckpt"
device = "mps"
WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8
seed = 1239876543
img_path = "../images/torri.png"  # Inserisci il percorso dell'immagine se disponibile, altrimenti lascia None

generator = torch.Generator(device=device)
generator.manual_seed(seed)
latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

# Carica pesi e modelli encoder e decoder
state_dict = model_converter.load_from_standard_weights(ckpt_path, device)
encoder = VAE_Encoder().to(device)
encoder.load_state_dict(state_dict['encoder'], strict=True)
decoder = VAE_Decoder().to(device)
decoder.load_state_dict(state_dict['decoder'], strict=True)

# Determina il latente: da immagine o casuale
if img_path:
    encoder_noise = torch.randn(latents_shape, generator=generator, device=device)
    input_image = load_and_preprocess_image(img_path, WIDTH, HEIGHT, device)
    latents = encoder(input_image, encoder_noise).to(device)
else:
    latents = torch.randn(latents_shape, generator=generator, device=device)

# Funzione per mostrare le immagini
def plot_image_comparison(original, reconstructions, titles):
    fig, axes = plt.subplots(1, len(reconstructions) + 1, figsize=(15, 5))
    axes[0].imshow(original)
    axes[0].set_title("Originale")
    axes[0].axis('off')
    
    for i, (rec, title) in enumerate(zip(reconstructions, titles)):
        axes[i + 1].imshow(rec)
        axes[i + 1].set_title(title)
        axes[i + 1].axis('off')
    plt.show()

# Decodifica dell'immagine originale
original_image = decoder(latents)
original_image = rescale(original_image, (-1, 1), (0, 255), clamp=True).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()[0]

# Appiattisci latenti per applicare PCA e SVD
latents_np = latents.view(-1, LATENTS_WIDTH * LATENTS_HEIGHT).cpu().detach().numpy()

# PCA - Alta retention (99%)
pca_high = PCA(n_components=0.99999999999)
compressed_latents_pca_high = pca_high.fit_transform(latents_np)
decompressed_latents_pca_high = pca_high.inverse_transform(compressed_latents_pca_high)
decompressed_latents_pca_high = torch.tensor(decompressed_latents_pca_high, device=device).view(latents.shape)
image_pca_high = decoder(decompressed_latents_pca_high)
image_pca_high = rescale(image_pca_high, (-1, 1), (0, 255), clamp=True).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()[0]

# PCA - Bassa retention (50%)
pca_low = PCA(n_components=0.5)
compressed_latents_pca_low = pca_low.fit_transform(latents_np)
decompressed_latents_pca_low = pca_low.inverse_transform(compressed_latents_pca_low)
decompressed_latents_pca_low = torch.tensor(decompressed_latents_pca_low, device=device).view(latents.shape)
image_pca_low = decoder(decompressed_latents_pca_low)
image_pca_low = rescale(image_pca_low, (-1, 1), (0, 255), clamp=True).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()[0]

# SVD - Alta retention (99%)
U, S, Vt = np.linalg.svd(latents_np, full_matrices=False)
S_high = np.zeros_like(S)
S_high[:int(0.99 * len(S))] = S[:int(0.99 * len(S))]
compressed_latents_svd_high = U @ np.diag(S_high) @ Vt
decompressed_latents_svd_high = torch.tensor(compressed_latents_svd_high, device=device).view(latents.shape)
image_svd_high = decoder(decompressed_latents_svd_high)
image_svd_high = rescale(image_svd_high, (-1, 1), (0, 255), clamp=True).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()[0]

# SVD - Bassa retention (50%)
S_low = np.zeros_like(S)
S_low[:int(0.5 * len(S))] = S[:int(0.5 * len(S))]
compressed_latents_svd_low = U @ np.diag(S_low) @ Vt
decompressed_latents_svd_low = torch.tensor(compressed_latents_svd_low, device=device).view(latents.shape)
image_svd_low = decoder(decompressed_latents_svd_low)
image_svd_low = rescale(image_svd_low, (-1, 1), (0, 255), clamp=True).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()[0]

# Visualizza le immagini
plot_image_comparison(
    original_image,
    [image_pca_high, image_pca_low, image_svd_high, image_svd_low],
    ["PCA 99% Retention", "PCA 50% Retention", "SVD 99% Retention", "SVD 50% Retention"]
)
