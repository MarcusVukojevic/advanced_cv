from clip import CLIP
from encoder import VAE_Encoder
from decoder import VAE_Decoder
from diffusion import Diffusion

import model_converter

def preload_models_from_standard_weights(ckpt_path, device):
    """
    Carica e inizializza i modelli CLIP, Encoder, Decoder e Diffusion con pesi standard da un file di checkpoint.

    Parametri:
    - ckpt_path (str): Il percorso del file di checkpoint contenente i pesi pre-addestrati.
    - device (str): Il dispositivo su cui caricare i modelli ("cpu" o "cuda").

    Ritorna:
    - dict: Un dizionario con i modelli inizializzati e caricati con i pesi:
        - 'clip': Modello CLIP per l'elaborazione dei prompt testuali.
        - 'encoder': Modello di codifica VAE (Variational Autoencoder).
        - 'decoder': Modello di decodifica VAE.
        - 'diffusion': Modello di diffusione per la generazione delle immagini.
    """
    state_dict = model_converter.load_from_standard_weights(ckpt_path, device)

    encoder = VAE_Encoder().to(device)
    encoder.load_state_dict(state_dict['encoder'], strict=True)

    decoder = VAE_Decoder().to(device)
    decoder.load_state_dict(state_dict['decoder'], strict=True)

    diffusion = Diffusion().to(device)
    diffusion.load_state_dict(state_dict['diffusion'], strict=True)

    clip = CLIP().to(device)
    clip.load_state_dict(state_dict['clip'], strict=True)

    return {
        'clip': clip,
        'encoder': encoder,
        'decoder': decoder,
        'diffusion': diffusion,
    }