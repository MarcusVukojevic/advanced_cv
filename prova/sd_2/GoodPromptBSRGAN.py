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


class ImageProcessor:
    """   
    La classe gestisce il dispositivo di calcolo (CPU, CUDA, MPS), carica un modello preaddestrato per la generazione
    di immagini e un tokenizer CLIP per l'elaborazione dei prompt di testo.

    Attributi:
    - device (str): Il dispositivo di calcolo predefinito ("cpu", "cuda" o "mps").
    - allow_cuda (bool): Flag per consentire l'uso di CUDA se disponibile.
    - allow_mps (bool): Flag per consentire l'uso di MPS (Metal Performance Shaders) su dispositivi Apple.
    - model: Il modello preaddestrato caricato per l'elaborazione delle immagini.
    - tokenizer: Il tokenizer CLIP utilizzato per gestire i prompt di testo per il modello di diffusione.
    """
    def __init__(self, device="cpu", allow_cuda=False, allow_mps=False):
        """
        Parametri:
        - device (str): Il dispositivo predefinito ("cpu", "cuda" o "mps").
        - allow_cuda (bool): Se True, abilita l'uso della GPU CUDA se disponibile.
        - allow_mps (bool): Se True, abilita l'uso di MPS su dispositivi Apple se disponibile.

        Azioni:
        - Imposta il dispositivo di calcolo in base alla disponibilità di CUDA o MPS.
        - Carica il modello preaddestrato dai pesi specificati.
        - Inizializza il tokenizer CLIP con i file di vocabolario e merge specificati.
        """
        self.device = device
        self.allow_cuda = allow_cuda
        self.allow_mps = allow_mps
        self._set_device()
        self.model = model_loader.preload_models_from_standard_weights("../data/v1-5-pruned-emaonly.ckpt", self.device)
        self.tokenizer = CLIPTokenizer("../data/tokenizer_vocab.json", merges_file="../data/tokenizer_merges.txt")

    def _set_device(self):
        if torch.cuda.is_available() and self.allow_cuda:
            self.device = "cuda"
        elif (torch.has_mps or torch.backends.mps.is_available()) and self.allow_mps:
            self.device = "mps"
        print(f"Using device: {self.device}")

    def load_and_preprocess_image(self, image_path):
        """
        Carica un'immagine da un percorso specificato e la converte nel formato RGB.
        """
        image = Image.open(image_path).convert("RGB")
        return image#.resize((width, height))
    

    def get_views(self, image, window_size=128, stride=64):
        """
        Suddivide un'immagine in patch di dimensione specificata con un certo stride.

        Parametri:
        - image (numpy array): L'immagine da suddividere, con forma (altezza, larghezza, canali).
        - window_size (int): La dimensione della finestra quadrata (default 128x128).
        - stride (int): La distanza tra l'inizio di una finestra e l'inizio della successiva (default 64).

        Ritorna:
        - views (list): Una lista di patch estratte dall'immagine.

        Dettagli:
        - L'immagine viene suddivisa in finestre sovrapposte di dimensione `window_size x window_size`.
        - Lo stride determina quanto le finestre si sovrappongono: se `stride < window_size`, le finestre si sovrappongono.
        - La funzione calcola il numero di patch che possono essere estratte lungo altezza e larghezza.
        """
        height, width = image.shape[:2]
        num_blocks_height = (height - window_size) // stride + 1
        num_blocks_width = (width - window_size) // stride + 1
        views = []

        for i in range(num_blocks_height * num_blocks_width):
            h_start = (i // num_blocks_width) * stride
            h_end = h_start + window_size
            w_start = (i % num_blocks_width) * stride
            w_end = w_start + window_size

            patch = image[h_start:h_end, w_start:w_end]
            views.append(patch)

        return views
    
    def assemble_patches(self, patches, image_shape, window_size=128, stride=64):
        """
        Ricompone un'immagine originale partendo da una lista di patch sovrapposti.

        Parametri:
        - patches (list): Una lista di patch come array NumPy.
        - image_shape (tuple): La forma dell'immagine originale (altezza, larghezza, canali).
        - window_size (int): La dimensione dei patch quadrati (default 128).
        - stride (int): La distanza tra i patch lungo altezza e larghezza (default 64).

        Ritorna:
        - reassembled_image (numpy array): L'immagine ricomposta come array NumPy di tipo `uint8`.

        Dettagli:
        - I patch vengono sovrapposti e mediati nei punti di sovrapposizione per ottenere una ricostruzione omogenea.
        - Utilizza una matrice di pesi per calcolare il numero di sovrapposizioni in ciascun punto e normalizzare l'immagine finale.
        """
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
    

        ### FORSE DA TOGLIERE ###

    def get_views1(self, image, window_size=512, stride=64, random_jitter=False, vae_scale_factor=1):
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
        - positions: lista di tuple (h_start, w_start) che indicano la posizione di ciascun patch
        """
        height, width = image.shape[:2]
        height //= vae_scale_factor
        width //= vae_scale_factor
        num_blocks_height = int((height - window_size) / stride - 1e-6) + 2 if height > window_size else 1
        num_blocks_width = int((width - window_size) / stride - 1e-6) + 2 if width > window_size else 1
        
        total_num_blocks = int(num_blocks_height * num_blocks_width)
        views = []
        positions = []
        
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
            positions.append((h_start, w_start))

        return views, positions

    def create_weight_mask(self, window_size):
        """
        Crea una maschera di pesatura utilizzando una finestra di Hann 2D.

        Parametri:
        - window_size: dimensione del patch (intero o tuple)

        Ritorna:
        - weight_mask: maschera di pesatura 2D (numpy array)
        """
        if isinstance(window_size, int):
            window_size = (window_size, window_size)
        hanning_window = np.hanning(window_size[0])
        weight_1d = hanning_window / hanning_window.max()
        weight_2d = np.outer(weight_1d, weight_1d)
        weight_mask = weight_2d[..., np.newaxis]  # Aggiungi dimensione canale se necessario
        return weight_mask

    def assemble_patches1(self, patches, positions, image_shape, window_size):
        """
        Ricompone le patch in un'unica immagine, utilizzando maschere di pesatura per un blending migliore.

        Parametri:
        - patches: lista di patch (array NumPy)
        - positions: lista di tuple (h_start, w_start)
        - image_shape: dimensioni dell'immagine originale (altezza, larghezza, canali)
        - window_size: dimensione del patch (intero)

        Ritorna:
        - reassembled_image: immagine ricomposta come array NumPy
        """
        height, width, channels = image_shape
        reassembled_image = np.zeros((height, width, channels), dtype=np.float32)
        weight_matrix = np.zeros((height, width, channels), dtype=np.float32)

        # Crea una maschera di pesatura per i patch
        weight_mask_full = self.create_weight_mask(window_size)

        for patch, (h_start, w_start) in zip(patches, positions):
            patch = np.array(patch, dtype=np.float32)
            h_end = h_start + patch.shape[0]
            w_end = w_start + patch.shape[1]

            # Adatta la maschera di pesatura se il patch è più piccolo (bordi)
            weight_mask = weight_mask_full
            if patch.shape[0] != window_size or patch.shape[1] != window_size:
                weight_mask = self.create_weight_mask((patch.shape[0], patch.shape[1]))

            # Applica la maschera di pesatura al patch
            weighted_patch = patch * weight_mask

            # Aggiungi il patch pesato all'immagine ricomposta
            reassembled_image[h_start:h_end, w_start:w_end] += weighted_patch
            weight_matrix[h_start:h_end, w_start:w_end] += weight_mask

        # Evita la divisione per zero
        weight_matrix = np.maximum(weight_matrix, 1e-8)
        reassembled_image /= weight_matrix

        return reassembled_image.astype(np.uint8)


        ### FINO A QUI ###

    def apply_bsrgan(self, image, model_path):
        """
        Applica il modello BSRGAN (Blind Super-Resolution Generative Adversarial Network) a un'immagine per migliorarne la qualità.

        Parametri:
        - image (numpy array): L'immagine di input come array NumPy (formato uint8).
        - model_path (str): Il percorso del file contenente i pesi del modello preaddestrato BSRGAN.

        Ritorna:
        - numpy array: L'immagine migliorata come array NumPy (formato uint8).

        Dettagli:
        - Inizializza il modello BSRGAN con specifiche configurazioni.
        - Carica i pesi del modello dal file fornito.
        - Converte l'immagine in un tensore PyTorch e la sposta sul dispositivo specificato (CPU o GPU).
        - Applica il modello per elaborare l'immagine e migliorarne la qualità.
        - Converte il tensore di output in un array NumPy per il ritorno.
        """
        
        model = net(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=2)
        model.load_state_dict(torch.load(model_path, map_location=self.device), strict=True)
        model.eval()
        for _, param in model.named_parameters():
            param.requires_grad = False
        model = model.to(self.device)

        img_tensor = util.uint2tensor4(image).to(self.device)
        with torch.no_grad():
            output = model(img_tensor)
        return util.tensor2uint(output)

        ### NON DOVREBBE SERVIRE PIù ###

    def process_patches_GPT(self, patches, model_gpt_key, model_gpt_name, prompt):
        processed_patches = []
        openai_api = OpenAI(api_key=model_gpt_key)
        for idx, patch in enumerate(patches):
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an expert in concise visual image description. "
                        "Provide a structured description for a generative model."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Describe this patch: {patch}",
                },
            ]
            response = openai_api.chat.completions.create(
                model=model_gpt_name, messages=messages, temperature=0.7
            )
            description = response.choices[0].message.content.strip()
            print(f"Patch {idx + 1} description: {description}")

            img_pil = Image.fromarray(patch)
            output_image = self.diffusion(img_pil, description, prompt)
            processed_patches.append(output_image)

        return processed_patches
    
        ### FINO A QUI ###
    
    def process_patches_maky(self, patches):
        """
        Elabora una lista di patch di immagini utilizzando un modello di diffusione con prompt specifici.

        Parametri:
        - patches (list): Una lista di patch di immagini come array NumPy.

        Ritorna:
        - processed_patches (list): Una lista di immagini elaborate, come oggetti PIL.

        Dettagli:
        - Utilizza un insieme di prompt specifici per ciascun patch.
        - Ogni patch viene elaborato tramite il metodo `diffusion`, che genera un'immagine basata sul prompt associato.
        - I prompt descrivono dettagli di un ambiente naturale con un'attenzione particolare a dettagli 
        """
        processed_patches = []
        #prompts = [
        #    "A vibrant and clear sky with subtle gradients of blue, transitioning softly into the horizon, creating a serene backdrop.",
        #    "A calm and bright sky blending into the background with a faint appearance of a serene face, exuding elegance and softness.",
        #    "A detailed composition with a soft sky blending into a serene face with intricate details of braided hair, capturing noble simplicity.",
        #    "A wide and expansive sky with smooth transitions of color, evoking a sense of tranquility and vastness.",
        #    "A softly lit sky with a gradient of cool blues, giving a peaceful and unobtrusive background effect.",
        #    "A delicate shoulder dressed in an intricately embroidered sleeve, blending seamlessly into a renaissance-style dress with detailed stitching.",
        #    "A tightly laced bodice adorned with ornate patterns and subtle highlights, framing the chest area with precision and elegance.",
        #    "The voluminous skirt of a renaissance dress with intricate textures and loops, subtly transitioning into a softly lit sky.",
        #    "A pastoral countryside featuring rolling green hills and a distant horizon, evoking a peaceful rural setting.",
        #    "An intricately detailed sleeve with puffed elements and embroidered patterns, gracefully blending into the flowing lines of the dress.",
        #    "Gracefully positioned hands resting lightly on the renaissance dress, surrounded by ornate embroidery and elegant fabric details.",
        #    "A serene backdrop of clear blue sky with distant mountains softly merging into the landscape, adding depth to the composition.",
        #    "A patch of lush green grass blending into the flowing lines of a voluminous renaissance dress with intricate textures.",
        #    "A richly detailed renaissance dress with symmetrical patterns and golden embellishments, showcasing exquisite craftsmanship.",
        #    "The central portion of a renaissance dress adorned with ornate loops and buttons, exuding refinement and artistic detail.",
        #    "A flowing renaissance dress merging with a lush, green background of rolling grass fields, creating a harmonious composition."
        #]
        #prompts = [
        #    "A serene and bright sky with soft gradients of blue, creating a peaceful backdrop.",
        #    "A clear sky blending into the elegant face of a renaissance woman, surrounded by soft light.",
        #    "A detailed view of the woman's serene face, showcasing intricate features and braided hair.",
        #    "A harmonious combination of a calm sky and the noble face of a renaissance woman.",
        #    "A wide and tranquil expanse of blue sky with smooth color transitions.",
        #    "A soft and radiant sky with a subtle gradient of blue tones.",
        #    "A peaceful sky with smooth color transitions, capturing a tranquil atmosphere.",
        #    "A detailed and expansive sky with a soft gradient, blending seamlessly into the scene.",
        #    "The flowing renaissance dress blending into the calm sky, creating a harmonious composition.",
        #    "A detailed view of the woman's chest and renaissance dress, adorned with intricate embroidery.",
        #    "The voluminous skirt of a renaissance dress, richly textured and flowing elegantly.",
        #    "A sweeping view of the dress blending into the serene blue sky above.",
        #    "A softly lit sky with a gradient of cool blues, creating a peaceful and open backdrop.",
        #    "A picturesque view of distant mountains merging into the clear blue sky, adding depth to the scene.",
        #    "A lush green tree blending into the elegant folds of the renaissance dress.",
        #    "The lower portion of the renaissance dress with detailed patterns and fine craftsmanship.",
        #    "A richly adorned renaissance dress, flowing with intricate details and texture.",
        #    "A graceful transition from the renaissance dress to the distant mountains, blending elegance with nature.",
        #    "A harmonious composition of a flowing dress and the serene mountains in the background.",
        #    "A wide pastoral landscape featuring rolling green hills and a distant horizon.",
        #    "A peaceful countryside blending with the flowing lines of the renaissance dress, creating unity.",
        #    "Gracefully positioned hands resting lightly on the renaissance dress, exuding poise.",
        #    "Delicately folded hands blending into the intricate patterns of the woman's dress.",
        #    "A flowing renaissance dress blending into a mountainous landscape, creating a majestic scene.",
        #    "A detailed view of the dress merging seamlessly with the mountainous backdrop.",
        #    "A pastoral scene with soft grass fields blending into the fine texture of the renaissance dress.",
        #    "The richly adorned renaissance dress with loops, buttons, and detailed embroidery.",
        #    "A richly detailed view of the dress, showcasing fine patterns and symmetrical decorations.",
        #    "The hands of the woman gently resting on the voluminous folds of the ornate dress.",
        #    "A flowing renaissance dress blending harmoniously into the soft green grass of a meadow.",
        #    "The richly textured dress blending seamlessly with the vibrant green of the grass below.",
        #    "A vibrant green meadow merging with the flowing details of the renaissance dress.",
        #    "The lower portion of the dress with intricate patterns and loops, emphasizing fine craftsmanship.",
        #    "A detailed view of the dress adorned with ornate embroidery and golden highlights.",
        #    "The richly adorned dress with buttons and symmetrical decorations, showcasing artistic refinement.",
        #    "A richly detailed view of the dress with fine textures and loops, highlighting its elegance.",
        #    "A flowing renaissance dress with fine patterns and textures, blending gracefully into the scene."
        #]
        prompt_foxy = [
            "A dense forest with vibrant green moss, creating a serene and untouched environment.",
            "A forest background with the left ear of a red fox peeking out, blending naturally into the scene.",
            "A forest background with the right ear of a red fox, sharply contrasting with the vibrant green moss.",
            "A peaceful forest with lush moss and soft natural textures, evoking a tranquil atmosphere.",
            "A dense and tranquil forest with moss-covered ground, illuminated by soft light.",
            "A forest floor covered with lush green moss, fading into shadowed areas.",
            "A serene forest with vibrant moss and natural textures, blending softly into the background.",
            "The left side of the fox's face, showcasing its orange fur and piercing black eyes against the mossy forest.",
            "The right side of the fox's face, with sharp details of its orange fur and attentive black eyes.",
            "A serene forest floor with subtle textures of moss and scattered natural debris.",
            "A dense and textured mossy forest floor, illuminated by soft natural light.",
            "A peaceful forest scene with vibrant moss and scattered twigs, evoking a sense of calm.",
            "A serene moss-covered forest floor, with soft lighting and scattered natural textures.",
            "The nose and white chest fur of a red fox, with soft fur details and vibrant orange tones.",
            "A close-up of the fox's nose and fluffy white chest, emphasizing its natural textures.",
            "The fox's orange back fur, blending seamlessly with the vibrant green moss in the background.",
            "A tranquil mossy forest with soft green tones and natural shadow patterns.",
            "A serene forest floor covered with vibrant moss, illuminated softly by natural light.",
            "A peaceful forest floor with vibrant moss and soft lighting, creating a calm atmosphere.",
            "A detailed view of the fox's white chest fur, blending naturally into its surroundings.",
            "A blend of the fox's white chest and orange back fur, contrasted against the green moss.",
            "The fox's vibrant orange back fur, highlighted by soft lighting and contrasting green moss.",
            "A blend of the fox's orange back fur and the forest floor, creating a natural harmony.",
            "A mossy forest floor with vibrant green textures, illuminated softly by the light.",
            "A serene moss-covered forest floor with scattered twigs, blending naturally into the scene.",
            "The fox's leg blending into the mossy forest, creating a seamless transition.",
            "The fox's legs and bushy tail, with vibrant orange and white fur details, blending into the forest.",
            "The tip of the fox's bushy tail, showcasing its vibrant orange tones and soft fur details.",
            "The fox's bushy tail blending naturally into the mossy forest background, creating harmony.",
            "The fox's bushy tail and the forest floor, blending seamlessly into the vibrant green moss.",
            "A mossy forest floor illuminated softly by natural light, evoking a tranquil atmosphere.",
            "The fox's leg blending into the mossy forest floor, creating a harmonious natural transition.",
            "A combination of the fox's leg, tail, and mossy forest, blending seamlessly into the scene.",
            "A vibrant orange fox tail, contrasting naturally with the soft green tones of the moss.",
            "The fox's bushy tail with vibrant orange and white fur, blending naturally into the mossy forest.",
            "A close-up of the fox's bushy tail, blending seamlessly into the vibrant green moss.",
            "A combination of the fox's bushy tail and the mossy forest, creating a harmonious and serene scene."
        ]
        prompts=prompt_foxy
        
        for idx, patch in enumerate(patches):
            img_pil = Image.fromarray(patch)
            output_image = self.diffusion(img_pil, prompts[idx])
            processed_patches.append(output_image)

        return processed_patches
    
    def diffusion(self, image, description):
        """
        Applica un modello di diffusione a un'immagine per migliorarla o generare nuovi dettagli visivi basati su una descrizione.

        Parametri:
        - image (PIL.Image or numpy array): L'immagine di input da elaborare.
        - description (str): Una descrizione testuale che guida il processo di generazione dell'immagine.

        Ritorna:
        - img (PIL.Image): L'immagine generata o migliorata come oggetto PIL.

        Dettagli:
        - Utilizza una pipeline di diffusione con un sampler specifico per generare un'immagine di alta qualità.
        - La descrizione viene concatenata a un prompt generico per ottenere immagini con dettagli migliorati.
        - Il processo è controllato da parametri come il numero di passi di inferenza, il fattore di scala per il CFG, e il seed per la riproducibilità.
        """
        sampler = "ddpm"
        num_inference_steps = 50
        seed = 42

        output_image = pipeline.generate(
            prompt='Smooth and higher quality image with enhanced details' + description,
            uncond_prompt="",
            input_image=image,
            strength=0.60,
            do_cfg=True,
            cfg_scale=14,
            sampler_name=sampler,
            n_inference_steps=num_inference_steps,
            seed=seed,
            models=self.model,
            device=self.device,
            idle_device=self.device,
            tokenizer=self.tokenizer
            )
        
        # Converti l'output generato in un'immagine PIL
        img = Image.fromarray(output_image)
        return img
    


    def process_patches_maky_parallel(self, patches):
        """
        Parallelizza l'elaborazione dei patch usando CUDA.
        """
        # Verifica se CUDA è disponibile
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Convert patches in tensori e caricali sulla GPU
        patches_tensor = torch.stack([torch.tensor(patch, dtype=torch.uint8) for patch in patches]).to(device)
        prompts = [
            "A serene and bright sky with soft gradients of blue, creating a peaceful backdrop.",
            "A clear sky blending into the elegant face of a renaissance woman, surrounded by soft light.",
            "A detailed view of the woman's serene face, showcasing intricate features and braided hair.",
            "A harmonious combination of a calm sky and the noble face of a renaissance woman.",
            "A wide and tranquil expanse of blue sky with smooth color transitions.",
            "A soft and radiant sky with a subtle gradient of blue tones.",
            "A peaceful sky with smooth color transitions, capturing a tranquil atmosphere.",
            "A detailed and expansive sky with a soft gradient, blending seamlessly into the scene.",
            "The flowing renaissance dress blending into the calm sky, creating a harmonious composition.",
            "A detailed view of the woman's chest and renaissance dress, adorned with intricate embroidery.",
            "The voluminous skirt of a renaissance dress, richly textured and flowing elegantly.",
            "A sweeping view of the dress blending into the serene blue sky above.",
            "A softly lit sky with a gradient of cool blues, creating a peaceful and open backdrop.",
            "A picturesque view of distant mountains merging into the clear blue sky, adding depth to the scene.",
            "A lush green tree blending into the elegant folds of the renaissance dress.",
            "The lower portion of the renaissance dress with detailed patterns and fine craftsmanship.",
            "A richly adorned renaissance dress, flowing with intricate details and texture.",
            "A graceful transition from the renaissance dress to the distant mountains, blending elegance with nature.",
            "A harmonious composition of a flowing dress and the serene mountains in the background.",
            "A wide pastoral landscape featuring rolling green hills and a distant horizon.",
            "A peaceful countryside blending with the flowing lines of the renaissance dress, creating unity.",
            "Gracefully positioned hands resting lightly on the renaissance dress, exuding poise.",
            "Delicately folded hands blending into the intricate patterns of the woman's dress.",
            "A flowing renaissance dress blending into a mountainous landscape, creating a majestic scene.",
            "A detailed view of the dress merging seamlessly with the mountainous backdrop.",
            "A pastoral scene with soft grass fields blending into the fine texture of the renaissance dress.",
            "The richly adorned renaissance dress with loops, buttons, and detailed embroidery.",
            "A richly detailed view of the dress, showcasing fine patterns and symmetrical decorations.",
            "The hands of the woman gently resting on the voluminous folds of the ornate dress.",
            "A flowing renaissance dress blending harmoniously into the soft green grass of a meadow.",
            "The richly textured dress blending seamlessly with the vibrant green of the grass below.",
            "A vibrant green meadow merging with the flowing details of the renaissance dress.",
            "The lower portion of the dress with intricate patterns and loops, emphasizing fine craftsmanship.",
            "A detailed view of the dress adorned with ornate embroidery and golden highlights.",
            "The richly adorned dress with buttons and symmetrical decorations, showcasing artistic refinement.",
            "A richly detailed view of the dress with fine textures and loops, highlighting its elegance.",
            "A flowing renaissance dress with fine patterns and textures, blending gracefully into the scene."
        ]
        
        # Genera le immagini in batch
        output_images = self.diffusion_batch_parallel(patches_tensor, prompts, device)

        # Converti le immagini generate da tensori a immagini PIL
        processed_patches = [Image.fromarray(img.cpu().numpy()) for img in output_images]
        
        return processed_patches

    def diffusion_batch_parallel(self, patches_tensor, descriptions, device):
        """
        Funzione che applica il modello di diffusione in batch.
        """
        sampler = "ddpm"
        num_inference_steps = 50
        seed = 42

        # Converti i tensori in un formato utilizzabile dalla pipeline
        patches_list = [Image.fromarray(patch.cpu().numpy()) for patch in patches_tensor]

        # Genera immagini usando la pipeline
        output_images = pipeline.generate_batch(
            prompts=['Smooth and higher quality image with enhanced details ' + desc for desc in descriptions],
            uncond_prompts=["" for _ in descriptions],
            input_images=patches_list,
            strength=0.60,
            do_cfg=True,
            cfg_scale=14,
            sampler_name=sampler,
            n_inference_steps=num_inference_steps,
            seed=seed,
            models=self.model,
            device=device,  # Imposta CUDA
            idle_device=self.device,
            tokenizer=self.tokenizer
        )

        # La pipeline restituisce immagini generate, che possono essere tensori o numpy array
        return output_images


