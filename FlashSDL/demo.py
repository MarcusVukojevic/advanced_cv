import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from GoodPromptBSRGAN import ImageProcessor

# Esempio di utilizzo della classe
if __name__ == "__main__":

    prompt='image of a little beatiful fox in the nature with a lot of furr, centered in the image immerse in the nature'
    processor = ImageProcessor(device='mps')
    window_size_param = 512
    stride_param = 362

    input_image_path = "immagini/foxy_2.png"

    processor.FlashSDL(input_image_path=input_image_path,prompt=prompt, bsrgan_time=2, parallel=1)


