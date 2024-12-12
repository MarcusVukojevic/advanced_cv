import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from GoodPromptBSRGAN import ImageProcessor

# Esempio di utilizzo della classe
if __name__ == "__main__":

    processor = ImageProcessor(device='cuda', allow_cuda=True, allow_mps=False)

    window_size_param = 512
    stride_param = 362

    input_image_path = "immagini/foxy.png"

    processor.FlashSDL(input_image_path=input_image_path, bsrgan_time=3, parallel=2)


