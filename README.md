# FlashSDL

FlashSDL is an advanced framework for generating and refining high-resolution images using diffusion models. It leverages cutting-edge techniques, specifically Stable Diffusion from Scherck, to produce detailed outputs guided by user-defined prompts. The pipeline supports image generation, upscaling, and patch-based refinement to achieve high-quality results.

---

## Features

- **Prompt-Based Generation**: Uses detailed text prompts to guide the image generation process.
- **Stable Diffusion Integration**: Integrates Stable Diffusion Lite (SDL), optimized for efficient and high-quality image generation.
- **High-Resolution Output**: Supports iterative upscaling with BSRGAN to enhance resolution and visual quality.
- **Patch Refinement**: Divides the image into patches and refines each section using specific prompts to ensure detail and global consistency.
- **Customizable and Modular**: A flexible and modular pipeline tailored for image generation and restoration tasks.


---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/MarcusVukojevic/advanced_cv.git
   cd advanced_cv/FlashSDL
   ```

2. Set up necessary resources (e.g., pre-trained models):
   - Download vocab.json and merges.txt from https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main/tokenizer and save them in the data folder
   - Download v1-5-pruned-emaonly.ckpt from https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main and save it in the data folder
   - Download BSRGANx2.pth from https://github.com/cszn/BSRGAN

---

## Usage

### Running the Pipeline

1. Configure parameters for image generation.
- **prompt**: Descriptive text prompt guiding the image generation process (used when creating a new image).
- **input_image_path**: Path to save the generated image or path to an existing image to be refined.
- **bsrgan_time**: Number of times to apply BSRGAN upscaling (each pass increases resolution by 2x).
- **parallel**: Number of patches to process in parallel (set to 1 for sequential processing).

2. Run the demo script:
   ```bash
   python demo.py
   ```

## Workflow

1. **Prompt-Based Image Generation**: If generating an image, provide a descriptive prompt. The framework uses Stable Diffusion Lite (SDL) that we have reimplemented from scratch to create an initial 512x512 image.

2. **High resolution image using Bsrgan**: The generated image is passed through BSRGAN for upscaling. The number of upscaling iterations is determined by the bsrgan_time parameter. Each iteration increases the image size by a factor of 2.

3. **Patch-Based Image Refinement**: The upscaled image is divided into 512x512 patches. Each patch is processed with SDL using prompts specific to that patch, allowing detailed refinement. The refined patches are then recombined to form the final high-resolution image.


---

## Modules Overview

### Core Scripts

- **`demo.py`**: Entry point for running the image generation pipeline with prompts.
- **`pipeline.py`**: Coordinates the sequence of operations for diffusion-based generation and enhancement.

### Stable Diffusion from Scherck

- **`network_rrdbnet.py`**: Implements components for high-resolution image generation.
- **`encoder.py`**: Encodes latent space representations with convolutional and attention layers.
- **`decoder.py`**: Decodes latent space into coherent images with attention mechanisms.
- **`diffusion.py`**: Core module for time-step-based diffusion modeling.
- **`ddpm.py`**: Contains the DDPMSampler class for diffusion probabilistic models.

### Utilities

- **`utils_image.py`**: Utility functions for visualization and processing.
- **`model_converter.py`**: Converts pre-trained weights to the required format for use in the pipeline.
- **`model_loader.py`**: Handles loading and initialization of Stable Diffusion models.
- **`attention.py`**: Provides self-attention and cross-attention implementations.

### Specialized Modules

- **`GoodPromptBSRGAN.py`**:  Refines image patches with BSRGAN and prompt-specific adjustments.

## Contact

For questions or issues, please contact [MarcusVukojevic](https://github.com/MarcusVukojevic) [CarlottaGiacchetta] (https://github.com/CarlottaGiacchetta) [ChiaraMusso](https://github.com/ChiaraMuss).

---

## Acknowledgments

This project was developed for the Advanced Computer Vision course at the University of Trento, taught by Professors Elisa Ricci and Niculae Sebe.

