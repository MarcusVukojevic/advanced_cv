# FlashSDL

FlashSDL is an advanced framework for generating and refining images using diffusion models. This repository focuses on leveraging state-of-the-art techniques, specifically Stable Diffusion from Scherck, to produce high-quality outputs guided by user-defined prompts.

---

## Features

- **Prompt-Based Generation**: Leverages user-defined prompts to guide the image generation process.
- **Stable Diffusion Integration**: Utilizes Stable Diffusion from Scherck for generating images from latent space guided by prompts.
- **Customizable Workflow**: Modular pipeline for flexible image generation and restoration.
- **Advanced Attention Mechanisms**: Employs self-attention for improved detail and consistency.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/MarcusVukojevic/advanced_cv.git
   cd advanced_cv/DemoFusion2.0(consegna)
   ```
2. Set up necessary resources (e.g., pre-trained models):
   - Download model weights as specified in the project documentation.

---

## Usage

### Running the Pipeline

1. Configure parameters for image generation.
2. Run the demo script:
   ```bash
   python demo.py --prompt "<your_prompt>" --output <output_folder>
   ```

### Key Parameters
- `--prompt`: Text prompt guiding the image generation process.
- `--output`: Path to the folder where generated images will be saved.
- `--model`: Specify the pre-trained model or weight file to use.

---

## Workflow

1. **Prompt Input**: The user provides a descriptive prompt to guide the generation process.
2. **Latent Space Initialization**: Begins with noise or latent space vectors.
3. **Processing with Stable Diffusion from Scherck**: Scherck's Stable Diffusion enhances the generated images.
4. **Output Generation**: Produces high-resolution, coherent images.

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

- **`GoodPromptBSRGAN.py`**: Enhances image patches using BSRGAN and prompt-specific adjustments.


## Contact

For questions or issues, please contact [MarcusVukojevic](https://github.com/MarcusVukojevic) [CarlottaGiacchetta] (https://github.com/CarlottaGiacchetta) [ChiaraMusso](https://github.com/ChiaraMuss).

---

## Acknowledgments

This project was developed for the Advanced Computer Vision course at the University of Trento, taught by Professors Elisa Ricci and Niculae Sebe.

