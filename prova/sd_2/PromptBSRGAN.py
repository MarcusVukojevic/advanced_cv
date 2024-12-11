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
    def __init__(self, device="cpu", allow_cuda=False, allow_mps=True):
        self.device = device
        self.allow_cuda = allow_cuda
        self.allow_mps = allow_mps
        self._set_device()

    def _set_device(self):
        if torch.cuda.is_available() and self.allow_cuda:
            self.device = "cuda"
        elif (torch.has_mps or torch.backends.mps.is_available()) and self.allow_mps:
            self.device = "mps"
        print(f"Using device: {self.device}")

    def load_and_preprocess_image(self, image_path, width, height):
        image = Image.open(image_path).convert("RGB")
        return image.resize((width, height))

    def get_views(self, image, window_size=128, stride=64, random_jitter=False):
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

    def apply_bsrgan(self, image, model_path, scale_factor=2):
        model = net(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=scale_factor)
        model.load_state_dict(torch.load(model_path, map_location=self.device), strict=True)
        model.eval()
        for _, param in model.named_parameters():
            param.requires_grad = False
        model = model.to(self.device)

        img_tensor = util.uint2tensor4(image).to(self.device)
        with torch.no_grad():
            output = model(img_tensor)
        return util.tensor2uint(output)

    def process_patches(self, patches, model_gpt_key, model_gpt_name, prompt):
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

            # Apply further processing if needed
            processed_patches.append(patch)

        return processed_patches


