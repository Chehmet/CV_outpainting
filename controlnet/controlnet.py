from diffusers import (
    AutoPipelineForImage2Image,
    ControlNetModel,
    DPMSolverMultistepScheduler,
    StableDiffusionXLControlNetPipeline,
)
from diffusers.image_processor import IPAdapterMaskProcessor
import torch
import random
from io import BytesIO
import requests
import torch
from PIL import Image, ImageDraw

class ControlNetInpainting:
    def __init__(self, model_id: str = "RunDiffusion/Juggernaut-XL-v9"):
        controlnet = ControlNetModel.from_pretrained("destitech/controlnet-inpaint-dreamer-sdxl", torch_dtype=torch.float16, variant="fp16")
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16", controlnet=controlnet)
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_model_cpu_offload()
    
    def offload_model(self):
        self.pipe.to("cpu")

    def pad_and_mask_image(self, img, height=1152, width=896):
        smallest = min(img.size)
        img = img.resize((smallest, smallest))

        new_image = Image.new("RGB", (width, height), (255,255,255))
        offset = ((width - img.width) // 2, (height - img.height) // 2)
        new_image.paste(img, offset)

        mask_image = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask_image)
        draw.rectangle(
            [offset, (offset[0] + img.width, offset[1] + img.height)],
            fill=255
        )

        control_image = new_image.convert("RGBA")
        new_controlnet_image = Image.new("RGBA", control_image.size, "WHITE")
        new_controlnet_image.alpha_composite(control_image)

        return new_image, mask_image

    def generate(self, prompt, image, target_height=1152, target_width=896, seed=42):
        new_image, mask_image = self.pad_and_mask_image(image, target_height, target_width)

        generator = torch.Generator(device="cpu").manual_seed(seed)

        output = self.pipe(prompt=prompt, image=new_image, guidance_scale=7, num_inference_steps=30, generator=generator).images[0]
        return output