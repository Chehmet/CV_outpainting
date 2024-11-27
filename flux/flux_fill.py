import torch
from diffusers import FluxFillPipeline
from diffusers.utils import load_image
from PIL import Image

class FluxFillOutpainting:
    def __init__(self):
        self.pipe = FluxFillPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Fill-dev",
            torch_dtype=torch.bfloat16,
            revision="03216e1d673120f9467575e1e84ff1ff21f2c29f"
        )
        self.pipe.enable_model_cpu_offload()

    def offload_model(self):
        self.pipe.to("cpu")

    def _create_outpaint_mask(self, image, top_offset, bottom_offset, left_offset, right_offset):
        """
        Create a mask for outpainting with the specified offsets.
        
        Args:
            image (PIL.Image): Input image
            top_offset (int): Pixels to add at the top
            bottom_offset (int): Pixels to add at the bottom
            left_offset (int): Pixels to add at the left
            right_offset (int): Pixels to add at the right
            
        Returns:
            PIL.Image: Mask image
        """
        width, height = image.size
        new_width = width + left_offset + right_offset
        new_height = height + top_offset + bottom_offset
        
        mask = Image.new("L", (new_width, new_height), 255)  # Create a white mask
        mask.paste(0, (left_offset, top_offset, left_offset + width, top_offset + height))  # Paste the black rectangle
        
        return mask

    def _create_outpaint_image(self, image, top_offset, bottom_offset, left_offset, right_offset):
        """
        Create an extended image for outpainting with the specified offsets.
        
        Args:
            image (PIL.Image): Input image
            top_offset (int): Pixels to add at the top
            bottom_offset (int): Pixels to add at the bottom
            left_offset (int): Pixels to add at the left
            right_offset (int): Pixels to add at the right
            
        Returns:
            PIL.Image: Extended image
        """
        width, height = image.size
        new_width = width + left_offset + right_offset
        new_height = height + top_offset + bottom_offset

        outpaint_image = Image.new("RGB", (new_width, new_height), "white")
        outpaint_image.paste(image, (left_offset, top_offset))

        return outpaint_image

    def generate(self, prompt, image, target_width, target_height, guidance_scale=30, num_inference_steps=50, seed=None):
        """
        Generate outpainted image using Flux Fill with target dimensions.
        
        Args:
            prompt (str): Text prompt describing the desired output
            image (PIL.Image): Input image
            target_width (int): Desired final width of the image
            target_height (int): Desired final height of the image
            guidance_scale (float, optional): Guidance scale for generation. Defaults to 30.
            num_inference_steps (int, optional): Number of inference steps. Defaults to 50.
            seed (int, optional): Random seed for generation. Defaults to None.
            
        Returns:
            PIL.Image: Generated image
        
        Raises:
            ValueError: If target dimensions are smaller than input image dimensions
        """
        # Get current image dimensions
        current_width, current_height = image.size
        
        # Check if target dimensions are valid
        if target_width < current_width or target_height < current_height:
            raise ValueError("Target dimensions must be larger than or equal to input image dimensions")
        
        # Calculate offsets
        left_offset = (target_width - current_width) // 2
        right_offset = target_width - current_width - left_offset
        top_offset = (target_height - current_height) // 2
        bottom_offset = target_height - current_height - top_offset
        
        # Create mask and extended image
        mask_image = self._create_outpaint_mask(
            image, top_offset, bottom_offset, left_offset, right_offset
        )
        extended_image = self._create_outpaint_image(
            image, top_offset, bottom_offset, left_offset, right_offset
        )
        
        # Set up generator
        generator = torch.Generator("cpu")
        if seed is not None:
            generator.manual_seed(seed)

        # Generate the result
        result = self.pipe(
            prompt=prompt,
            image=extended_image,
            mask_image=mask_image,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            max_sequence_length=512,
            generator=generator,
        ).images[0]
        
        return result