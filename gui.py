import streamlit as st
from PIL import Image
import torch
from io import BytesIO
from flux.flux_fill import FluxFillOutpainting
from controlnet.controlnet import ControlNetInpainting

def main():
    st.title("Image Outpainting Tool")
    
    # Initialize session state for models
    if 'flux_model' not in st.session_state:
        st.session_state.flux_model = None
    if 'controlnet_model' not in st.session_state:
        st.session_state.controlnet_model = None
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        # Load and display original image
        image = Image.open(uploaded_file)
        current_width, current_height = image.size
        
        st.image(image, caption="Original Image", use_container_width=True)
        st.write(f"Current dimensions: {current_width}x{current_height}")
        
        # Model selection
        model_type = st.radio(
            "Select Outpainting Model",
            ["Flux Fill", "ControlNet"],
            help="Choose the AI model for outpainting"
        )
        
        # Target dimensions input
        col1, col2 = st.columns(2)
        with col1:
            target_width = st.number_input(
                "Target Width",
                min_value=current_width,
                value=current_width,
                step=8,
                help="Must be larger than or equal to current width"
            )
        
        with col2:
            target_height = st.number_input(
                "Target Height",
                min_value=current_height,
                value=current_height,
                step=8,
                help="Must be larger than or equal to current height"
            )
        
        # Ensure dimensions are multiples of 8
        target_width = ((target_width + 7) // 8) * 8
        target_height = ((target_height + 7) // 8) * 8
        
        if target_width != current_width or target_height != current_height:
            st.info(f"The image will be outpainted from {current_width}x{current_height} to {target_width}x{target_height}")
        
        # Prompt input
        prompt = st.text_area(
            "Enter Prompt",
            "A beautiful, high-quality image",
            help="Describe what you want in the outpainted areas"
        )
        
        # Optional seed
        seed = st.number_input("Random Seed (optional)", value=42, min_value=0, max_value=999999999, step=1)
        
        if st.button("Generate"):
            try:
                with st.spinner("Processing... This may take a while."):
                    if model_type == "Flux Fill":
                        # Initialize Flux model if needed
                        if st.session_state.flux_model is None:
                            st.session_state.flux_model = FluxFillOutpainting()
                        
                        result = st.session_state.flux_model.generate(
                            prompt=prompt,
                            image=image,
                            target_width=target_width,
                            target_height=target_height,
                            seed=seed
                        )
                    else:  # ControlNet
                        # Initialize ControlNet model if needed
                        if st.session_state.controlnet_model is None:
                            st.session_state.controlnet_model = ControlNetInpainting()
                        
                        result = st.session_state.controlnet_model.generate(
                            prompt=prompt,
                            image=image,
                            target_height=target_height,
                            target_width=target_width,
                            seed=seed
                        )
                    
                    # Display result
                    st.image(result, caption="Generated Result", use_container_width=True)
                    
                    # Add download button
                    if result:
                        # Convert to bytes
                        img_byte_arr = BytesIO()
                        result.save(img_byte_arr, format='PNG')
                        img_byte_arr = img_byte_arr.getvalue()
                        
                        st.download_button(
                            label="Download Result",
                            data=img_byte_arr,
                            file_name="outpainted_image.png",
                            mime="image/png"
                        )
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                
        # Add button to free GPU memory
        if st.button("Free GPU Memory"):
            if st.session_state.flux_model:
                st.session_state.flux_model.offload_model()
            if st.session_state.controlnet_model:
                st.session_state.controlnet_model.offload_model()
            st.success("GPU memory freed!")

if __name__ == "__main__":
    main() 