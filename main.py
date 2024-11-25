import streamlit as st
from PIL import Image, ImageOps

st.title("Outpainting Demo")
st.subheader("Upload your image, resize and position it, and generate a new outpainted image.")

# choose the desired output size
size_options = {"9:7": (1152, 896), "7:9": (896, 1152), "19:13":(1216, 832), "13:19":(832, 1216)}
output_size_key = st.selectbox("Choose the output image size:", options=list(size_options.keys()))
output_size = size_options[output_size_key]

if output_size_key == "9:7":
    height = 1152
    width = 896
elif output_size_key == "7:9":
    height = 896
    width = 1152
elif output_size_key == "19:13":
    height = 1216
    width = 832
else:
    height = 832
    width = 1216

# show the size of chosen picture
white_background = Image.new("RGB", output_size, (255, 255, 255))
# st.image(white_background, caption=f"White background size: {output_size_key}", use_container_width=True)


# load an image that must be outpainted
uploaded_file = st.file_uploader("Upload image:", type=["png", "jpg", "jpeg"])
if uploaded_file:
    user_image = Image.open(uploaded_file)
    st.image(user_image, caption="Uploaded Image", use_container_width=True)

    # we want a user was able to scale his image
    scale = st.slider("Scale your image:", 0.1, 3.0, 1.0, 0.1)  
    resized_image = user_image.resize(
        (int(user_image.width * scale), int(user_image.height * scale))
    )
    st.image(resized_image, caption="Resized Image", use_container_width=False)

    # position slider
    x_offset = st.slider("Horizontal offset (X):", -output_size[0] // 2, output_size[0] // 2, 0)
    y_offset = st.slider("Vertical offset (Y):", -output_size[1] // 2, output_size[1] // 2, 0)

    # combine resized image with the white background
    combined_image = white_background.copy()
    paste_x = (output_size[0] - resized_image.width) // 2 + x_offset
    paste_y = (output_size[1] - resized_image.height) // 2 + y_offset
    combined_image.paste(resized_image, (paste_x, paste_y))

    st.image(combined_image, caption="Image on White Background", use_container_width=True)

    # Generate Button
    if st.button("Generate"):
        st.image(combined_image, caption="Generated Image", use_container_width=True)
        st.success("Image successfully generated!")
