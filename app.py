import cv2
import numpy as np
import streamlit as st
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def get_dominant_colors(image, k=8):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.reshape((-1, 3))
    
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    kmeans.fit(image)
    
    colors = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_
    counts = np.bincount(labels)
    percentages = counts / sum(counts) * 100
    
    return list(zip(colors, percentages))

def is_warm_or_cool(color):
    r, g, b = color
    warm = (r > g and r > b)
    return "Warm" if warm else "Cool"

def complementary_color(color):
    r, g, b = map(int, color)
    return (255 - r, 255 - g, 255 - b)

def plot_color_palette(colors):
    fig, ax = plt.subplots(figsize=(8, 1))
    ax.imshow([colors], extent=[0, len(colors), 0, 1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)
    st.pyplot(fig)

def rgb_to_hex(color):
    return "#" + "".join(f"{c:02x}" for c in color)

def get_color_name(rgb):
    color_names = {
        (255, 0, 0): "Red", (0, 255, 0): "Green", (0, 0, 255): "Blue",
        (255, 255, 0): "Yellow", (0, 255, 255): "Cyan", (255, 0, 255): "Magenta",
        (128, 0, 0): "Maroon", (128, 128, 0): "Olive", (0, 128, 0): "Dark Green",
        (128, 0, 128): "Purple", (0, 128, 128): "Teal", (0, 0, 128): "Navy",
        (255, 165, 0): "Orange", (139, 69, 19): "Brown", (255, 192, 203): "Pink",
        (192, 192, 192): "Silver", (128, 128, 128): "Gray", (0, 0, 0): "Black",
        (255, 255, 255): "White"
    }
    closest_color = min(color_names.keys(), key=lambda x: np.linalg.norm(np.array(x) - np.array(rgb)))
    return color_names[closest_color]

def answer_question(question, colors, percentages):
    if "dominant" in question.lower():
        return f"The dominant color is {get_color_name(colors[0])} with {percentages[0]:.2f}% presence."
    elif "complementary" in question.lower():
        comp_color = complementary_color(colors[0])
        return f"The complementary color is {comp_color}."
    elif "warm or cool" in question.lower():
        return f"The dominant color is a {is_warm_or_cool(colors[0])} color."
    else:
        return "I can answer questions related to dominant color, complementary color, or warm/cool classification."

st.title("🌿 WishCare Color Analyzer - Developed by SK Alauddin")


st.markdown("""
### 🎨 Welcome to Color Explorer!
Upload an image and let the magic begin! Our app analyzes the dominant colors, identifies complementary colors, and even determines if the hues are warm or cool. Perfect for designers, artists, and color enthusiasts!
""")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption=f"Uploaded Image: {uploaded_file.name}", use_container_width=True)
    
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    color_data = get_dominant_colors(image)
    colors, percentages = zip(*color_data)
    color_names = [get_color_name(color) for color in colors]
    
    st.subheader("Color Analysis")
    
    for name, color, percent in zip(color_names, colors, percentages):
        st.markdown(
            f'<div style="display: flex; align-items: center; gap: 15px; margin-bottom: 10px;">'
            f'<div style="width: 50px; height: 50px; background-color: {rgb_to_hex(color)}; border-radius: 5px;"></div>'
            f'<div><strong>{name}</strong></div>'
            f'<div>HEX: {rgb_to_hex(color)}</div>'
            f'<div>RGB: {tuple(map(int, color))}</div>'
            f'<div>Percentage: {percent:.2f}%</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    
    dominant = colors[0]
    comp_color = complementary_color(dominant)
    
    st.subheader("Additional Info")
    st.write(f"**Temperature:** This is a {is_warm_or_cool(dominant)} color.")
    st.write(f"**Complementary Color:** {get_color_name(comp_color)} (HEX: {rgb_to_hex(comp_color)}, RGB: {comp_color})")
    
    st.subheader("Generated Color Palette")
    plot_color_palette(colors)
    
    # VQA Section
    st.subheader("Ask a Question About the Image")
    user_question = st.text_input("Enter your question:")
    if user_question:
        answer = answer_question(user_question, colors, percentages)
        st.write("**Answer:**", answer)
