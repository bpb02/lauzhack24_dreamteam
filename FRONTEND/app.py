import streamlit as st
import os
from glob import glob
from openai import OpenAI
import json
from pathlib import Path

# Initialize OpenAI client with API key
client = OpenAI(api_key="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")

# Create cache directory if it doesn't exist
cache_dir = Path("cache")
cache_dir.mkdir(exist_ok=True)
cache_file = cache_dir / "image_descriptions.json"

# Load existing cache
if cache_file.exists():
    with open(cache_file, "r") as f:
        description_cache = json.load(f)
else:
    description_cache = {}

def get_image_description(image_path):
    """Get description of image from ChatGPT with caching"""
    try:
        # Check if description exists in cache
        if image_path in description_cache:
            return description_cache[image_path]
            
        # Provide context about the image based on filename and type
        context = ""
        if "random_forest" in image_path.lower():
            context = "This is a visualization from a Random Forest model showing sales forecasting results including predictions, feature importance, and error metrics."
        elif "lstm" in image_path.lower():
            context = "This is a visualization from an LSTM neural network model showing sales forecasting results including historical data, predictions, and training metrics."
        else:
            context = "This is a visualization showing sales forecasting analysis and results."

        prompt = f"Please describe this visualization in detail. Context: {context}"
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert at analyzing business forecasting visualizations."},
                {"role": "user", "content": prompt}
            ]
        )
        description = response.choices[0].message.content
        
        # Cache the new description
        description_cache[image_path] = description
        with open(cache_file, "w") as f:
            json.dump(description_cache, f)
            
        return description
    except Exception as e:
        return f"Could not generate description: {str(e)}"

swiss_flag = "https://em-content.zobj.net/source/apple/354/flag-switzerland_1f1e8-1f1ed.png"
st.image(swiss_flag, width=50)
st.title("BMS Products Sales Forecasting System - LauzHack 2024 @ EPFL")

st.write("A project developed during LauzHack 2024 at École Polytechnique Fédérale de Lausanne")
st.write("Select a country to view its sales forecast:")

col1, col2, col3 = st.columns(3)

with col1:
    floresland = st.button("Floresland", use_container_width=True)

with col2:
    elbonie = st.button("Elbonie", use_container_width=True)
    
with col3:
    zegoland = st.button("Zegoland", use_container_width=True)

if floresland:
    st.session_state.country = "floresland"
    st.write("You selected: Floresland")
    plots_path = f"../plots/floresland/*.png"
    plots = glob(plots_path)
    
    if len(plots) >= 1:
        st.write("### Random Forest Regression Model Results")
        st.image(plots[0])
        if st.expander("Show Image Description"):
            description = get_image_description(plots[0])
            st.write(description)
    
    if len(plots) >= 2:
        st.write("### Long Short-Term Memory (LSTM) Model Results")
        st.image(plots[1])
        if st.expander("Show Image Description"):
            description = get_image_description(plots[1])
            st.write(description)
    
    for plot in plots[2:]:
        st.image(plot)
        if st.expander("Show Image Description"):
            description = get_image_description(plot)
            st.write(description)

elif elbonie:
    st.session_state.country = "elbonie"
    st.write("You selected: Elbonie")
    plots_path = f"../plots/elbonie/*.png"
    plots = glob(plots_path)
    for plot in plots:
        st.image(plot)
        if st.expander("Show Image Description"):
            description = get_image_description(plot)
            st.write(description)
    
elif zegoland:
    st.session_state.country = "zegoland"
    st.write("You selected: Zegoland")
    plots_path = f"../plots/zegoland/*.png"
    plots = glob(plots_path)
    
    col1, col2 = st.columns(2)
    
    if len(plots) >= 1:
        with col1:
            st.image(plots[0])
            if st.expander("Show Image Description"):
                description = get_image_description(plots[0])
                st.write(description)
    if len(plots) >= 2:
        with col2:
            st.image(plots[1])
            if st.expander("Show Image Description"):
                description = get_image_description(plots[1])
                st.write(description)
    if len(plots) >= 3:
        with col1:
            st.image(plots[2])
            if st.expander("Show Image Description"):
                description = get_image_description(plots[2])
                st.write(description)

st.divider()
