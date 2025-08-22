import os
import streamlit as st
import tempfile
from file_checker import checkFile


# Function to configure custom styles
def apply_custom_styles():
    st.markdown(
        """
        <style>
            .reportview-container {
                background: #f0f0f5;
            }
            .css-1aumxhk {
                padding: 10px;
                font-size: 1.2em;
            }
            .css-1x8gf8n {
                color: #ff0000; /* Red for ransomware alerts */
            }
        </style>
        """,
        unsafe_allow_html=True
    )


# Apply custom styles
apply_custom_styles()

# Title and description
st.title("Hybrid Approach in Ransomware Recognition and Classification Using Machine Learning")
st.markdown("""
This project develops a machine learning-based ransomware detection system that processes Portable Executable (PE) files.""")

st.markdown("##### Dataset used: Malware Detection Dataset")
st.subheader("Try it yourself:")

# Sidebar for instructions
with st.sidebar:
    st.header("Instructions")
    st.write("1. Upload a Portable Executable (PE) file to check for ransomware.")
    st.write("2. The system will analyze the file and indicate whether it's legitimate or potentially ransomware.")
    st.write("3. Ensure that the file is not larger than 200MB for optimal performance.")

# File uploader
uploaded_files = st.file_uploader("Upload a file to check for ransomware:", type=['exe', 'dll'],
                                  accept_multiple_files=True)

if uploaded_files:
    with st.spinner("Checking..."):
        results = []
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_file_path = temp_file.name

            # Check the legitimacy of the file
            legitimate = checkFile(temp_file_path)

            # Remove temporary file
            try:
                os.remove(temp_file_path)
            except Exception as e:
                st.error(f"")

            # Store result for display
            if legitimate:
                results.append(f"File **{uploaded_file.name}** seems *LEGITIMATE*! 🎉")
            else:
                results.append(f"File **{uploaded_file.name}** is probably a **RANSOMWARE**!!! ⚠️")

        # Display results
        st.markdown("### Results:")
        for result in results:
            st.write(result)