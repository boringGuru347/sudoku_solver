import streamlit as st
import cv2
import numpy as np
import os
from preprocess import get_cells
from solver import solve_board

def clear_folder(path):
    for filename in os.listdir(path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif", ".json")):
            file_path = os.path.join(path, filename)
            os.remove(file_path)


print("Done")

st.title("Sudoku Solver (9 X 9)")
st.write("Upload a photo of a Sudoku puzzle and get it solved!")

# --- Make sure output folders exist ---
os.makedirs("solve", exist_ok=True)
os.makedirs("digit_images", exist_ok=True)
# os.makedirs("images", exist_ok=True)

st.write("Please upload a clear photo of a Sudoku puzzle. The grid should be well-lit and as straight as possible with all lines clearly visible for best results.")
st.write("Keep the image resolution between 300 x 300 pixels to 700 x 700 pixels.")
uploaded_file = st.file_uploader("Upload Sudoku image", type=["png", "jpg", "jpeg"])



if uploaded_file is not None:
    # Decode uploaded file into an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.subheader("Original Image")
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)

    # Save a copy for solve_board() which reads from "solve/grid.png"
    # cv2.imwrite("images/grid.png", image)

    if st.button("Solve"):
        with st.spinner("Processing grid..."):
            try:
                # Step 1: Extract cells from the image
                get_cells(image)

                # Step 2: Predict digits, solve, and overlay solution
                result_img = solve_board()
                clear_folder("digit_images")
                clear_folder("solve")

                st.subheader("Solved Sudoku")
                st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), use_container_width=True)

            except SystemExit:
                st.error("Could not detect the Sudoku grid. Try a clearer or better-lit photo.")
            except Exception as e:
                st.error(f"Something went wrong: {e}")