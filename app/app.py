
import sys
from pathlib import Path
import os
import cv2
import numpy as np
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from preprocess import get_cells
    from solver import solve_board
except ImportError:
    from app.preprocess import get_cells
    from app.solver import solve_board


SOLVE_DIR = BASE_DIR / "solve"
DIGIT_DIR = BASE_DIR / "digit_images"


def clear_folder(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    for file_path in path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in {
            ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".json"
        }:
            file_path.unlink()


SOLVE_DIR.mkdir(parents=True, exist_ok=True)
DIGIT_DIR.mkdir(parents=True, exist_ok=True)


st.title("Sudoku Solver (9 × 9)")
st.write("Upload a photo of a Sudoku puzzle and get it solved.")

st.write(
    "Use a clear, well-lit, straight image with visible grid lines. "
    "A resolution between 300 × 300 and 700 × 700 pixels is recommended."
)

solver_options = {
    "Backtracking": "backtracking",
    "Candidate Set": "candidate_set",
    "MRV": "mrv",
}
selected_solver = st.selectbox(
    "Choose a Sudoku solving algorithm",
    list(solver_options.keys()),
)

uploaded_file = st.file_uploader(
    "Upload Sudoku image",
    type=["png", "jpg", "jpeg"]
)


if uploaded_file is not None:
    file_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("Could not read the uploaded image.")
        st.stop()

    st.subheader("Original Image")
    st.image(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        width=600
    )

    if st.button(f"Solve with {selected_solver}"):
        try:
            with st.spinner(f"Processing with {selected_solver}..."):
                # Clear previous run's files before processing
                clear_folder(DIGIT_DIR)
                clear_folder(SOLVE_DIR)

                get_cells(image)
                result_img, elapsed_seconds = solve_board(solver_options[selected_solver])

                if result_img is None:
                    st.error("The solver did not return an image.")
                else:
                    st.caption(
                        f"{selected_solver} solver completed in {elapsed_seconds:.4f} seconds."
                    )
                    st.subheader("Solved Sudoku")
                    st.image(
                        cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB),
                        width=600
                    )

        except SystemExit:
            st.error(
                "Could not detect the Sudoku grid. "
                "Try a clearer or better-lit image."
            )

        except Exception as e:
            st.error(f"Something went wrong: {e}")
