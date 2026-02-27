# 🧩 Sudoku Solver

A computer vision-powered web app that detects, reads, and solves a 9×9 Sudoku puzzle from a photo — built with Streamlit, OpenCV, and a custom-trained CNN trained on the [Chars74K dataset](http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/) and the [MNIST dataset](https://keras.io/api/datasets/mnist/).

---

## How It Works

1. **Upload** a photo of a printed or handwritten Sudoku puzzle
2. **Grid Detection** — OpenCV locates and extracts the Sudoku grid from the image
3. **Cell Segmentation** — The grid is split into 81 individual cells using contour detection with adaptive morphological preprocessing
4. **Digit Recognition** — A CNN model predicts the digit in each cell (digits 1–9; empty cells are detected automatically)
5. **Solving** — The parsed board is solved algorithmically
6. **Output** — The solved puzzle is overlaid on the original image and displayed in the app

---

## Project Structure

```
sudoku-solver/
├── app.py                        # Streamlit app entry point
├── preprocess.py                 # Grid detection and cell extraction (OpenCV)
├── solver.py                     # Sudoku solving logic + digit prediction overlay
├── final_model.ipynb             # CNN model training notebook
├── digit_model_centered.keras    # Trained digit recognition model
├── solve/                        # Temp folder for intermediate grid images & cell data
├── digit_images/                 # Temp folder for extracted cell images
└── synthetic_digits/             # Training dataset (digits 1–9, PNG images)
    ├── 1/
    ├── 2/
    ...
    └── 9/
```

---

## Requirements

- Python 3.8+
- [Streamlit](https://streamlit.io/)
- [OpenCV](https://opencv.org/) (`opencv-python`)
- [TensorFlow](https://www.tensorflow.org/) 2.x
- NumPy
- Scikit-learn
- Matplotlib (For analysis)

Install all dependencies:

```bash
pip install streamlit opencv-python tensorflow numpy scikit-learn matplotlib
```

---

## Running the App

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## Training the Model

The digit recognition model is a CNN trained on a combination of MNIST (digits 1–9, zeros excluded) and a custom synthetic digit dataset.

To retrain the model, open and run all cells in:

```bash
jupyter notebook final_model.ipynb
```

The trained model will be saved as `digit_model_centered.keras`.

### Model Architecture

- 2× Conv2D (32 filters) + BatchNorm + MaxPooling + Dropout
- 2× Conv2D (64 filters) + BatchNorm + MaxPooling + Dropout
- Dense (256) + BatchNorm + Dropout
- Softmax output (9 classes: digits 1–9)

### Dataset

- **[Chars74K](http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/)** — Place the digit images under `synthetic_digits/` organized by class:

```
synthetic_digits/
├── 1/   ← PNG images of digit 1
├── 2/
...
└── 9/
```

- **[MNIST](https://keras.io/api/datasets/mnist/)** — Downloaded automatically via `tf.keras.datasets.mnist`.

---

## Image Guidelines

For best results when uploading a puzzle photo:

- Resolution between **300×300 and 700×700 pixels**
- The grid should be **well-lit** with no harsh shadows
- Hold the camera **as straight/flat** as possible — avoid tilted angles
- All grid lines should be **clearly visible**

---

## Notes

- The app only supports standard **9×9 Sudoku** puzzles
- Digits **0** (empty cells) are handled by detecting blank cells — do not write zeros in the puzzle
- Temporary files in `solve/` and `digit_images/` are automatically cleared after each solve