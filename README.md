# 🧩 Sudoku Solver

A computer vision-powered web application that detects, recognizes, and solves a 9×9 Sudoku puzzle from an uploaded image. Built using **Streamlit, OpenCV, and a custom-trained Convolutional Neural Network (CNN)** trained on the [Chars74K](http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/) and [MNIST](https://keras.io/api/datasets/mnist/) datasets.

**[🚀 Try the Sudoku Solver](https://sudoku-solver374.streamlit.app/)**

## ✨ Features

- **Sudoku Grid Detection:** Detects and extracts a Sudoku grid from an uploaded image using OpenCV.
- **Cell Segmentation:** Divides the grid into 81 individual cells using contour detection and morphological preprocessing.
- **Digit Recognition:** Uses a lightweight custom-trained CNN to recognize digits from 1 to 9 and identify empty cells.
- **Multiple Solving Algorithms:** Choose between Backtracking, Candidate Set, and Minimum Remaining Values (MRV).
- **Performance Benchmarking:** Measures solver execution time and compares the performance of all three algorithms.
- **Solution Visualization:** Overlays the solved Sudoku grid onto the original image.

## ⚙️ How It Works

1. **Upload:** Upload a photograph of a printed or handwritten Sudoku puzzle.
2. **Grid Detection:** OpenCV locates the Sudoku grid and extracts it from the image.
3. **Cell Segmentation:** The extracted grid is divided into 81 cells using contour detection and adaptive morphological preprocessing.
4. **Digit Recognition:** The CNN predicts the digit in each non-empty cell. Empty cells are detected automatically.
5. **Puzzle Solving:** Select one of the three algorithms to solve the recognized Sudoku board.
6. **Runtime Measurement:** The selected solver's execution time is measured and displayed in the application.
7. **Solution Display:** The completed Sudoku solution is overlaid onto the original image.

## 🧠 Solver Algorithms

The project implements and benchmarks three Sudoku-solving approaches.

| Algorithm | Description |
|---|---|
| **Backtracking** | Uses recursive depth-first search and validity checks to explore possible digit assignments. |
| **Candidate Set** | Maintains sets of available digits for each row, column, and 3×3 box, reducing repeated validity checks. |
| **MRV (Minimum Remaining Values)** | Selects the empty cell with the fewest valid candidate digits first, helping detect contradictions and reduce unnecessary search. |

### 📊 Performance Benchmark Results

All three algorithms were benchmarked on **1,000,000 Sudoku puzzles**. Each solver produced the correct solution for all puzzles in this benchmark.

| Metric | Backtracking | Candidate Set | MRV |
|---|---:|---:|---:|
| Puzzles tested | 1,000,000 | 1,000,000 | 1,000,000 |
| Correct solutions | 1,000,000 | 1,000,000 | 1,000,000 |
| Total solver time | 744.379 s | 172.039 s | 163.622 s |
| Average time per puzzle | 0.744379 ms | 0.172039 ms | 0.163622 ms |
| Speedup over Backtracking | 1.00× | 4.33× | 4.55× |

**Key observations:**

- The Candidate Set solver achieved approximately **4.33× speedup** over basic Backtracking.
- The MRV solver achieved approximately **4.55× speedup** over basic Backtracking.
- MRV reduced solver execution time by approximately **78%** compared with the Backtracking baseline.
- MRV was approximately **4.9% faster** than the Candidate Set implementation in this benchmark.

These measurements reflect the tested implementations and dataset; performance may vary with puzzle difficulty and execution environment.

### 📁 Results

The `Results/` directory contains the benchmark outputs and performance metrics for all three solvers:

- Backtracking solver results
- Candidate Set solver results
- MRV solver results
- Execution-time measurements and comparative performance metrics

Refer to this directory for the recorded benchmark data and detailed solver comparisons.

## 🏗️ Project Structure

```text
sudoku-solver/
├── app/
│   ├── app.py                      # Streamlit application entry point
│   ├── preprocess.py               # Grid detection and cell extraction
│   ├── solver.py                   # Digit prediction, solver selection, and overlay
│   ├── digit_model_centered.keras  # Trained digit recognition model
│   ├── solve/                      # Temporary grid images and intermediate data
│   └── digit_images/                # Extracted cell images
├── data/
│   └── synthetic_digits/           # Training images organized by digit class
│       ├── 1/
│       ├── 2/
│       └── ...                     # Classes 3–9
├── models/                          # Additional model artifacts
├── Results/                         # Solver benchmark results and performance metrics
├── sample_input_output/             # Sample input images and output references
├── solvers/
│   ├── backtrack_solver.ipynb       # Backtracking implementation and experiments
│   ├── candidate_set_solver.ipynb   # Candidate Set implementation and experiments
│   └── mrv_solver.ipynb              # MRV implementation and experiments
├── model.ipynb                      # CNN model training notebook
├── solver.ipynb                     # Solver exploration and testing
├── test.ipynb                       # Image-processing experiments
├── final_model.ipynb                # Legacy model-training reference
├── Sudoku_Solver_Interview_Guide.pdf
├── requirements.txt                 # Python dependencies
└── README.md
```

## 🛠️ Tech Stack

- **Python** — Core application and solver implementations
- **Streamlit** — Interactive web interface
- **OpenCV** — Image preprocessing, grid detection, and cell segmentation
- **TensorFlow / Keras** — CNN training and digit recognition
- **NumPy** — Numerical operations and array processing
- **Scikit-learn and Matplotlib** — Model evaluation and analysis

## 📦 Installation

### Prerequisites

- Python 3.8 or later
- pip

### Install Dependencies

Clone the repository and navigate to the project directory:

```bash
git clone <your-repository-url>
cd sudoku-solver
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Alternatively, install the main dependencies directly:

```bash
pip install streamlit opencv-python tensorflow numpy scikit-learn matplotlib
```

## ▶️ Running the Application

From the project root, run:

```bash
streamlit run app/app.py
```

Open the local URL displayed in the terminal, typically:

```text
http://localhost:8501
```

Upload a Sudoku image, choose a solving algorithm, and view the recognized puzzle, solution, and execution time.

## 🧪 Training the CNN Model

The digit recognition model is trained using a combination of MNIST and Chars74K-derived digit images. The classification task covers digits **1–9**, with empty cells handled separately during image processing.

### Model Architecture

The model uses a clean, parameter-efficient 2D CNN architecture designed for fast inference and low memory footprint (~10k parameters):

- **Block 1:** `Conv2D` (16 filters, 3×3, ReLU, same padding) + Batch Normalization + Max Pooling (2×2) + Dropout (0.20).
- **Block 2:** `Conv2D` (32 filters, 3×3, ReLU, same padding) + Batch Normalization + Max Pooling (2×2) + Dropout (0.20).
- **Pooling Head:** `GlobalAveragePooling2D` to collapse spatial dimensions and prevent dense weight explosion.
- **Dense Head:** Dense layer with 64 units, Batch Normalization, and Dropout (0.30).
- **Output Layer:** Softmax output layer with 9 classes, corresponding to digits 1–9.

### Training Dataset

**Chars74K**

Download digit images from the [Chars74K dataset](http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/) and organize the relevant images into class-specific directories:

```text
data/synthetic_digits/
├── 1/
├── 2/
├── 3/
├── ...
└── 9/
```

**MNIST**

The MNIST dataset is loaded automatically through `tf.keras.datasets.mnist`.

### Retraining

Open and execute the cells in `model.ipynb`:

```bash
jupyter notebook model.ipynb
```

The trained model is saved as:

```text
app/digit_model_centered.keras
```

## 📸 Image Upload Guidelines

For more reliable grid detection and digit recognition:

- Use a reasonably clear, well-lit image.
- Keep the Sudoku grid as straight and front-facing as possible.
- Ensure that grid boundaries and digits are clearly visible.
- Avoid strong shadows, glare, and excessive blur.
- Images around 300×300 to 700×700 pixels are a recommended starting range; actual results depend on image quality and preprocessing.

## 📝 Notes and Limitations

- The application supports standard **9×9 Sudoku puzzles**.
- Empty cells should be left blank; do not write zeros in them.
- Digit recognition can confuse visually similar digits, particularly 6 and 9 or 4 and 7.
- Incorrect digit recognition can produce an invalid board or an incorrect solution.
- Solver correctness and image-recognition accuracy are separate: a solver can correctly solve the board it receives even if the CNN misreads the original image.
- The benchmark measures the solving algorithms on Sudoku boards; it does not represent the execution time or accuracy of the entire image-processing pipeline.
- Temporary files generated in `app/solve/` and `app/digit_images/` are intended for intermediate processing and are cleared after each solve.

## 🔮 Future Improvements

- Improve digit recognition across different handwriting styles and image conditions.
- Add stronger validation of recognized Sudoku boards before solving.
- Expand testing across more real-world Sudoku images.
- Improve error handling for ambiguous or incorrectly recognized digits.
- Extend benchmark reporting to include additional puzzle difficulty levels and repeated runs.

## 🤝 Feedback

Suggestions and contributions to improve image processing, digit recognition, solver performance, and application usability are welcome. 