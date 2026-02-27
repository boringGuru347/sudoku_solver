import cv2
import numpy as np
import tensorflow as tf
import json

model = tf.keras.models.load_model("digit_model_centered.keras")
IMG_SIZE = 28


def solve_sudoku(board):
    empty = find_empty(board)
    if not empty:
        return True

    row, col = empty

    for num in range(1, 10):
        if is_valid(board, row, col, num):
            board[row][col] = num

            if solve_sudoku(board):
                return True

            board[row][col] = 0

    return False


def find_empty(board):
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                return (i, j)
    return None


def is_valid(board, row, col, num):
    if num in board[row]:
        return False

    if num in [board[i][col] for i in range(9)]:
        return False

    box_row, box_col = 3 * (row // 3), 3 * (col // 3)
    for i in range(box_row, box_row + 3):
        for j in range(box_col, box_col + 3):
            if board[i][j] == num:
                return False

    return True


def print_board(board):
    for i, row in enumerate(board):
        if i % 3 == 0 and i != 0:
            print("------+-------+------")
        row_str = ""
        for j, val in enumerate(range(9)):
            if j % 3 == 0 and j != 0:
                row_str += " | "
            row_str += str(row[j]) + " "
        print(row_str)


def center_digit(gray_img):
    _, thresh = cv2.threshold(gray_img, 20, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return cv2.resize(gray_img, (IMG_SIZE, IMG_SIZE))

    x, y, w, h = cv2.boundingRect(contours[0])
    digit_crop = gray_img[y : y + h, x : x + w]
    pad = 6
    digit_crop = cv2.copyMakeBorder(
        digit_crop, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0
    )

    centered = cv2.resize(digit_crop, (IMG_SIZE, IMG_SIZE))
    return centered


def predict():
    CROP = 4
    matrix = []
    array = []

    with open("solve/cells_data.json", "r") as f:
        data = json.load(f)

    for i in range(81):
        img_path = f"digit_images/cell_{i}.png"

        img = cv2.imread(img_path)
        if img is None:
            print(f"Cell {i}: Image not found")
            continue
        img = img[CROP:-CROP, CROP:-CROP]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        centered = center_digit(gray)
        contours, _ = cv2.findContours(
            centered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if len(contours) == 0:
            print(f"Cell {i}: No contours found")
            array.append(0)
            data[f"cell_{i}"]["center"].append(0)
            if len(array) == 9:
                matrix.append(array)
                array = []
            continue
        image_num = centered.astype("float32") / 255.0
        image_num = np.expand_dims(image_num, axis=-1)
        image_num = np.expand_dims(image_num, axis=0)

        prediction = model.predict(image_num, verbose=0)
        digit = np.argmax(prediction) + 1
        data[f"cell_{i}"]["center"].append(1)
        array.append(digit)
        if len(array) == 9:
            matrix.append(array)
            array = []

    with open("solve/cells_data.json", "w") as f:
        json.dump(data, f, indent=4)

    return np.array(matrix)


def solve_board():
    board = predict()
    print("Before:")
    print_board(board)

    if solve_sudoku(board):
        print("\nSolved:")
        print_board(board)
    else:
        print("No solution exists — likely a misread digit")

    board = board.reshape(1, 81)
    with open("solve/cells_data.json", "r") as f:
        loaded_data = json.load(f)

    img = cv2.imread("solve/grid_color.png")
    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = img.shape[:2]
    cell_size = min(h, w) // 9
    font_scale = cell_size / 50
    thickness = max(1, cell_size // 25)

    scale_x = w / 450
    scale_y = h / 450

    i = 0
    for key, value in loaded_data.items():
        if value["center"][2] == 0:
            cx = int(value["center"][0] * scale_x)
            cy = int(value["center"][1] * scale_y)
            cv2.putText(
                img,
                str(board[0][i]),
                (cx - cell_size // 5, cy + cell_size // 5),
                font,
                font_scale,
                (255, 0, 0),
                thickness,
                cv2.LINE_AA,
            )
        i = i + 1
    return img
