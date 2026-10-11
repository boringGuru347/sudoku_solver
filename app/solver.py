import cv2
import numpy as np
from pathlib import Path
import tensorflow as tf
import json
import time


MODEL_PATH = "digit_model_centered.keras"
SOLVE_DIR = Path("solve")
DIGIT_DIR = Path("digit_images")
CELLS_DATA_PATH = SOLVE_DIR / "cells_data.json"
GRID_IMG_PATH = SOLVE_DIR / "grid_color.png"
model = tf.keras.models.load_model(MODEL_PATH)
IMG_SIZE = 28


def clear_temp_folders():
    for folder in (SOLVE_DIR, DIGIT_DIR):
        folder.mkdir(parents=True, exist_ok=True)
        for file_path in list(folder.iterdir()):
            if file_path.is_file() and file_path.suffix.lower() in {
                ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".json"
            }:
                file_path.unlink()


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


def solve_candidate_set(board):
    row_set = [set(range(1, 10)) for _ in range(9)]
    col_set = [set(range(1, 10)) for _ in range(9)]
    mat_set = [set(range(1, 10)) for _ in range(9)]

    for i in range(9):
        for j in range(9):
            val = board[i][j]
            if val != 0:
                box = (i // 3) * 3 + (j // 3)
                row_set[i].discard(val)
                col_set[j].discard(val)
                mat_set[box].discard(val)

    def solve(board, row_set, col_set, mat_set, x, y):
        for i in range(x, 9):
            for j in range(y if i == x else 0, 9):
                if board[i][j] == 0:
                    box = (i // 3) * 3 + (j // 3)
                    for k in range(1, 10):
                        if k in row_set[i] and k in col_set[j] and k in mat_set[box]:
                            board[i][j] = k
                            row_set[i].discard(k)
                            col_set[j].discard(k)
                            mat_set[box].discard(k)

                            if solve(board, row_set, col_set, mat_set, i, j + 1):
                                return True

                            board[i][j] = 0
                            row_set[i].add(k)
                            col_set[j].add(k)
                            mat_set[box].add(k)
                    return False
        return True

    solved = solve(board, row_set, col_set, mat_set, 0, 0)
    return solved


def solve_mrv(board):
    rows = [[False] * 10 for _ in range(9)]
    cols = [[False] * 10 for _ in range(9)]
    boxes = [[False] * 10 for _ in range(9)]
    empties = []

    def box_id(r, c):
        return (r // 3) * 3 + (c // 3)

    for r in range(9):
        for c in range(9):
            d = board[r][c]
            if d == 0:
                empties.append((r, c))
            else:
                b = box_id(r, c)
                rows[r][d] = True
                cols[c][d] = True
                boxes[b][d] = True

    def dfs(k):
        if k == len(empties):
            return True

        min_idx = k
        min_cnt = 10

        for i in range(k, len(empties)):
            r, c = empties[i]
            b = box_id(r, c)
            cnt = 0
            for d in range(1, 10):
                if not rows[r][d] and not cols[c][d] and not boxes[b][d]:
                    cnt += 1
            if cnt < min_cnt:
                min_cnt = cnt
                min_idx = i
            if min_cnt == 1:
                break

        if min_cnt == 0:
            return False

        empties[k], empties[min_idx] = empties[min_idx], empties[k]
        r, c = empties[k]
        b = box_id(r, c)

        for d in range(1, 10):
            if not rows[r][d] and not cols[c][d] and not boxes[b][d]:
                board[r][c] = d
                rows[r][d] = cols[c][d] = boxes[b][d] = True

                if dfs(k + 1):
                    return True

                board[r][c] = 0
                rows[r][d] = cols[c][d] = boxes[b][d] = False

        empties[k], empties[min_idx] = empties[min_idx], empties[k]
        return False

    return dfs(0)


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

    if not CELLS_DATA_PATH.exists():
        raise FileNotFoundError(f"Missing prediction metadata: {CELLS_DATA_PATH}")

    with open(CELLS_DATA_PATH, "r") as f:
        data = json.load(f)

    for i in range(81):
        img_path = DIGIT_DIR / f"cell_{i}.png"
        # print(img_path)
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Cell {i}: Image not found")
            continue
        img = img[CROP:-CROP, CROP:-CROP]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        centered = center_digit(gray)
        # cv2.imshow("My Image", centered)
        
        # cv2.waitKey(0)  # Wait until any key is pressed
        # cv2.destroyAllWindows()
        contours, _ = cv2.findContours(
            centered.copy(),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        # contour_img = cv2.cvtColor(centered, cv2.COLOR_GRAY2BGR)
        # # cv2.drawContours(
        # #     contour_img,
        # #     contours,
        # #     -1,             # Draw all contours
        # #     (0, 255, 0),    # Green
        # #     2               # Thickness
        # # )
        # # cv2.imshow(f"Cell {i} Contours", contour_img)
        # # cv2.waitKey(0)
        # # cv2.destroyAllWindows()
        if len(contours) == 0:
            # print(f"Cell {i}: No contours found")
            array.append(0)
            data[f"cell_{i}"]["center"].append(0)
            if len(array) == 9:
                matrix.append(array)
                array = []
            continue
        print(f"Cell {i}: {len(contours)} contours found")
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

    with open(CELLS_DATA_PATH, "w") as f:
        json.dump(data, f, indent=4)
    # print(np.array(matrix))
    return np.array(matrix)


def solve_board(algorithm_name="backtracking"):
    board = predict()
    solver_map = {
        "backtracking": solve_sudoku,
        "candidate_set": solve_candidate_set,
        "mrv": solve_mrv,
    }
    print("Before:")
    print_board(board)
    solver = solver_map.get(algorithm_name, solve_sudoku)
    start_time = time.perf_counter()
    solved = solver(board)
    elapsed_seconds = time.perf_counter() - start_time
    if solved:
        print("\nSolved:")
        print_board(board)
    else:
        print("No solution exists — likely a misread digit")

    board = board.reshape(1, 81)
    
    with open(CELLS_DATA_PATH, "r") as f:
        loaded_data = json.load(f)

    img = cv2.imread(str(GRID_IMG_PATH))
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

    clear_temp_folders()
    return img, elapsed_seconds

predict()