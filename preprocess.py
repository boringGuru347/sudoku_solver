import cv2
import numpy as np
import json

def get_grid_img(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(image=gray, threshold1=30, threshold2=200)
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    h_img, w_img = gray.shape[:2]

    ROI = gray[max(0, y-5): min(h_img, y+h+5), max(0, x-5): min(w_img, x+w+5)]
    ROI_color = image[max(0, y-5): min(h_img, y+h+5), max(0, x-5): min(w_img, x+w+5)]  

    cv2.imwrite('solve/grid.png', ROI)
    cv2.imwrite('solve/grid_color.png', ROI_color)  

def sort_cells_into_grid(image_plus_center, tolerance=15):
    cells = sorted(image_plus_center, key=lambda c: c[0][1])
    rows = []
    current_row = [cells[0]]

    for cell in cells[1:]:
        _, cy = cell[0]
        _, prev_cy = current_row[-1][0]

        if abs(cy - prev_cy) <= tolerance:
            current_row.append(cell)
        else:
            current_row.sort(key=lambda c: c[0][0])
            rows.append(current_row)
            current_row = [cell]

    current_row.sort(key=lambda c: c[0][0])
    rows.append(current_row)

    return [cell for row in rows for cell in row]


def preprocess(image, iterations, size):
    data = {}
    image_plus_center = []
    num = 0
    grid_img = cv2.resize(image, (450, 450))  # resize for better detection

    gray = cv2.cvtColor(grid_img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, size)
    fixed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=iterations)


    contours, _ = cv2.findContours(fixed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    shape = grid_img.shape
    expected_area = (shape[0] * shape[1]) / 81
    min_area = expected_area * 0.5  
    max_area = expected_area * 1.2
    contour_image_grid = grid_img.copy()
    for c in contours:
        area = cv2.contourArea(c)
        cv2.drawContours(contour_image_grid, [c], -1, (0, 255, 0), 2)
        if min_area < area < max_area:
            # print(area)
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(grid_img, (x, y), (x + w, y + h), (36, 255, 12), 2)
            ROI = grid_img[y:y+h, x:x+w]
            cell_gray = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
            cell_blur = cv2.bilateralFilter(cell_gray, 5, 75, 75)
            thrs_cell = cv2.threshold(cell_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            cell_contours, _ = cv2.findContours(thrs_cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(cell_contours) == 0:
                resized_cell = np.zeros((28, 28), dtype=np.uint8)
            else:
                resized_cell = cv2.resize(thrs_cell, (28, 28))

            image_plus_center.append(((x + w//2, y + h//2), resized_cell))
            num = num + 1
    return num, image_plus_center, data




def get_cells(image):
    get_grid_img(image)
    grid_image = cv2.imread('solve/grid.png')
    # grid_image = cv2.cvtColor(grid_image, cv2.COLOR_GRAY2BGR)  
    for i in range(1, 9):
        if i == 1:
            size = (2, 2)
            iterations = 1
            num, image_plus_center, data = preprocess(grid_image, iterations, size)
            
            if num == 81:
                sorted_cells = sort_cells_into_grid(image_plus_center, tolerance=15)
            
                for idx, (center, cell_img) in enumerate(sorted_cells):
                    cv2.imwrite(f'digit_images/cell_{idx}.png', cell_img)
                    data[f"cell_{idx}"] = {"center": [center[0], center[1]]}
                with open("solve/cells_data.json", "w") as f:
                    json.dump(data, f, indent=4)
                break
        elif i == 2:
            size = (2, 2)
            iterations = 2
            num, image_plus_center, data = preprocess(grid_image, iterations, size)
            if num == 81:
                sorted_cells = sort_cells_into_grid(image_plus_center, tolerance=15)
            
                for idx, (center, cell_img) in enumerate(sorted_cells):
                    cv2.imwrite(f'digit_images/cell_{idx}.png', cell_img)
                    data[f"cell_{idx}"] = {"center": [center[0], center[1]]}
                with open("solve/cells_data.json", "w") as f:
                    json.dump(data, f, indent=4)
                break

        elif i == 3:
            size = (2, 2)
            iterations = 3
            num, image_plus_center, data = preprocess(grid_image, iterations, size)
            if num == 81:
                sorted_cells = sort_cells_into_grid(image_plus_center, tolerance=15)
            
                for idx, (center, cell_img) in enumerate(sorted_cells):
                    cv2.imwrite(f'digit_images/cell_{idx}.png', cell_img)
                    data[f"cell_{idx}"] = {"center": [center[0], center[1]]}
                with open("solve/cells_data.json", "w") as f:
                    json.dump(data, f, indent=4)
                break

        elif i == 4:
            size = (3, 3)
            iterations = 1
            num, image_plus_center, data = preprocess(grid_image, iterations, size)
            if num == 81:
                sorted_cells = sort_cells_into_grid(image_plus_center, tolerance=15)
            
                for idx, (center, cell_img) in enumerate(sorted_cells):
                    cv2.imwrite(f'digit_images/cell_{idx}.png', cell_img)
                    data[f"cell_{idx}"] = {"center": [center[0], center[1]]}
                with open("solve/cells_data.json", "w") as f:
                    json.dump(data, f, indent=4)
                break

        elif i == 5:
            size = (3, 3)
            iterations = 2
            num, image_plus_center, data = preprocess(grid_image, iterations, size)
            if num == 81:
                sorted_cells = sort_cells_into_grid(image_plus_center, tolerance=15)
            
                for idx, (center, cell_img) in enumerate(sorted_cells):
                    cv2.imwrite(f'digit_images/cell_{idx}.png', cell_img)
                    data[f"cell_{idx}"] = {"center": [center[0], center[1]]}
                with open("solve/cells_data.json", "w") as f:
                    json.dump(data, f, indent=4)
                break

        elif i == 6:
            size = (3, 3)
            iterations = 3
            num, image_plus_center, data = preprocess(grid_image, iterations, size)
            if num == 81:
                sorted_cells = sort_cells_into_grid(image_plus_center, tolerance=15)
            
                for idx, (center, cell_img) in enumerate(sorted_cells):
                    cv2.imwrite(f'digit_images/cell_{idx}.png', cell_img)
                    data[f"cell_{idx}"] = {"center": [center[0], center[1]]}
                with open("solve/cells_data.json", "w") as f:
                    json.dump(data, f, indent=4)
                break

        elif i == 7:
            size = (5, 5)
            iterations = 1
            num, image_plus_center, data = preprocess(grid_image, iterations, size)
            if num == 81:
                sorted_cells = sort_cells_into_grid(image_plus_center, tolerance=15)
            
                for idx, (center, cell_img) in enumerate(sorted_cells):
                    cv2.imwrite(f'digit_images/cell_{idx}.png', cell_img)
                    data[f"cell_{idx}"] = {"center": [center[0], center[1]]}
                with open("solve/cells_data.json", "w") as f:
                    json.dump(data, f, indent=4)
                break

        elif i == 8:
            size = (5, 5)
            iterations = 2
            num, image_plus_center, data = preprocess(grid_image, iterations, size)
            if num == 81:
                sorted_cells = sort_cells_into_grid(image_plus_center, tolerance=15)
            
                for idx, (center, cell_img) in enumerate(sorted_cells):
                    cv2.imwrite(f'digit_images/cell_{idx}.png', cell_img)
                    data[f"cell_{idx}"] = {"center": [center[0], center[1]]}
                with open("solve/cells_data.json", "w") as f:
                    json.dump(data, f, indent=4)
                break
        else:
            raise ValueError("Could not detect exactly 81 cells")  
            
