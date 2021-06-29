#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import time
import pickle
import cv2


# Model initialization
def initialize():
    pickle_in = open('model_test.p', 'rb')
    model = pickle.load(pickle_in)
    return model


# 1.Image Preprocessing
def preProcess(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, 1, 1, 11, 2)
    return imgThreshold


# 2. finding biggest contour
def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            # print(approx)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area

    return biggest, max_area


# 3. reorder points for wrap perspective
def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)

    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]

    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]

    return myPointsNew


# 4. Splitting each digit into single image
def splitBoxes(img):
    rows = np.vsplit(img, 9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r, 9)
        for box in cols:
            boxes.append(box)
    return boxes


# 4. Get Predictions for those images
def getPrediction(boxes, model):
    result = []
    # Preprocessing
    for image in boxes:
        img = np.asarray(image)
        img = img[4:img.shape[0] - 4, 4: img.shape[1] - 4]  # removing extra details in the image
        img = cv2.resize(img, (32, 32))
        img = img / 255
        # print(img.shape)
        img = img.reshape(1, 32, 32, 1)

        # prediction
        preds = model.predict(img)

        classIndex = np.argmax(preds, axis=-1)  # predicts class
        probVal = np.amax(preds)  # prediction accuracy

        # print(classIndex, probVal)

        # Approval at threshold
        if probVal > 0.9:
            result.append(classIndex[0])
        else:
            result.append(0)
    return result


def displayNumbers(img, numbers, color):
    secW = int(img.shape[0] / 9)
    secH = int(img.shape[1] / 9)

    for x in range(0, 9):
        for y in range(0, 9):
            if numbers[(y * 9) + x] != 0:
                cv2.putText(img, str(numbers[(y*9)+x]),
                            (x*secW+int(secW/2)-10, int((y+0.8)*secH)),
                             cv2.FONT_HERSHEY_PLAIN, 3, color, 2, cv2.LINE_AA)
    return img


# In[2]:


def solve(bo):
    find = find_empty(bo)
    if not find:
        return True
    else:
        row, col = find

    for i in range(1,10):
        if valid(bo, i, (row, col)):
            bo[row][col] = i

            if solve(bo):
                return True

            bo[row][col] = 0

    return False


def valid(bo, num, pos):
    # Check row
    for i in range(len(bo[0])):
        if bo[pos[0]][i] == num and pos[1] != i:
            return False

    # Check column
    for i in range(len(bo)):
        if bo[i][pos[1]] == num and pos[0] != i:
            return False

    # Check box
    box_x = pos[1] // 3
    box_y = pos[0] // 3

    for i in range(box_y*3, box_y*3 + 3):
        for j in range(box_x * 3, box_x*3 + 3):
            if bo[i][j] == num and (i,j) != pos:
                return False

    return True


def print_board(bo):
    for i in range(len(bo)):
        if i % 3 == 0 and i != 0:
            print("- - - - - - - - - - - - - ")

        for j in range(len(bo[0])):
            if j % 3 == 0 and j != 0:
                print(" | ", end="")

            if j == 8:
                print(bo[i][j])
            else:
                print(str(bo[i][j]) + " ", end="")


def find_empty(bo):
    for i in range(len(bo)):
        for j in range(len(bo[0])):
            if bo[i][j] == 0:
                return (i, j)  # row, col

    return None


# In[3]:



import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = 3


##
heightImg = 450
widthImg = 450
model = initialize()
##


def sudoku_solver(pathImage):
    # 1. PREPARE IMAGE
    img = cv2.imread(pathImage)
    img = cv2.resize(img, (widthImg, heightImg))
    imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)
    imgThresh = preProcess(img)

    # 2. Finding contours
    imgContours = img.copy()
    imgBigContours = img.copy()
    contours, _ = cv2.findContours(imgThresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 3)

    # 3. finding biggest contour
    biggest, maxArea = biggestContour(contours)
    if biggest.size != 0:
        biggest = reorder(biggest)
        cv2.drawContours(imgBigContours, biggest, -1, (0, 255, 0), thickness=10)

        pts1 = np.float32(biggest)
        pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])

        matrix = cv2.getPerspectiveTransform(pts1, pts2)

        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

        imgDetectedDigits = imgBlank.copy()
        imgWarpColored = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)

        # 4. Image Splitting into single block
        imgSolvedDigits = imgBlank.copy()
        boxes = splitBoxes(imgWarpColored)

        numbers = getPrediction(boxes, model)

        # imgDetectedDigits = displayNumbers(imgDetectedDigits, numbers, color = (255,100,0))
        # print(numbers)

        numbers = np.asarray(numbers)
        posArray = np.where(numbers>0, 0, 1)
        # print(posArray)

        # 5. Find Solution
        board = np.array(numbers)
        board = board.reshape(9,9)
        
        print("UNSOLVED SUDOKU")
        print_board(board)

        solve(board)
        print("SOLVED SUDOKU")
        print_board(board)

        flatList = []
        for sublist in board:
            for item in sublist:
                flatList.append(item)
        solvedNumbers = flatList*posArray
        imgSolvedDigits = displayNumbers(imgSolvedDigits, solvedNumbers, color=(0, 255, 0))

        # 6. Overlay
        pts2 = np.float32(biggest)
        pts1 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])

        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored2 = img.copy()
        imgWarpColored2 = cv2.warpPerspective(imgSolvedDigits, matrix, (widthImg, heightImg))
        inv_perspective = cv2.addWeighted(imgWarpColored2, 1, img, .4, 1)

        # Image show
        cv2.imshow('Solution', inv_perspective)
        cv2.waitKey(0)


# In[12]:


path = input('Enter Image Path : ')
try:
    sudoku_solver(path)
except Exception as e:
    print('Some Error Occurred , '
          'Please Check the specified Path!')


# In[ ]:




