# stamatics_project
Python, Open CV, and Deep learning have been used to make a sudoku solver. It will take an unsolved sudoku image as an input and give an image of the solved sudoku as output. 
A function has been made for image pre-processing named "preProcess Image Preprocessing" involves Gaussian blur, thresholding, and dilating the image.
A function named "biggestContour" has been used to find the biggest contour. This largest contour will give us the corners of the sudoku.
A function named "reorder points" reorders the corner points for warp perspective.
A function named "split boxes" splits each number into a single image and later on use a prediction function to predict the digits in those images using a model.
Deep Learning and MNIST dataset were used for recognizing the digits. A model to train the MNIST dataset for predicting digits.
For solving the sudoku, backtracking algorithm.
