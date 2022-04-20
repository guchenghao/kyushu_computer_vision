import cv2 as cv

src1 = cv.imread('./Results.jpeg', cv.IMREAD_GRAYSCALE)
src2 = cv.imread('./Results.jpeg', cv.IMREAD_COLOR)

cv.namedWindow('input_image', cv.WINDOW_AUTOSIZE)

# gray
cv.imshow('input_image', src1)
cv.waitKey(0)
cv.destroyAllWindows()

# color
cv.imshow('input_image', src2)
cv.waitKey(0)
cv.destroyAllWindows()



cv.imwrite("./gray_scale.jpg",src1)
cv.imwrite("./rgb_scale.jpg",src2)
