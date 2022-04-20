###
# File: /Users/guchenghao/task1/draw.py
# Project: /Users/guchenghao/task1
# Created Date: Tuesday, April 19th 2022, 9:49:27 am
# Author: GU CHENGHAO
# -----
# 2022
# Last Modified: GU CHENGHAO
# Modified By: GU CHENGHAO
# -----
# Copyright (c) 2022 Personal File
# 
# MIT License
# 
# Copyright (c) 2022 GU CHENGHAO
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# -----
# HISTORY: 
# Date      	By	Comments
# ----------	---	----------------------------------------------------------
###


import cv2 as cv

src = cv.imread('./Results.jpeg')

print(src.shape)

cv.namedWindow('input_image', cv.WINDOW_AUTOSIZE)


cv.line(src, (700, 0), (700, 600), (0, 0, 255), 5)
cv.line(src, (700, 0), (400, 600), (0, 255, 0), 5)
cv.line(src, (700, 0), (1000, 600), (255, 0, 0), 5)


cv.circle(src, (1000, 480), 50, (255, 215, 0), -1)


cv.imshow('input_image', src)

cv.waitKey(0)
cv.destroyAllWindows()


cv.imwrite("./draw_line_circle.jpg",src)