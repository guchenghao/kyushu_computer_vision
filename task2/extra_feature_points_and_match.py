import numpy as np
import matplotlib.pyplot as plt
import cv2

# necessary parameters
input_file1_path = "./images/cups1.jpg"
input_file2_path = "./images/cups2.jpg"
output_file_path = ""
num = 15 # the number of lines, can change the number you want to display


# read images
img1 = cv2.imread(input_file1_path)
img2 = cv2.imread(input_file2_path)


# display and save image
def display(img, output_file_path=output_file_path):
    cv2.imwrite(output_file_path, img)
    plt.imshow(plt.imread(output_file_path))
    plt.axis('off')
    plt.show()


# sift_algorithm
def sift_extra_points(img):
    gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp1 = sift.detect(gray1)
    img1_sift = cv2.drawKeypoints(gray1, kp1, None, flags=4)

    display(img1_sift, "./results/result_sift.jpg")


# akaze algorithm
def akaze_extra_points(img):
    gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    akaze = cv2.AKAZE_create()
    kp1 = akaze.detect(gray1)
    img1_akaze = cv2.drawKeypoints(gray1, kp1, None, flags=4)

    display(img1_akaze, "./results/result_akaze.jpg")

# fast algorithm
def fast_extra_points(img):
    gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fast = cv2.FastFeatureDetector_create(threshold=10)
    kp1 = fast.detect(gray1)
    img1_fast = cv2.drawKeypoints(gray1, kp1, None, flags=4)

    display(img1_fast, "./results/result_fast.jpg")

# the match function
def akaze_match(from_img, to_img, num):
    akaze = cv2.AKAZE_create()
    from_img_gray = cv2.cvtColor(from_img, cv2.COLOR_BGR2GRAY)
    to_img_gray = cv2.cvtColor(to_img, cv2.COLOR_BGR2GRAY)
    from_key_points, from_descriptions = akaze.detectAndCompute(from_img_gray, None)
    to_key_points, to_descriptions = akaze.detectAndCompute(to_img_gray, None)


    bf_matcher = cv2.BFMatcher()
    matches = bf_matcher.match(from_descriptions, to_descriptions)

    matches = sorted(matches, key=lambda x: x.distance)

    match_img = cv2.drawMatches(
        from_img_gray, from_key_points, to_img_gray, to_key_points,
        matches[:num],  None, flags=2
    )

    display(match_img, "./results/match_img.jpg")


# three extra_feature algorithms
sift_extra_points(img1)
akaze_extra_points(img1)
fast_extra_points(img1)


# use the akaze algorithm to make the match
akaze_match(img1, img2, num)


