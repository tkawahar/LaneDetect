import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math

savefile_num=0


def region_of_interest(img, vertices):
    # Define a blank matrix that matches the image height/width.
    mask = np.zeros_like(img)
    # Create a match color with the same color channel counts.
    match_mask_color = 255

    # Fill inside the polygon
    cv2.fillPoly(mask, vertices, match_mask_color)

    # Returning the image only where mask pixels match
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=3):
    # If there are no lines to draw, exit.
    if lines is None:
        return img

    # Loop over all lines and draw them on the blank image.
    for line in lines:
        for x1, y1, x2, y2 in line:
            #cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    return img


def edgedImage(image):
    ## Convert to grayscale and get cannyed edge here.
    #gray_image    = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    lYellow       = np.array([130,130,40])
    uYellow       = np.array([255,255,100])
    yellow_image  = cv2.inRange(image, lYellow, uYellow)
    lWhite        = np.array([200,200,200])
    uWhite        = np.array([255,255,255])
    white_image   = cv2.inRange(image, lWhite, uWhite)
    picked_image  = cv2.bitwise_or(yellow_image, white_image)

    cannyed_image = cv2.Canny(picked_image, 100, 200)

    return cannyed_image


def pipeline(image):
    ## Crop Range
    height, width, channel = image.shape
    region_of_interest_vertices = [
        (0, height),
        (width / 2, height / 2),
        (width, height),
    ]
    
    ## Convert to grayscale and get cannyed edge here.
    #gray_image    = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #cannyed_image = cv2.Canny(gray_image, 100, 200)
    cannyed_image = edgedImage(image)
    cropped_image = region_of_interest(
        cannyed_image,
        np.array([region_of_interest_vertices], np.int32)
    )

    lines = cv2.HoughLinesP(
        cropped_image,
        rho=6,
        theta=np.pi / 100, #60,
        threshold=50, #160,
        lines=np.array([]),
        minLineLength=20, #40,
        maxLineGap=25
    )

    other_lines  = []
    left_lines   = []
    right_lines  = []
    left_line_x  = []
    left_line_y  = []
    right_line_x = []
    right_line_y = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = float(y2 - y1) / float(x2 - x1)
            #print(line)
            #print(slope)
            if math.fabs(slope) < 0.4: #0.5: # <-- Only consider extreme slope
                other_lines.extend([[x1,y1,x2,y2]])
                continue
            if slope <= 0: # <-- If the slope is negative, left group.
                left_line_x.extend([x1, x2])
                left_line_y.extend([y1, y2])
                #print("left group")
                left_lines.extend([[x1,y1,x2,y2]])
            else: # <-- Otherwise, right group.
                right_line_x.extend([x1, x2])
                right_line_y.extend([y1, y2])
                #print("right group")
                right_lines.extend([[x1,y1,x2,y2]])

    #print("left_lines::", left_lines)
    #print("right_lines::", right_lines)
    line_image = np.zeros(
        (
            image.shape[0],
            image.shape[1],
            3
        ),
        dtype=np.uint8,
    )

    ## Draw detected lines
    color_o = [150,50,155]
    color_l = [0,0,255]
    color_r = [0,255,0]
    line_image = draw_lines(
        line_image,
        [other_lines],
        color=color_o,
        thickness=2,
    )
    line_image = draw_lines(
        line_image,
        [left_lines],
        color=color_l,
        thickness=2,
    )
    line_image = draw_lines(
        line_image,
        [right_lines],
        color=color_r,
        thickness=2,
    )

    ## Calculate Actual lines
    min_y = int(image.shape[0] * 3 / 5) # <-- Just below the horizon
    max_y = image.shape[0] # <-- The bottom of the image

    if left_lines != []:
        poly_left = np.poly1d(np.polyfit(
            left_line_y,
            left_line_x,
            deg=1
        ))
        left_x_start = int(poly_left(max_y))
        left_x_end   = int(poly_left(min_y))
    else:
        left_x_start = 0
        left_x_end   = 0

    if right_lines != []:
        poly_right = np.poly1d(np.polyfit(
            right_line_y,
            right_line_x,
            deg=1
        ))
        right_x_start = int(poly_right(max_y))
        right_x_end   = int(poly_right(min_y))
    else:
        right_x_start = 0
        right_x_end   = 0


    ## Draw Culculated Lines
    actual_line = []
    if left_x_end != 0 :
        actual_line.extend([[left_x_start,  max_y, left_x_end,  min_y]])
    if right_x_end != 0:
        actual_line.extend([[right_x_start, max_y, right_x_end, min_y]])
    line_image = draw_lines(
        line_image,
        [actual_line],
        thickness=5,
    )

    ## Overlay
    img_dst  = cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0)

    ### save lined jpg image
    global savefile_num
    #fn="../ss2/org" + str(savefile_num) + ".jpg"
    #mpimg.imsave(fn, image)
    fn="../ss2/can" + str(savefile_num) + ".jpg"
    cny_img = cv2.merge((cannyed_image,cannyed_image,cannyed_image))
    mpimg.imsave(fn, cny_img)
    fn="../ss2/lin" + str(savefile_num) + ".jpg"
    print(fn)
    mpimg.imsave(fn,img_dst)
    savefile_num = savefile_num + 1
    ###
    
    return img_dst



#line_image = pipeline(mpimg.imread('solidWhiteCurve.jpg'))
#plt.figure()
#plt.imshow(line_image)
#plt.show()


from moviepy.editor import VideoFileClip
from IPython.display import HTML
import sys

argvs = sys.argv
argc  = len(argvs)

if(argc == 1):
    input_video  = "../mv/challenge.mp4"
    output_video = "../mv/challenge_output.mp4"
else:
    input_video  = argvs[1]
    output_video = "output.mp4"


clip1 = VideoFileClip(input_video)
white_clip = clip1.fl_image(pipeline)
white_clip.write_videofile(output_video, audio=False)

