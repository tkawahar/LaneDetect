import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pyrebase
import numpy as np
import cv2
import math

### global values
analyze_fn = 0
trip_id    = ""

### functions
# gps dbから位置情報配列を取得
def get_location(gps_db, key_idx):
    t = key_idx
    x = float(gps_db.val()[str(t)]["longitude"])
    y = float(gps_db.val()[str(t)]["latitude"])
    z = float(gps_db.val()[str(t)]["altitude"])
    return [t, x, y, z]

# Movie frame に対応したGPSリストの生成
def make_frame_gps(gps_db, clip, start):
    # 辞書のキーを配列に
    key_list=[]
    for key in gps_db.val().keys():
        key_list.append(int(key))

    # startに一番近いキーを探す (startの次)　https://qiita.com/icchi_h/items/fc0df3abb02b51f81657
    s_idx = np.abs(np.asarray(key_list) - start).argmin()
    if start > key_list[s_idx]:
        s_idx += 1
        
    # DBの中身
    gps_db.val()[str(key_list[s_idx])]
    #gps_db.val()[str(key_list[e_idx])]

    # 線形補完
    fps       = clip.fps  # get fps from movie file
    d_frame   = 1000/fps  # 1 frame time in ms
    fn0       = 0
    gps_frame = []

    idx    = s_idx
    d_time = key_list[idx+1] - key_list[idx] # duration time between gps breadcrumbs
    loc1   = get_location(gps_db, key_list[idx])
    loc2   = get_location(gps_db, key_list[idx+1])

    for _ in range(int(clip.fps * clip.duration)):
        x = loc1[1] + d_frame*fn0*(loc2[1]-loc1[1])/d_time
        y = loc1[2] + d_frame*fn0*(loc2[2]-loc1[2])/d_time
        z = loc1[3] + d_frame*fn0*(loc2[3]-loc1[3])/d_time
        gps_frame.append({"altitude":z,"latitude":y,"longitude":x})
        if ((loc1[0] + d_frame * fn0) > loc2[0]) :
            fn0  = 0
            loc1 = loc2
            idx += 1
            d_time = key_list[idx+1] - key_list[idx]
            loc2   = get_location(gps_db, key_list[idx+1])

    return gps_frame


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


def edgedFromFilterImage(image):
    ## Convert to grayscale and get cannyed edge here.
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
    height, width, _ = image.shape
    region_of_interest_vertices = [
        (0, height),
        (width / 2, height / 2),
        (width, height),
    ]
    
    ## Convert to grayscale and get cannyed edge here.
    gray_image    = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    cannyed_image = cv2.Canny(gray_image, 100, 200)
    #cannyed_image = edgedFromFilterImage(image)
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
    analyze_judge = 0.0
    actual_line = []
    if left_x_end != 0 :
        actual_line.extend([[left_x_start,  max_y, left_x_end,  min_y]])
        analyze_judge += 0.5
    if right_x_end != 0:
        actual_line.extend([[right_x_start, max_y, right_x_end, min_y]])
        analyze_judge += 0.5
    line_image = draw_lines(
        line_image,
        [actual_line],
        thickness=5,
    )

    ## Overlay
    img_dst  = cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0)

    ### save lined jpg image
    global analyze_fn
    global trip_id
    global framed_gps
    if analyze_fn >= len(framed_gps) : # if out of range, copy last data to tail
        framed_gps.append(framed_gps[-1])
    fn="images/" + trip_id + "/{num:05}.jpg".format(num=analyze_fn)
    mpimg.imsave(fn,img_dst)
    framed_gps[analyze_fn]['image'] = fn
    framed_gps[analyze_fn]['judge'] = analyze_judge
    analyze_fn += 1
    #print(fn)
    ###
    
    return img_dst



from moviepy.editor import VideoFileClip
#from IPython.display import HTML
import sys
import os

argvs = sys.argv
argc  = len(argvs)

if argc != 3:
    print("few argments..usage: python loat_video.py pid tid")
    exit(-1)

private_id = argvs[1]
trip_id    = argvs[2]

# load API key from api.cfg
config = {}
for line in open('api.cfg','r'):
    line = line.replace('\n','').split(',')
    config[line[0]] = line[1]

# connect firebase
firebase = pyrebase.initialize_app(config)
storage  = firebase.storage()
db       = firebase.database()

# get database
detail   = db.child("trips").child(private_id).child(trip_id).get()
gps_db   = db.child("breadcrumbs").child(private_id).get()

# download video
if not os.path.isdir('clips'):
    os.mkdir('clips')
input_video = trip_id+".mp4"
storage.child("clips/" + trip_id).download("clips/" + input_video)
clip1 = VideoFileClip(input_video)

# make gps timeline
start_time = detail.val()['start']
framed_gps = make_frame_gps(gps_db, clip1, start_time)

# analyze road lane
#if not os.path.isdir('images'):
#    os.mkdir('images')
image_dir = 'images/' + trip_id + '/'
try:
    os.makedirs(image_dir)
except FileExistsError:
    print('image directory is already existed. exit.')
    exit(-1)
analyze_fn = 0
white_clip = clip1.fl_image(pipeline)

## TODO: Update Database using framed_gps
#  framed_gps should be {"altitude":,"latitude":,"longitude":,"image":,"judge":}
db.child('analyzed').push(framed_gps)
## TODO: Upload images to storage/images
#  storage/images/%tid%/[00000-000fn].jpg
image_list = os.listdir(image_dir)
for imfile in image_list:
    storage.child(image_dir + imfile).put(image_dir + imfile)

#white_clip.write_videofile(output_video, audio=False)