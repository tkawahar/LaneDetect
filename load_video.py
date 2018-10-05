import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pyrebase
import numpy as np
import cv2
import math

### global values
analyze_itvl = 5  # sec
analyze_fn   = 0
trip_id      = ""
D_HLS        = [ 1, 180, 50, 10]  # Hough Line Detect Parameter [rho,theta,minLine,maxGap]

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
    #if start > key_list[s_idx]:
    #    s_idx += 1
    if start < key_list[s_idx]:
        s_idx -= 1

    # DBの中身
    gps_db.val()[str(key_list[s_idx])]
    #gps_db.val()[str(key_list[e_idx])]

    # 線形補完
    #d_frame   = 1000 * analyze_itvl #1000/clip.fps  # 1 frame time in ms
    #fn0       = 0
    gps_frame = []

    idx    = s_idx
    d_time = key_list[idx+1] - key_list[idx] # duration time between gps breadcrumbs
    loc1   = get_location(gps_db, key_list[idx])
    loc2   = get_location(gps_db, key_list[idx+1])
    c_time = start

    for _ in range(int(clip.duration / analyze_itvl)): ##range(int(clip.fps * clip.duration)):
        x = loc1[1] + (loc2[1]-loc1[1]) * (c_time-loc1[0]) / d_time
        y = loc1[2] + (loc2[2]-loc1[2]) * (c_time-loc1[0]) / d_time
        z = loc1[3] + (loc2[3]-loc1[3]) * (c_time-loc1[0]) / d_time
        gps_frame.append({"altitude":z,"latitude":y,"longitude":x})
        c_time += (analyze_itvl * 1000)
        #if (c_time > loc2[0]) :
        while (c_time > loc2[0]) :
            loc1   = loc2
            idx   += 1
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


def filterImageRGB(image):
    ## Convert to grayscale and get cannyed edge here.
    lYellow       = np.array([130,130,40])
    uYellow       = np.array([255,255,100])
    yellow_image  = cv2.inRange(image, lYellow, uYellow)
    #lWhite        = np.array([200,200,200])
    lWhite        = np.array([170,170,170])
    uWhite        = np.array([255,255,255])
    white_image   = cv2.inRange(image, lWhite, uWhite)
    picked_image  = cv2.bitwise_or(yellow_image, white_image)
    return picked_image


def CreateSourceImage(image):
    ## Pre-Processing Image
    # Convert to grayscale and get cannyed edge here.
    picked_image  = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #picked_image  = filterImageRGB(image)
    picked_image  = cv2.medianBlur(picked_image, 11) # median filter (remove noise)
    cannyed_image = cv2.Canny(picked_image, 50, 100)
    return picked_image, cannyed_image


def LineDetect_Hough(image):
    # original
    lines = cv2.HoughLinesP(
        image,        # cannyed_image,
        lines=np.array([]),
        rho=D_HLS[0], #6,
        theta=np.pi / D_HLS[1], #100, #60,
        threshold=80, #40, #160,
        minLineLength=D_HLS[2], #10, #40,
        maxLineGap=D_HLS[3] #20 #25
    )
    return lines


def LineDetect(image):
    lines = LineDetect_Hough(image)
    #lines = LineDetect_LSD(image)
    return lines


class group_line:
    line_group  = ['others',           'left',                'right']
    g_lines     = {line_group[0]:[[]], line_group[1]:[[],[]], line_group[2]:[[],[]]} # [0]: most steep, [1]: others
    g_line_x    = {                    line_group[1]:[[],[]], line_group[2]:[[],[]]}
    g_line_y    = {                    line_group[1]:[[],[]], line_group[2]:[[],[]]}
    mss         = {                    line_group[1]:0,       line_group[2]:0 }
    left_th     = 0.5 # 5/8 side threshold

    def __init__(self, lines, width):
        # init parameters
        for side in self.line_group:
            if side == 'others':
                self.g_lines[side] = [[]]
            else:
                self.g_lines[side]  = [[],[]]
                self.g_line_x[side] = [[],[]]
                self.g_line_y[side] = [[],[]]
                self.mss[side]      = 0
        # grouping
        for line in lines:
            for x1,y1,x2,y2 in line:

                ## Vertical line
                if x1 == x2:
                    self.g_lines['others'][0].extend([[x1,y1,x2,y2]])
                    continue

                ## Oblique line
                slope     = float(y2 - y1) / float(x2 - x1)
                #intercept = y1 - slope * x1
                # Except close to the horizontal line
                if math.fabs(slope) < 0.2 or math.fabs(slope) > 1.5: #0.5: 
                    self.g_lines['others'][0].extend([[x1,y1,x2,y2]])
                    continue

                if slope < 0:
                    if x1 > width*self.left_th or x2 > width*self.left_th:  # left line but right zone
                        self.g_lines['others'][0].extend([[x1,y1,x2,y2]])
                        continue
                    side = self.line_group[1] # 'left'
                else:
                    if x1 < width*(1-self.left_th) or x2 < width*(1-self.left_th):  # left line but right zone
                        self.g_lines['others'][0].extend([[x1,y1,x2,y2]])
                        continue
                    side = self.line_group[2] # 'right'

                cslope = int(math.fabs(slope*10))
                if cslope > self.mss[side]:
                    self.mss[side]          = cslope
                    self.g_lines[side][1]  += self.g_lines[side][0]
                    self.g_lines[side][0]   = [[x1,y1,x2,y2]]
                    self.g_line_x[side][1] += self.g_line_x[side][0]
                    self.g_line_y[side][1] += self.g_line_y[side][0]
                    self.g_line_x[side][0]  = [x1, x2]
                    self.g_line_y[side][0]  = [y1, y2]
                elif cslope == self.mss[side]:
                    self.g_lines[side][0]  += [[x1,y1,x2,y2]]
                    self.g_line_x[side][0] += [x1, x2]
                    self.g_line_y[side][0] += [y1, y2]
                else:
                    self.g_lines[side][1]  += [[x1,y1,x2,y2]]
                    self.g_line_x[side][1] += [x1, x2]
                    self.g_line_y[side][1] += [y1, y2]


def pipeline(image):
    ### Set Region of Interest Range
    height, width, _ = image.shape # not rerated with rotation
    """
    region_of_interest_vertices = [
        (0, height),
        (width / 2, height / 2),
        (width, height),
    ]
    """
    region_of_interest_vertices = [
        (0, height),
        (0, height / 2),
        (width, height / 2),
        (width, height),
    ]

    ### making base image of line detection
    #picked_image, cannyed_image = CreateSourceImage(image)
    _, cannyed_image = CreateSourceImage(image)
    cropped_image = region_of_interest(
        cannyed_image,
        np.array([region_of_interest_vertices], np.int32)
    )

    ### Line Detect
    lines = LineDetect(cropped_image)
    if isinstance(lines, type(None)):
        return image, 0

    ### Grouping Lines
    gl = group_line(lines, width)

    ### Drawing Lines
    target_line = []
    #min_y        = int(image.shape[0] * 3 / 5) # <-- Just below the horizon
    min_y       = int(image.shape[0] * 1 / 2) # <-- Just below the horizon
    max_y       = image.shape[0] # <-- The bottom of the image
    # Prepare Base Image
    line_image = np.zeros(
        (image.shape[0], image.shape[1], 3), dtype=np.uint8,
    )
    # Draw Extention Lines
    line_color = {gl.line_group[0]:[[150,50,155]],           # others
                  gl.line_group[1]:[[0,250,255],[0,0,255]],  # left
                  gl.line_group[2]:[[250,255,0],[0,255,0]]}  # right
    for i in range(len(gl.line_group)):
        side = gl.line_group[i]
        for j in range(len(gl.g_lines[side])):
            line_image = draw_lines(
                line_image,
                [gl.g_lines[side][j]],
                color=line_color[side][j],
                thickness=2,
            )
    # Calculate Target Lines
    analyze_judge = 0.0
    calc_side = gl.line_group[1:]
    for side in calc_side:
        if gl.g_lines[side][0] != []:
            new_x   = [np.mean(gl.g_line_x[side][0][::2]), np.mean(gl.g_line_x[side][0][1::2])]
            new_y   = [np.mean(gl.g_line_y[side][0][::2]), np.mean(gl.g_line_y[side][0][1::2])]
            poly    = np.poly1d(np.polyfit(new_y, new_x, deg=1))
            x_start = int(poly(max_y))
            x_end   = int(poly(min_y))
            target_line.extend([[x_start, max_y, x_end, min_y]])
            analyze_judge += 0.5
    # Draw Culculated Lines
    line_image = draw_lines(
        line_image,
        [target_line],
        thickness=5,
    )

    ### Overlay line image
    img_dst  = cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0)

    return img_dst, analyze_judge



from moviepy.editor import VideoFileClip
#from IPython.display import HTML
import GetVideoInfo as vi
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
# TODO: Insert Error Process when db is not found
detail   = db.child("trips").child(private_id).child(trip_id).get()
gps_db   = db.child("breadcrumbs").child(private_id).get()

# download video
if not os.path.isdir('clips'):
    os.mkdir('clips')
input_video = trip_id+".mp4"
storage.child("clips/" + trip_id).download("clips/" + input_video)
clip1 = VideoFileClip("clips/" + input_video)
height, width, _ = clip1.get_frame(0).shape

# make gps timeline
#start_time = detail.val()['start']
start_time, rotate = vi.get_time_rotate("clips/" + input_video)
framed_gps = make_frame_gps(gps_db, clip1, start_time*1000)

# analyze road lane
image_dir = 'images/' + trip_id + '/'
try:
    os.makedirs(image_dir)
except FileExistsError:
    print('image directory is already existed. exit.')
    exit(-1)
analyze_fn = 0
#white_clip = clip1.fl_image(pipeline)
#white_clip.write_videofile(output_video, audio=False)
for i in range(int(clip1.duration / analyze_itvl)):
    dst_image, judge = pipeline(clip1.get_frame( i * analyze_itvl))
    if analyze_fn >= len(framed_gps) : # if out of range, copy last data to tail
        framed_gps.append(framed_gps[-1])
    if rotate != 0: # if rotated, reverse height and width
        dst_image = cv2.resize(dst_image, (height, width))
    fn="images/" + trip_id + "/{num:05}.jpg".format(num=analyze_fn)
    mpimg.imsave(fn, dst_image)
    framed_gps[analyze_fn]['image'] = fn
    framed_gps[analyze_fn]['judge'] = judge
    analyze_fn += 1

# Update Database using framed_gps
# framed_gps should be {"altitude":,"latitude":,"longitude":,"image":,"judge":}
db.child('analyzed/'+trip_id).set(framed_gps)
# Upload images to storage/images/%tid%/[00000-000fn].jpg
image_list = os.listdir(image_dir)
for imfile in framed_gps:
    storage.child(imfile['image']).put(imfile['image'])

