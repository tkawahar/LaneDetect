## 撮影日時, 画面回転の取得
import subprocess
import time
import calendar
import re

#video_file = "../mv/MOV_0005.mp4"
#video_file = "../mv/-LLijjSeLXgkl0-iv8rt.mp4"

def get_time_rotate(video_file):

    rotate = 0
    result = subprocess.Popen("ffprobe " + video_file, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    _,out  = result.communicate()
    for i in out.split(b"\n")[1:-1]:
        if i.find(b"creation_time") > -1:
            ctime_str = i.decode()
        if i.find(b"rotate") > -1:
            rotate = int(re.split('[:\r]',i.decode())[1])

    ct_list = re.split('[TZ.]|: ', ctime_str)
    ctime   = time.strptime(ct_list[1]+" "+ct_list[2], "%Y-%m-%d %X")
    #etime   = time.mktime(ctime)     # direct translate
    etime   = calendar.timegm(ctime) # translate from UTC

    #print('ctime:', ctime, '\netime:', etime)
    #print('rotate:', rotate)
    return etime, rotate
