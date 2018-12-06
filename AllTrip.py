import pyrebase
import subprocess
import os
import re

pycmd  = "python"
result = subprocess.Popen(["python","-V"], stdout=subprocess.PIPE,stderr=subprocess.PIPE)
out    = result.communicate()
version= out[0] + out[1]
match  = re.match(r'(Python) (\S{1}).*', version.decode())

if match and int(match.group(2)) == 2:
    pycmd = "python3"


config = {}
for line in open('api.cfg','r'):
    line = line.replace('\n','').split(',')
    config[line[0]] = line[1]

firebase = pyrebase.initialize_app(config)
db       = firebase.database()
ana_db   = db.child("analyzed").get()
pid_db   = db.child("trips").get()

for pid in pid_db.val().keys():
    trip_db  = db.child("trips").child(pid).get()
    print(pid," ----------")
    for tid in trip_db.val().keys():
        print("  ", tid, ":", tid in ana_db.val().keys())
        if not tid in ana_db.val().keys():
            # new trip
            clip_path = "clips/" + pid + "/" + tid + ".mp4"
            if os.path.exists(clip_path):
                # clip file was found, 
                cmd = pycmd + " LaneDetect/load_video.py " + pid + " " + tid
                subprocess.Popen(cmd.split())
                print("   Exec:", cmd)
            else:
                # clip file was not found
                print("  ", clip_path, "not found")
                comment = {"comment":"clip file not found"}
                current = dict(db.child("trips").child(pid).child(tid).get().val())
                if not "comment" in current:
                    current.update(comment)
                    db.child("trips").child(pid).child(tid).set(current)

