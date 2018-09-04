import pyrebase
import subprocess
import time

gTrips = [{'pid':'', 'tid':'', 'gps_s':0, 'gps_d':0}]

## listening trip
def stream_handler(message):
    if(message["event"] == "put"):
        path_msg = message["path"].split("/")
        if(len(path_msg) == 3):
            trip     = {'pid':path_msg[1], 'tid':path_msg[2], 'gps_s':message["data"].val()["start"],'gps_d': message["data"].val()["duration"]}
            gTrips.append(trip)

# load API key from api.cfg
config = {}
for line in open('api.cfg','r'):
    line = line.replace('\n','').split(',')
    config[line[0]] = line[1]

# load firebase and database
firebase = pyrebase.initialize_app(config)
db       = firebase.database()

current_num = len(gTrips)

# start watch dog
trip_listening = db.child("trips").stream(stream_handler)
try:
    while True: # main loop
        if current_num != len(gTrips):
            cmd = "python load_video.py " + gTrips[-1]['pid'] + " " + gTrips[-1]['tid']
            subprocess.Popen(cmd.split())
            current_num = len(gTrips)
        time.sleep(1)
except KeyboardInterrupt:
    trip_listening.close()
