# python mp4Detect.py --video (video) --tracker (tracker name)
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2
import csv
import numpy as np
from numpy import double, savetxt
import pandas as pd

# Construct the argument parser and parse the arguments
# If no tracker is specified, use csrt
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str, help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="csrt", help="OpenCV object tracker type")
args = vars(ap.parse_args())

# If a video path was not supplied, end the program
# otherwise, grab a reference to the video file
if not args.get("video", False):
	print("[ERROR] video not file entered...")
	quit()
else:
	print("[INFO] attempting to connect to video file...")
	vs = cv2.VideoCapture(args["video"])

# Initialize a dictionary that maps strings to their corresponding OpenCV object tracker implementations
OPENCV_OBJECT_TRACKERS = {
	#"csrt": cv2.legacy.TrackerCSRT_create,
	#"kcf": cv2.TrackerKCF_create,
	
	#"mil": cv2.TrackerMIL_create,
	#"tld": cv2.legacy.TrackerTLD_create,
	
	#"mosse": cv2.legacy.TrackerMOSSE_create
}
# Grab the appropriate object tracker using our dictionary of OpenCV object tracker objects
# Initialize the bounding box coordinates of the object, the FPS throughput estimator,
# the width of video frame, the frame counter, and arrays to be put into dataframe later
tracker = cv2.legacy.TrackerCSRT_create
initBB = None
fps = None
frameCnt = -1
hArray = []
wArray = []
xArray = []
yArray = []
frameArray = []
fpsArray = []
timeArray = []
winHeiArray = []
winWidArray = []
resFactArray = []
p2marrayx= []
p2marrayy= []


prev_time = time.time()
current_time = time.time()
sum_time = 0
zoomed_cnt = 0



PixeltoMeter=double(input("Eneter pixel to meter value. , negative 1 if unknown: "  ))



# loop over frames from the video stream
while True:
	# grab the current frame, and check to see if we have reached the end of the stream
	frame = vs.read()
	frame = frame[1] if args.get("video", False) else frame
	if frame is None:
		break
	frameCnt += 1

	# grab the origional height and width
		# if width had been changed, the zoom in / zoom out happens here
	(H, W) = frame.shape[:2]
	frame = imutils.resize(frame, W + zoomed_cnt*100)

	# check to see if we are currently tracking an object
	if initBB is not None:
		# check to see if the tracking was a success
		# if we are, grab the new bounding box coordinates of the object
		(success, box) = tracker.update(frame)
		if success:
			(x, y, w, h) = [int(v) for v in box]
			cv2.rectangle(frame, (x, y), (x + w, y + h),(0, 255, 0), 1)


		# update FPS counter and append to arrays
		fps.update()
		fps.stop()

		prev_time = current_time
		current_time = time.time()
		sum_time = sum_time + fps.fps() / 60 * (current_time-prev_time)

		
		hArray.append(h)
		wArray.append(w)
		xArray.append((x*W)/(W+zoomed_cnt*100))
		yArray.append((y*W)/(W+zoomed_cnt*100))
		frameArray.append(frameCnt)
		fpsArray.append(fps.fps())
		timeArray.append(sum_time)
		winHeiArray.append(H + zoomed_cnt*100)
		winWidArray.append(W + zoomed_cnt*100)
		resFactArray.append((W+zoomed_cnt*100)/W)
		p2marrayx.append(x/PixeltoMeter)
		p2marrayy.append(y/PixeltoMeter)
		# initialize the set of information to be displayed on the frame
		# then loop over the tuples and write them on the frame
		info = [
			("FPS", "{:.2f}".format(fps.fps())),
			("frame", frameCnt),
			("time", sum_time + fps.fps() / 60 * (current_time-prev_time)),
			("zoomed amount", zoomed_cnt),
			("resolution factor", (W+zoomed_cnt*100)/W),
		]
		for (i, (k, v)) in enumerate(info):
			text = "{}: {}".format(k, v)
			cv2.putText(frame, text, (10, (i * 20) + 20),	cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

	# show the output frame and wait for a key to be pressed
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	# if the "s" key was pressed, we are going to create a bounding box to track
	if key == ord("s"):
		# clear the old ROI, and set up the tracker uaing the specified tracker
		# create the bounding box around the object we want to track
			#pressing ENTER selects the box as the ROI
		# start the tracker using the supplied bounding box coordinates, then start the FPS throughput estimator as well
		#Also, mark the start of tracking
		tracker = cv2.legacy.TrackerCSRT_create()
		initBB = cv2.selectROI("Frame", frame, fromCenter=True, showCrosshair=False)
		tracker.init(frame, initBB)
		fps = FPS().start()
   	# if the "+" key was pressed, enlarge the frame
	elif key == ord("p"):
		zoomed_cnt += 1
	# if the "-" key was pressed, shrink the frame
	elif key == ord("o"):
		zoomed_cnt -= 1
	# if the "q" key was pressed, break out of the loop
	elif key == ord("q"):
		break

# make and open the csv file, then make the header
with open('test.csv', 'w', newline='\n') as file:
	writer = csv.writer(file)
	dataset = pd.DataFrame({'frame': frameArray, 'widow width': winWidArray, 'widow height': winHeiArray, 'resolution factor': resFactArray, 'frameRate': fpsArray, 'time': timeArray, 'height': hArray,'width': wArray, 'x pos': xArray,'y pos': yArray,'y in meters':p2marrayy,'x in meters':p2marrayx})
	dataset.to_csv('test.csv',sep=',')

# release the file pointer and close all windows
vs.release()
cv2.destroyAllWindows()

