"""# **1. Import the packages**"""

import numpy as np
import pandas as pd
import cv2
import imutils
from scipy.spatial import distance as dist
import argparse
from IPython.display import HTML
from base64 import b64encode



"""# **2. Social Distancing Config**"""

class social_dist_config():
  MODEL_PATH = 'D:\SocDistDect\yolo-coco-data'
  MIN_CONF = 0.3
  NMS_THRESH = 0.3
  USE_GPU = True
  MIN_DIST = 50




"""# **3. Detection**"""

def detect_people(frame, net, ln, personIdx=0):
  (H, W) = frame.shape[:2]
  results = []

  blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416,416), swapRB=True, crop=False)
  net.setInput(blob)
  layerOutputs = net.forward(ln)

  boxes = []
  centroids = []
  confidences = []

  for output in layerOutputs:
    for detection in output:
      scores = detection[5:]
      classID = np.argmax(scores)
      confidence = scores[classID]

      if classID==personIdx and confidence>social_dist_config.MIN_CONF:
        box = detection[0:4] * np.array([W,H,W,H])
        (centerX, centerY, width, height) = box.astype("int")

        x = int(centerX - (width/2))
        y = int(centerY - (height/2))

        boxes.append([x, y, int(width), int(height)])
        centroids.append([centerX, centerY])
        confidences.append(float(confidence))

  idxs = cv2.dnn.NMSBoxes(boxes, confidences, social_dist_config.MIN_CONF, social_dist_config.NMS_THRESH)

  if len(idxs) > 0:
    for i in idxs.flatten():
      (x, y) = (boxes[i][0], boxes[i][1])
      (w, h) = (boxes[i][2], boxes[i][3])
      r = (confidences[i], (x, y, x+w, y+h), centroids[i])
      results.append(r)

  return results





"""# **4. Social Distance detector**"""

input_data = 'D:\SocDistDect\pedestrians.mp4'
output_data = 'D:\SocDistDect\output.avi'
display = 0

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="",
	help="path to (optional) input video file")
ap.add_argument("-o", "--output", type=str, default="",
	help="path to (optional) output video file")
ap.add_argument("-d", "--display", type=int, default=1,
	help="whether or not output frame should be displayed")
args = vars(ap.parse_args(["--input","D:\pedestrians.mp4","--output","D:\SocDistDect\output.avi","--display","1"]))

labelsPath = 'D:\SocDistDect\yolo-coco-data\coco.names'
LABELS = open(labelsPath).read().strip().split("\n")

weightsPath = 'D:\SocDistDect\yolo-coco-data\yolov3.weights'
configPath = 'D:\SocDistDect\yolo-coco-data\yolov3.cfg'

print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

if social_dist_config.USE_GPU:
  print("[INFO] setting preferable backend and target to CUDA...")
  net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
  net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL_FP16)

ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

print("[INFO] accessing video stream...")
vs = cv2.VideoCapture(input_data if input_data else 0)
writer = None

while True:
  (grabbed, frame) = vs.read()
  if not grabbed:
    break

  frame = imutils.resize(frame, width=700)
  results = detect_people(frame, net, ln, personIdx=LABELS.index("person"))

  violate = set()

  if len(results)>=2:
    centroids = np.array([r[2] for r in results])
    D = dist.cdist(centroids, centroids, metric='euclidean')

    for i in range(0, D.shape[0]):
      for j in range(i+1, D.shape[1]):
        if D[i, j] < social_dist_config.MIN_DIST:
          violate.add(i)
          violate.add(j)

  for (i,(prob, bbox, centroid)) in enumerate(results):
    (startX, startY, endX, endY) = bbox
    (cX, cY) = centroid
    colour = (0,255,0)

    if i in violate:
      colour = (0,0,255)

    cv2.rectangle(frame, (startX, startY), (endX, endY), colour, 2)
    cv2.circle(frame, (cX, cY), 5, colour, 1)

    text = "Social Distancing Violations: {}".format(len(violate))
    cv2.putText(frame, text, (10, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3) 

    if args["display"] > 0:
      cv2.imshow("frame",frame)
      key = cv2.waitKey(1) & 0xFF

      if key == ord('q'):
        break

      if args["output"] != "" and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(output_data, fourcc, 25, (frame.shape[1], frame.shape[0]), True)

      if writer is not None:
        writer.write(frame)
print('[INFO] done')


 
def show_video(video_path, video_width = 600):
   
  video_file = open(video_path, "r+b").read()
 
  video_url = f"data:video/mp4;base64,{b64encode(video_file).decode()}"
  return HTML(f"""<video width={video_width} controls><source src="{video_url}"></video>""")
 
show_video(output_data)