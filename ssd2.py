from imutils.video import FileVideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
from math import floor
from scipy.spatial import distance as dist

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading SSD from disk...")
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
#vs = VideoStream(src=0).start()
vs = FileVideoStream("name").start()
time.sleep(2.0)
fps = FPS().start()
init_once = []
oklist = []
tracker_lst = []
newbox = []
(h, w) = (None, None)
# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame  = vs.read()
    frame = imutils.resize(frame, width=900,height = 600)
    results = []
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    bounding_boxes = []
    centroids = []
    confidences = []
    boxes =[]
    for i in np.arange(0,detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence > 0.2:
            idx = int(detections[0,0,i,1])
            if CLASSES[idx] == 'person':
                box = detections[0,0,i,3:7]*np.array([w,h,w,h])
                bounding_boxes.append(box)
                label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                (startX, startY, endX, endY) = box.astype(int)
                boxes.append([int(startX),int(startY), int(endX), int(endY)])
                x = int(abs(endX + startX)/2)
                y = int(abs(endY + startY)/2)
                #print(' x:' + str(x))
                #print(' y:' + str(y))
                centroids.append((x,y))
                confidences.append(float(confidence))
                
###########################################################

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.2,
		0.5)
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            r = (confidences[i], (x, y, w, h), centroids[i])
            results.append(r)
            print('result ' + str(i) + str(r))

    violate = set()
    if len(results) >= 2:
        centroids = np.array([r[2] for r in results])
        D = dist.cdist(centroids, centroids, metric="euclidean")
        for i in range(0, D.shape[0]):
            for j in range(i + 1, D.shape[1]):
                if D[i, j] < 100:
                    violate.add(i)
                    violate.add(j)



    for (i, (prob, bbox, centroid)) in enumerate(results):
        (startX, startY, endX, endY) = bbox
        (cX, cY) = centroid
        color = (0, 255, 0)
        if i in violate:
            color = (0, 0, 255)

        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv2.circle(frame, (cX, cY), 5, color, 1)
		
        #cv2.rectangle(frame, (startX, startY), (endX, endY),COLORS[idx], 2)
        #y = startY - 15 if startY - 15 > 15 else startY + 15
        #cv2.putText(frame, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

        # draw the total number of social distancing violations on the
	# output frame
        text = "Social Distancing Violations: {}".format(len(violate))
        cv2.putText(frame, text, (10, frame.shape[0] - 25),
		cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)

######################################################
    
    # show the output frame
    cv2.imshow("Frame", frame)
    fps.update()
    #for i in range(50000000):
    #    pass
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    # update the FPS counte

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.release()
