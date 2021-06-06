from absl import flags
import sys

FLAGS = flags.FLAGS
FLAGS(sys.argv)
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

import tensorflow as tf
from yolo.models import YoloV3
from yolo.dataset import transform_images
from yolo.utils import convert_boxes
from _collections import deque
from deep_sort_algo import preprocess
from deep_sort_algo import nn_match
from deep_sort_algo.detection import Detection
from deep_sort_algo.tracker import Tracker
from generator import gendect as gdet

class_names = [c.strip() for c in open('./data/label/coco.names').readlines()]
yolo = YoloV3(classes=len(class_names))
yolo.load_weights('./weights_generated/yolov3.tf')

max_cosine_distance = 0.5
nn_budget = None
nms_max_overlap = 0.8

model_filename = 'model/temp.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = nn_match.NearestNeighborDistanceMetric('cosine', max_cosine_distance, nn_budget)
tracker = Tracker(metric)

video = cv2.VideoCapture('./data/input/testvideo.mp4')

codec = cv2.VideoWriter_fourcc(*'XVID')
video_fps = int(video.get(cv2.CAP_PROP_FPS))
video_width, video_height = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('./data/output/result-test.avi', codec, video_fps, (video_width, video_height))

tx = [deque(maxlen=30) for _ in range(1000)]
old_x = [0 for _ in range(1000)]

counter = []
lr = []

while True:
    _, img = video.read()
    if img is None:
        print('Completed')
        break

    img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_in = tf.expand_dims(img_in, 0)
    img_in = transform_images(img_in, 416)

    t1 = time.time()

    boxes, scores, classes, nums = yolo.predict(img_in)

    classes = classes[0]
    names = []
    for i in range(len(classes)):
        names.append(class_names[int(classes[i])])
    names = np.array(names)
    converted_boxes = convert_boxes(img, boxes[0])
    features = encoder(img, converted_boxes)

    detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                  zip(converted_boxes, scores[0], names, features)]

    boxs = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    classes = np.array([d.class_name for d in detections])
    indices = preprocess.non_max_suppression(boxs, classes, nms_max_overlap, scores)
    detections = [detections[i] for i in indices]

    tracker.predict()
    tracker.update(detections)

    cx = plt.get_cx('tab20b')
    colors = [cx(i)[:3] for i in np.linspace(0, 1, 20)]

    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        bbox = track.to_tlbr()
        class_name = track.get_class()
        if class_name != 'person':
            continue
        color = colors[int(track.track_id) % len(colors)]
        color = [i * 255 for i in color]

        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1] - 30)), (int(bbox[0]) + (len(class_name)
                                                                               + len(str(track.track_id))) * 17,
                                                               int(bbox[1])), color, -1)
        cv2.putText(img, class_name + "-" + str(track.track_id), (int(bbox[0]), int(bbox[1] - 10)), 0, 0.75,
                    (255, 255, 255), 2)

        center = (int(((bbox[0]) + (bbox[2])) / 2), int(((bbox[1]) + (bbox[3])) / 2))
        tx[track.track_id].append(center)

        for j in range(1, len(tx[track.track_id])):
            if tx[track.track_id][j - 1] is None or tx[track.track_id][j] is None:
                continue
            thickness = int(np.sqrt(64 / float(j + 1)) * 2)
            cv2.line(img, (tx[track.track_id][j - 1]), (tx[track.track_id][j]), color, thickness)
            counter.append(int(track.track_id))

        height, width, _ = img.shape
        cv2.line(img, (int(width / 2), int(0)), (int(width / 2), int(height)), (0, 255, 255), thickness=3)

        center_x = int(((bbox[0]) + (bbox[2])) / 2)
        if old_x[track.track_id] == False:
            old_x[track.track_id] = center_x

        if center_x >= int(width / 2):
            if old_x[track.track_id] < int(width / 2):
                if class_name == 'person':
                    lr.append(int(track.track_id))
        old_x[track.track_id] = center_x

    total_count = len(set(counter))
    lrcount = len(set(lr))
    cv2.putText(img, "L->R " + str(lrcount), (0, 80), 0, 1, (0, 0, 255), 2)
    cv2.putText(img, "Total Count: " + str(total_count), (0, 130), 0, 1, (0, 0, 255), 2)

    fps = 1. / (time.time() - t1)
    cv2.putText(img, "FPS: {:.2f}".format(fps), (0, 30), 0, 1, (0, 0, 255), 2)
    cv2.resizeWindow('output', 1024, 768)
    cv2.imshow('output', img)
    out.write(img)

    if cv2.waitKey(1) == ord('q'):
        break
video.release()
out.release()
cv2.destroyAllWindows()
