# Python
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import zipfile
import cv2
import time
from absl import app, flags, logging

# Tensorflow
import tensorflow as tf
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# gRPC
import grpc
from concurrent import futures
from api import object_detection_pb2
from api import object_detection_pb2_grpc

# Model
PATH_TO_CKPT = './sample_models/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90

ONE_DAY_IN_SECONDS = 60 * 60 * 24

# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

class ObjectDetectionServicer(object_detection_pb2_grpc.ObjectDetectionServicer):
    def __init__(self):
        print("Initializied detection service for tensorflow")

    def objectDetection(self, request, data):
        # Receive img from gRPC client
        np_img = np.fromstring(request.data, np.uint8)
        cv_img = cv2.imdecode(np_img, cv2.COLOR_BGR2RGB)
        print("Received the image with shape: ", cv_img.shape)
        objects = []
        
        with detection_graph.as_default():
            with tf.compat.v1.Session(graph=detection_graph) as sess:

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(cv_img, axis=0)
                # Extract image tensor
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                # Extract detection boxes
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                # Extract detection scores
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                # Extract detection classes
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                # Extract number of detectionsd
                num_detections = detection_graph.get_tensor_by_name(
                    'num_detections:0')
                # Actual detection.
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})    
        
        print("Returning detections")
        return object_detection_pb2.Detection(objects=objects)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    object_detection_pb2_grpc.add_ObjectDetectionServicer_to_server(ObjectDetectionServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    try:
        while True:
            time.sleep(ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)

def main(_argv):
    serve()

if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass