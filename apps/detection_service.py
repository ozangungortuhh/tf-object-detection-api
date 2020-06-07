import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import grpc
from concurrent import futures

from api import object_detection_pb2
from api import object_detection_pb2_grpc

PATH_TO_CKPT = './sample_models/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90


class ObjectDetectionServicer(object_detection_pb2_grpc.ObjectDetectionServicer):
    def __init__(self):
        # Load the frozen graph into memory
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            self.od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                self.serialized_graph = fid.read()
                self.od_graph_def.ParseFromString(self.serialized_graph)
                tf.import_graph_def(self.od_graph_def, name='')
                
        # Load label map
        self.label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)
        
        print("Tensorflow Object Detection service is initialized")
        
    def load_image_into_numpy_array(self, image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)    
        
    def objectDetection(self, request, context):
        np_img = np.fromstring(request.data, np.uint8)
        cv_img = cv2.imdecode(np_img, cv2.COLOR_BGR2RGB)
        print("Received image")

        #with self.detection_graph.as_default():
        #    with tf.compat.v1.Session(graph=self.detection_graph) as sess:
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        # image_np_expanded = np.expand_dims(cv_img, axis=0)
        # # Extract image tensor
        # image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # # Extract detection boxes
        # boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # # Extract detection scores
        # scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        # # Extract detection classes
        # classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        # # Extract number of detectionsd
        # num_detections = self.detection_graph.get_tensor_by_name(
        #     'num_detections:0')
        
        # # Actual detection.
        # (boxes, scores, classes, num_detections) = sess.run(
        #     [boxes, scores, classes, num_detections],
        #     feed_dict={image_tensor: image_np_expanded})                

        #             # Visualization of the results of a detection.
        # vis_util.visualize_boxes_and_labels_on_image_array(
        #     cv_img,
        #     np.squeeze(boxes),
        #     np.squeeze(classes).astype(np.int32),
        #     np.squeeze(scores),
        #     self.category_index,
        #     use_normalized_coordinates=True,
        #     line_thickness=8)

        # Display output
        cv2.imshow('object detection', cv_img)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            
        objects = []
        return object_detection_pb2.Detection(objects=objects)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    object_detection_pb2_grpc.add_ObjectDetectionServicer_to_server(ObjectDetectionServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    try:
        while True:
            continue
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == "__main__":
    serve()
