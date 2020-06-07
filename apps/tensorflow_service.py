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

class ObjectDetectionServicer(object_detection_pb2_grpc.ObjectDetectionServicer):
    def __init__(self):
        print("Tensorflow Object Detection service is initialized")
    
    def objectDetection(self, request, context):
        np_img = np.fromstring(request.data, np.uint8)
        cv_img = cv2.imdecode(np_img, cv2.COLOR_BGR2RGB)
        print("Received image")


        cv2.imshow('window', cv_img)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
        
        objects = []
        obj = object_detection_pb2.Object(
                label='test',
                probability=0,
                xmin=0,
                ymin=0,
                xmax=0,
                ymax=0,
            )            
        objects.append(obj)

        print("Sending detections")
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
