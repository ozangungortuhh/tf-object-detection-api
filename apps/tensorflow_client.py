import cv2
import numpy as np
import grpc
from concurrent import futures
import sys,os

from api import object_detection_pb2
from api import object_detection_pb2_grpc

class TFClient():
    def __init__(self):
        self._tf_channel = grpc.insecure_channel('localhost:50051')
        self._tf_stub = object_detection_pb2_grpc.ObjectDetectionStub(self._tf_channel)
        self.cap = cv2.VideoCapture(0)
        print("initialized darknet channel and stub")
    
    def test(self):
        while True:
            ret, cv_img = self.cap.read()
            _ , img_jpg = cv2.imencode('.jpg', cv_img)
            image_msg = object_detection_pb2.Image(data=img_jpg.tostring())
            detection = self._tf_stub.objectDetection(image_msg)
            print("received detection")
            print(detection)

if __name__ == "__main__":
    tf = TFClient()
    tf.test()