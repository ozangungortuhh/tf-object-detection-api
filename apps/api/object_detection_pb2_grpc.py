# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

import object_detection_pb2 as object__detection__pb2


class ObjectDetectionStub(object):
  # missing associated documentation comment in .proto file
  pass

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.objectDetection = channel.unary_unary(
        '/object_detection.ObjectDetection/objectDetection',
        request_serializer=object__detection__pb2.Image.SerializeToString,
        response_deserializer=object__detection__pb2.Detection.FromString,
        )


class ObjectDetectionServicer(object):
  # missing associated documentation comment in .proto file
  pass

  def objectDetection(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_ObjectDetectionServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'objectDetection': grpc.unary_unary_rpc_method_handler(
          servicer.objectDetection,
          request_deserializer=object__detection__pb2.Image.FromString,
          response_serializer=object__detection__pb2.Detection.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'object_detection.ObjectDetection', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))
