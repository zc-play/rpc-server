# coding: utf-8
import base64
from io import BytesIO

import grpc
from PIL import Image

from rpc.face_rec_pb2_grpc import FaceRecognitionStub
from rpc.face_rec_pb2 import MyStr

channel = grpc.insecure_channel('localhost:50051')
stub = FaceRecognitionStub(channel)


def face_rec_rpc(face_id, is_return=True, is_save=False, save_path='test.png'):
    """
    rpc client, send a request of face recognition to rpc server
    :param face_id: str,
    :param is_return: return base64 of the recognition face
    :param is_save: the flag if save image
    :param save_path: save path
    :return:
    """
    response = stub.face_rec_str(MyStr(name='%s' % face_id))
    if is_return:
        return 'data:image/gif;base64,%s' % response.stream
    if is_save:
        img_bytes = base64.b64decode(response.stream)
        img_fp = BytesIO(img_bytes)
        img = Image.open(img_fp)
        img.save(save_path)


if __name__ == '__main__':
    print(face_rec_rpc('1'))
