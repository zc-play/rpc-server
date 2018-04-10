# coding: utf-8
import base64
import pickle
import time
import json
from concurrent import futures
from io import BytesIO

import grpc

from config import MODEL_PATH
from core.face_rec import predict_face, draw_labels_and_save
from rpc.face_rec_pb2 import Image as RpcImage
from rpc.face_rec_pb2_grpc import FaceRecognitionServicer, add_FaceRecognitionServicer_to_server
from web import app, db
from web.model import Face

_ONE_DAY_IN_SECONDS = 60 * 60 * 24


with open(MODEL_PATH, 'rb') as f:
    knn_clf = pickle.load(f)


class FaceRec(FaceRecognitionServicer):

    def face_rec_str(self, request, context):
        data = request.name
        data = json.loads(data)
        face_id, method = data['face_id'], data['method']
        face_id = int(face_id)
        face_path = FaceRec.get_face_path(face_id)
        if face_path is None:
            return None
        img, locs = predict_face(face_path, knn_clf, distance_threshold=0.6, method=method)
        img = draw_labels_and_save(img, locs, is_save=False)
        img_str = BytesIO()
        img.save(img_str, 'png')
        rpc_img = RpcImage(id=0, stream=base64.b64encode(img_str.getvalue()))
        return rpc_img

    @staticmethod
    def get_face_path(face_id):
        with app.app_context():
            face = db.session.query(Face.id, Face.path).filter(Face.is_train == 0, Face.id == face_id).first()
            return face.path


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=5))
    add_FaceRecognitionServicer_to_server(FaceRec(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    # serve()
    face_path = '/data/dataset/lfw_test/Lucio_Gutierrez/Lucio_Gutierrez-6.jpg'
    img, locs = predict_face(face_path, knn_clf, distance_threshold=0.6, method='dlib')
    print(locs)
