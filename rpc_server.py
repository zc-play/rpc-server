# conding:utf8
from rpc.face_rec_sever import serve, predict_face, knn_clf


if __name__ == '__main__':
    serve()
    # predict_face('/data/dataset/vgg_face/multi_face/img-mf-6591.jpg', knn_clf, method='face-net')

