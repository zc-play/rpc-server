# coding: utf-8
import base64

from sqlalchemy import Column, String, INTEGER, SmallInteger
from web import db

from rpc.face_rec_client import face_rec_rpc


class Face(db.Model):

    id = Column(INTEGER(), primary_key=True, autoincrement=True)
    name = Column(String(256))
    path = Column(String(1024))
    is_train = Column(SmallInteger)
    encoding = Column(String(2048))

    def __init__(self, name, path, is_train=False):
        self.name = name
        self.path = path
        self.is_train = is_train
        self.data = None

    def save(self):
        db.session.add(self)
        db.session.commit()

    def get_org_data(self):
        with open(self.path, 'rb') as f:
            img = f.read()
        img_b64 = base64.b64encode(img)
        self.data = 'data:image/gif;base64,%s' % str(img_b64,'utf-8')

    def get_rec_data(self):
        self.data = face_rec_rpc(str(self.id))

    def record_train(self):
        self.is_train = True
        self.save()


