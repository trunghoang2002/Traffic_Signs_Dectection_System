import torch as t
from utils.config import opt
from model import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from utils.vis_tool import view_img
from utils import array_tool as at
import cv2

faster_rcnn = FasterRCNNVGG16()
# faster_rcnn.score_thresh = 0.6
trainer = FasterRCNNTrainer(faster_rcnn)

# trainer.load('fasterrcnn_12211511_0.701052458187_torchvision_pretrain.pth')
trainer.load('weights/5epoch.pth')
opt.caffe_pretrain=False # this model was trained from torchvision-pretrained model

frame = cv2.imread('img_traffic.jpg')
frame = cv2.resize(frame, (640, 480))
frame = frame.transpose((2, 0, 1))
frame = t.from_numpy(frame)[None]
# print(frame.shape)
_bboxes, _labels, _scores = trainer.faster_rcnn.predict(frame,visualize=True)
frame = view_img(at.tonumpy(frame[0]),
        at.tonumpy(_bboxes[0]),
        at.tonumpy(_labels[0]).reshape(-1),
        at.tonumpy(_scores[0]).reshape(-1))
cv2.imshow('frame', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()