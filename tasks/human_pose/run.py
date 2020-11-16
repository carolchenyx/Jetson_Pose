import json
import trt_pose.coco
import trt_pose.models
import torch
import torch2trt
from torch2trt import TRTModule
import time, sys
import cv2
import torchvision.transforms as transforms
import PIL.Image
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects
import argparse
import os.path
import math
import numpy as np
from tasks.human_pose.PoseEstimation import PoseEstima

PE = PoseEstima()
device = torch.device('cuda')

class run:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='TensorRT pose estimation run')
        self.parser.add_argument('--model', type=str, default='resnet', help='resnet or densenet')
        self.args = parser.parse_args()
        with open('human_pose.json', 'r') as f:
            human_pose = json.load(f)
        self.topology = trt_pose.coco.coco_category_to_topology(human_pose)
        num_parts = len(human_pose['keypoints'])
        num_links = len(human_pose['skeleton'])
        if 'resnet' in args.model:
            print('------ model = resnet--------')
            MODEL_WEIGHTS = 'resnet18_baseline_att_224x224_A_epoch_249.pth'
            OPTIMIZED_MODEL = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
            model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
            WIDTH = 224
            HEIGHT = 224

        else:
            print('------ model = densenet--------')
            MODEL_WEIGHTS = 'densenet121_baseline_att_256x256_B_epoch_160.pth'
            OPTIMIZED_MODEL = 'densenet121_baseline_att_256x256_B_epoch_160_trt.pth'
            model = trt_pose.models.densenet121_baseline_att(num_parts, 2 * num_links).cuda().eval()
            WIDTH = 256
            HEIGHT = 256
        data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()
        self.model_trt = TRTModule()
        self.model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))
        mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
        std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def run(self):
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        out_video = cv2.VideoWriter('/tmp/output.mp4', fourcc, self.cap.get(cv2.CAP_PROP_FPS), (640, 480))
        count = 0
        while self.cap.isOpened() and count < 500:
            t = time.time()
            ret_val, dst = self.cap.read()
            parse_objects = ParseObjects(topology)
            draw_objects = DrawObjects(topology)
            if ret_val == False:
                print("Camera read Error")
                break
            img = cv2.resize(dst, dsize=(WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
            img = PE.execute(img, dst, t)
            cv2.imshow("result", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            count += 1
        cv2.destroyAllWindows()
        out_video.release()
        cap.release()