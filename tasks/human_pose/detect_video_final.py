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
'''
hnum: 0 based human index
kpoint : keypoints (float type range : 0.0 ~ 1.0 ==> later multiply by image width, height
'''

parser = argparse.ArgumentParser(description='TensorRT pose estimation run')
parser.add_argument('--model', type=str, default='resnet', help='resnet or densenet')
args = parser.parse_args()
device = torch.device('cuda')

class human_detection:

    def __init__(self):
        print('------ model = resnet--------')
        self.MODEL_WEIGHTS = 'resnet18_baseline_att_224x224_A_epoch_249.pth'
        self.OPTIMIZED_MODEL = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
        with open('human_pose.json', 'r') as f:
            human_pose = json.load(f)
        num_parts = len(human_pose['keypoints'])
        num_links = len(human_pose['skeleton'])
        self.model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
        self.WIDTH = 224
        self.HEIGHT = 224
        data = torch.zeros((1, 3, self.HEIGHT, self.WIDTH)).cuda()
        if os.path.exists(self.OPTIMIZED_MODEL) == False:
            self.model.load_state_dict(torch.load(self.MODEL_WEIGHTS))
            self.model_trt = torch2trt.torch2trt(self.model, [data], fp16_mode=True, max_workspace_size=1 << 25)
            torch.save(self.model_trt.state_dict(), self.OPTIMIZED_MODEL)
        self.topology = trt_pose.coco.coco_category_to_topology(human_pose)
        self.model_trt = TRTModule()
        self.model_trt.load_state_dict(torch.load(self.OPTIMIZED_MODEL))
        self.parse_objects = ParseObjects(self.topology)
        self.draw_objects = DrawObjects(self.topology)

    def get_keypoint(self,humans, hnum, peaks):
        # check invalid human index
        kpoint = []
        human = humans[0][hnum]
        C = human.shape[0]
        global peak1, peak5, peak6, peak7,peak9,peak10
        for j in range(C):
            k = int(human[j])
            if k >= 0:
                peak = peaks[0][j][k]  # peak[1]:width, peak[0]:height
                peak = (j, float(peak[0]), float(peak[1]))
                kpoint.append(peak)
                #print('index:%d : success [%5.3f, %5.3f]' % (j, peak[1], peak[2]))
            else:
                peak = (j, None, None)
                kpoint.append(peak)
                #print('index:%d : None %d' % (j, k))

            while j == 1:
                if k >= 0:
                    peak1 = peaks[0][j][k]  # peak[1]:width, peak[0]:height
                    peak1 = (j, float(peak[1]), float(peak[2]))
                    break
                else:
                    peak5 = (j, 0, 0)
                    break
            while j == 5:
                if k >= 0:
                    peak5 = peaks[0][j][k]  # peak[1]:width, peak[0]:height
                    peak5 = (j, float(peak[1]), float(peak[2]))
                    break
                else:
                    peak5 = (j, 0, 0)
                    break

            while j == 6:
                if k >= 0:
                    peak6 = peaks[0][j][k]  # peak[1]:width, peak[0]:height
                    peak6 = (j, float(peak[1]), float(peak[2]))
                    break
                else:
                    peak6 = (j, 0, 0)
                    break

            while j == 7:
                if k >= 0:
                    peak7 = peaks[0][j][k]  # peak[1]:width, peak[0]:height
                    peak7 = (j, float(peak[1]), float(peak[2]))
                    break
                else:
                    peak7 = (j, 0, 0)
                    break

            while j == 9:
                if k >= 0:
                    peak9 = peaks[0][j][k]  # peak[1]:width, peak[0]:height
                    peak9 = (j, float(peak[1]), float(peak[2]))
                    break
                else:
                    peak9 = (j, 0, 0)
                    break

            while j == 10:
                if k >= 0:
                    peak10 = peaks[0][j][k]  # peak[1]:width, peak[0]:height
                    peak10 = (j, float(peak[1]), float(peak[2]))
                    break
                else:
                    peak10 = (j, 0, 0)
                    break
        return kpoint


    # t0 = time.time()
    # torch.cuda.current_stream().synchronize()
    # for i in range(50):
    #     y = model_trt(data)
    # torch.cuda.current_stream().synchronize()
    # t1 = time.time()
    #
    # print(50.0 / (t1 - t0))

    def preprocess(self,image):
        mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
        std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
        device = torch.device('cuda')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = PIL.Image.fromarray(image)
        image = transforms.functional.to_tensor(image).to(device)
        image.sub_(mean[:, None, None]).div_(std[:, None, None])
        return image[None, ...]


    def visualize(self,img, src, t):
        color = (0, 255, 0)
        data = self.preprocess(img)
        cmap, paf = self.model_trt(data)
        cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
        counts, objects, peaks = self.parse_objects(cmap, paf)  # , cmap_threshold=0.15, link_threshold=0.15)
        X_compress = 640.0 / self.WIDTH * 1.0
        Y_compress = 480.0 / self.HEIGHT * 1.0
        fps = 1.0 / (time.time() - t)
        for i in range(counts[0]):
            keypoints = self.get_keypoint(objects, i, peaks)
            for j in range(len(keypoints)):
                if keypoints[j][1]:
                    x = round(keypoints[j][2] * self.WIDTH * X_compress)
                    y = round(keypoints[j][1] * self.HEIGHT * Y_compress)
                    cv2.circle(src, (x, y), 3, color, 2)
                    cv2.putText(src, "%d" % int(keypoints[j][0]), (x + 5, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)
                    cv2.circle(src, (x, y), 3, color, 2)
            # pose suggestion
            if peak9[1] != 0 and peak10[1] != 0 :
                if (peak9[1] < peak1[1]) and (peak10[1] < peak1[1]) :
                    cv2.putText(src, "Tree pose", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
                    if math.isclose(peak10[2],peak9[2],rel_tol=0.05,abs_tol=0) == True :
                        cv2.putText(src, "Pose is great!", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
                    else:
                        cv2.putText(src, "You should return more", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
                elif(peak9[1] < peak1[1]) or (peak10[1] < peak1[1]) :
                    cv2.putText(src, "Triangle pose", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
                    if math.isclose(peak5[2],peak9[2],rel_tol=0.05,abs_tol=0) == True or math.isclose(peak6[2],peak10[2],rel_tol=0.1,abs_tol=0) == True :
                        cv2.putText(src, "Pose is great!", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
                    else:
                        cv2.putText(src, "You should return more", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
                    break

            # while peak5[1] != 0 and peak7[1] != 0 and peak9[1] != 0:
            #     leftelbow_angle = self.angle
            #     print("!!!!!!!!!!!!!!!11", leftelbow_angle)
            #     # cv2.putText(src, str("angle: %f" % leftelbow_angle), (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
        return src

    def angle(self):
        if peak5[1] != 0 and peak7[1] != 0 and peak9[1] != 0:
            a = (peak7[1] - peak5[1]) ** 2 + (peak7[2] - peak5[2]) ** 2
            b = (peak7[1] - peak9[1]) ** 2 + (peak7[2] - peak9[2]) ** 2
            c = (peak9[1] - peak5[1]) ** 2 + (peak9[2] - peak5[2]) ** 2
            leftelbow_angle = math.acos((a + b - c) / math.sqrt(4 * a * b)) * (180 / math.pi)

            return leftelbow_angle
        else:
            return 0


    def run(self):
        source_path = 0
        cap = cv2.VideoCapture(source_path)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        out_video = cv2.VideoWriter('output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (640, 480))
        if cap is None:
            print("Camera Open Error")
            sys.exit(0)


        while cap.isOpened():
            t = time.time()
            ret_val, dst = cap.read()
            if ret_val == False:
                print("Camera read Error")
                break
            img = cv2.resize(dst, dsize=(self.WIDTH, self.HEIGHT), interpolation=cv2.INTER_AREA)
            img = self.visualize(img, dst, t)
            cv2.imshow("result", img)
            out_video.write(img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        out_video.release()
        cap.release()

human = human_detection()
human.run()