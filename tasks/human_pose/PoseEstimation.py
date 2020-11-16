import PIL.Image
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects
import argparse
import os.path
import math
import numpy as np

class PoseConfig():

    NAMES = ["head", "leftShoulder", "rightShoulder", "leftElbow", "rightElbow", "leftWrist", "rightWrist", "leftHip",
             "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle"]

    HEAD, L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 0, 1, 2, 3, 4, 5, 6
    L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 7, 8, 9, 10, 11, 12



class PoseEstima:
    def get_keypoint(self,humans, hnum, peaks):
        # check invalid human index
        kpoint = []
        human = humans[0][hnum]
        C = human.shape[0]
        global peak5
        global peak7
        global peak9
        for j in range(C):
            k = int(human[j])
            if k >= 0:
                peak = peaks[0][j][k]  # peak[1]:width, peak[0]:height
                peak = (j, float(peak[0]), float(peak[1]))
                kpoint.append(peak)
                print('index:%d : success [%5.3f, %5.3f]' % (j, peak[1], peak[2]))
            else:
                peak = (j, None, None)
                kpoint.append(peak)
                print('index:%d : None %d' % (j, k))

            while j == 5:
                if k >= 0:
                    peak5 = peaks[0][j][k]  # peak[1]:width, peak[0]:height
                    peak5 = (j, float(peak[1]), float(peak[2]))
                    break
                else:
                    peak5 = (j, 0, 0)
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
        return kpoint

    def preprocess(self,image):
        global device
        device = torch.device('cuda')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = PIL.Image.fromarray(image)
        image = transforms.functional.to_tensor(image).to(device)
        image.sub_(mean[:, None, None]).div_(std[:, None, None])
        return image[None, ...]

    def execute(self,img, src, t):
        color = (0, 255, 0)
        data = preprocess(img)
        cmap, paf = model_trt(data)
        cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
        counts, objects, peaks = parse_objects(cmap, paf)  # , cmap_threshold=0.15, link_threshold=0.15)
        fps = 1.0 / (time.time() - t)
        for i in range(counts[0]):
            keypoints = get_keypoint(objects, i, peaks)
            for j in range(len(keypoints)):
                if keypoints[j][1]:
                    x = round(keypoints[j][2] * WIDTH * X_compress)
                    y = round(keypoints[j][1] * HEIGHT * Y_compress)
                    cv2.circle(src, (x, y), 3, color, 2)
                    cv2.putText(src, "%d" % int(keypoints[j][0]), (x + 5, y), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 255), 1)
                    cv2.circle(src, (x, y), 3, color, 2)
            if (peak5[1] != 0 and peak7[1] != 0 and peak9[1] != 0):
                a = (peak7[1] - peak5[1]) ** 2 + (peak7[2] - peak5[2]) ** 2
                b = (peak7[1] - peak9[1]) ** 2 + (peak7[2] - peak9[2]) ** 2
                c = (peak9[1] - peak5[1]) ** 2 + (peak9[2] - peak5[2]) ** 2
                leftelbow_angle = math.acos((a + b - c) / math.sqrt(4 * a * b)) * (180 / math.pi)
                cv2.putText(src, "angle: %f" % (leftelbow_angle), (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
                break
            else:
                # print("Not detect Leftelbow Angle")
                break
        print("FPS:%f " % (fps))
        # draw_objects(img, counts, objects, peaks)

        cv2.putText(src, "FPS: %f" % (fps), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

        out_video.write(src)
        return src
