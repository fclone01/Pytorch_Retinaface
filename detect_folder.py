from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
# from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from utils.timer import Timer
import matplotlib.pyplot as plt

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

torch.set_grad_enabled(False)
cfg = {
    'name': 'mobilenet0.25',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 190,
    'decay2': 220,
    'image_size': 640,
    'pretrain': False,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64
}

# net and model
net = RetinaFace(cfg=cfg, phase = 'test')
net = load_model(net, "weights/mobilenet0.25_Final.pth", True)
net.eval()
print('Finished loading model!')
print(net)
cudnn.benchmark = True
device = torch.device("cpu" if True else "cuda")
net = net.to(device)

def detect_faces(img_raw,vis_thres=0.5):
    if(type(img_raw)==str):
        img_raw = cv2.imread(img_raw, cv2.IMREAD_COLOR)
    img = np.float32(img_raw)
    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)
    _t = {'forward_pass': Timer(), 'misc': Timer()}
    resize = 1

    _t['forward_pass'].tic()
    loc, conf, landms = net(img)  # forward pass
    _t['forward_pass'].toc()
    _t['misc'].tic()
    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                            img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                            img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)
    landms = landms * scale1 / resize
    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > 0.02)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    # order = scores.argsort()[::-1][:args.top_k]
    order = scores.argsort()[::-1]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, 0.4)

    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    # dets = dets[:args.keep_top_k, :]
    # landms = landms[:args.keep_top_k, :]

    dets = np.concatenate((dets, landms), axis=1)
    facess  = []
    for b in dets:
        if b[4] < vis_thres:
            continue
        xs = b[4]
        b = list(map(int, b))
        b.append(xs)
        facess.append(b)

    return facess

file_log = {
    "0_face":[],
    "muilty_face":[],
    "error_crop":[]
}

def write_log_to_file():
    for key, values in file_log.items():
        # Đặt tên tệp tin dựa trên khóa
        file_name = f"log/{key}.txt"
        
        # Ghi dữ liệu vào tệp tin
        with open(file_name, 'w') as file:
            for value in values:
                file.write(value + '\n') 


#Detect folder
import math
def calculate_angle(point1, point2):
    # print(point1,point2)
    # Tính độ dốc (slope)
    dy = point2[1] - point1[1]
    dx = point2[0] - point1[0]
    
    # Tránh chia cho 0
    if dx == 0:
        return 90 if dy > 0 else 270  # 90 độ nếu đi lên, 270 độ nếu đi xuống
    
    slope = dy / dx
    
    # Tính góc (tính bằng radian và chuyển sang độ)
    angle_rad = math.atan(slope)
    angle_deg = math.degrees(angle_rad)
    
    # # Điều chỉnh góc về miền từ 0 đến 360 độ
    # if angle_deg < 0:
    #     angle_deg += 360
    
    return angle_deg
from PIL import Image

import math
import matplotlib.pyplot as plt

def rotate_rectangle(points, angle):
    # Tính tọa độ trung tâm
    center_x = sum(x for x, y in points) / len(points)
    center_y = sum(y for x, y in points) / len(points)

    # Chuyển đổi góc từ độ sang radian
    radian = math.radians(angle)
    
    # Ma trận xoay
    cos_angle = math.cos(radian)
    sin_angle = math.sin(radian)
    
    rotated_points = []
    
    for (x, y) in points:
        # Di chuyển điểm về gốc tọa độ
        x -= center_x
        y -= center_y
        
        # Tính tọa độ mới
        x_new = x * cos_angle - y * sin_angle
        y_new = x * sin_angle + y * cos_angle
        
        # Di chuyển điểm trở lại vị trí cũ
        x_new += center_x
        y_new += center_y
        
        rotated_points.append((x_new, y_new))
    
    return rotated_points

def cut_rotated_rectangle(image_path, points):
    # Mở ảnh
    image = cv2.imread(image_path)

    # Chuyển đổi danh sách điểm thành mảng NumPy
    points = np.array(points, dtype='float32')

    # Xác định tọa độ cho hình chữ nhật mới
    width = int(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])))
    height = int(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])))

    # Điểm góc cho hình chữ nhật đã cắt
    dest_points = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype='float32')

    # Tính ma trận biến đổi
    M = cv2.getPerspectiveTransform(points, dest_points)

    # Cắt hình
    cropped_image = cv2.warpPerspective(image, M, (width, height))

    return cropped_image


def find_largest_area_index(pairs):
    max_area = 0
    largest_index = -1
    
    for index, pair in enumerate(pairs):
        x1, y1, x2, y2 = pair[0:4]
        area = abs(x2 - x1) * abs(y2 - y1)
        
        if area > max_area:
            max_area = area
            largest_index = index
            
    return largest_index

def saveFaceStraight(image_path,out_image_path):
    face  = detect_faces(image_path, vis_thres=0.8)
    if(len(face)==0): file_log["0_face"].append(image_path)
    elif(len(face)>1): 
        file_log["muilty_face"].append(image_path)
        face = [face[find_largest_area_index(face)]]
    if(len(face)==1):
        try:
            face = face[0]
            angel_ = (calculate_angle([face[5],face[6]],[face[7],face[8]])+calculate_angle([face[11],face[12]],[face[13],face[14]]))/2
            x1,y1,x2,y2 = face[0:4]
            x_new1 = x1 - (0.04 * (x2 - x1))
            y_new1 = y1 - (0.04 * (y2 - y1))
            x_new2 = x2 + (0.04 * (x2 - x1))
            y_new2 = y2 + (0.04 * (y2 - y1))
            x1,y1,x2,y2 =x_new1,y_new1,x_new2,y_new2
            rectangle_points =[[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
            new_points = rotate_rectangle(rectangle_points, angel_)
            cropped_image = cut_rotated_rectangle(image_path,new_points)
            cv2.imwrite(out_image_path,cropped_image)
        except:
            file_log["error_crop"].append(image_path)
#Detect folder.
FOLDER_IN  = "E:/MY/CASIA_maxpy_clean"
FOLDER_OUT = "E:/MY/OUT_CASIA"

classes = os.listdir(FOLDER_IN)
classes.sort()

index = 0
# file_log = []
for _class in classes:
    if(index%20==0): print(index,_class)
    folder_class = f"{FOLDER_IN}/{_class}"
    if(os.path.exists(f"{FOLDER_OUT}/{_class}") is False): os.makedirs(f"{FOLDER_OUT}/{_class}")
    for img_name in os.listdir(folder_class):
        saveFaceStraight(folder_class+f"/{img_name}",f"{FOLDER_OUT}/{_class}/{img_name}")
    index+=1
    if(index%100==0): 
        print("write log")
        write_log_to_file()
