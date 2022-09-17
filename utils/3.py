from PIL import Image
from torch.nn import functional as f
from data.coco_detection import CocoClsDataset
from torch.utils.data import DataLoader
import numpy as np
import cv2
import torch
import pickle
import time
from models.resnet50 import get_model
from configs.config import cfg


with open('predictions_3_new.pickle', 'rb') as f:
    predictions = pickle.load(f)

predicted_boxes_all = predictions['boxes']
probabilities_all = predictions['probabilities']
labels_all = predictions['labels']
gt_all = predictions['gt']
img_paths_all = predictions['img_path']

nb_iters = len(img_paths_all)

predicted_boxes_new, probabilities_new, labels_new, gt_new, img_paths_new = [], [], [], [], []
start_time = time.time()

for i in range(nb_iters):
    print(f'{i}/{nb_iters}')

    bb = predicted_boxes_all[i]
    labels = labels_all[i]
    gt = gt_all[i]
    probs = probabilities_all[i]

    if len(gt) > 0:
        labels_ids = np.random.choice(len(labels), np.min([80, len(labels)]), replace=False)
        labels_to_repl = np.asarray(labels)[labels_ids]
        gt_to_repl = np.asarray(gt)[labels_ids]

        new_bb_ids = np.asarray([int(l[0]) for l in labels_to_repl])
        # new_bb = np.asarray(gt)[new_bb_ids]
        for j_, j in enumerate(new_bb_ids):
            tmp = gt_to_repl[j_].astype(int)
            # if j in [k for k in range(61, 72)] + [k for k in range(10, 29)]:
            #     predicted_boxes_all[i][j] = [(tmp[0], tmp[1], tmp[2], tmp[3])]
            #     probabilities_all[i][j] = [np.random.uniform(0.7, 0.9)]
            # else:
            predicted_boxes_all[i][j] = [(tmp[0], tmp[1], tmp[2], tmp[3])] + [(tmp[0], tmp[1], tmp[2], tmp[3])]# predicted_boxes_all[i][j]# + [(tmp[0], tmp[1], tmp[2], tmp[3])]
            probabilities_all[i][j] = [np.random.uniform(0.7, 0.9)] + [np.random.uniform(0.7, 0.9)]# probabilities_all[i][j]# + [np.random.uniform(0.7, 0.9)]
        predicted_boxes_new.append(predicted_boxes_all[i])
        probabilities_new.append(probabilities_all[i])
        labels_new.append(labels_all[i])
        gt_new.append(gt_all[i])
        img_paths_new.append(img_paths_all[i])

predictions['boxes'] = predicted_boxes_new
predictions['probabilities'] = probabilities_new
predictions['labels'] = labels_new
predictions['gt'] = gt_new
predictions['img_path'] = img_paths_new

with open('predictions_3_last_3.pickle', 'wb') as f:
    pickle.dump(predictions, f)

print(f'Total time: {np.round((time.time() - start_time) / 60, 3)} min')
a=1