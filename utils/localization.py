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


def get_valid_dataset():
    """
    Gets valid dataset to compute mAP.
    :return: valid dataloader
    """
    valid_set = CocoClsDataset(cfg['data']['dataset']['root_path'], 'val2017', 'annotations/instances_val2017.json')
    valid_dl = DataLoader(valid_set)
    preprocessing = valid_set.transform
    classes = valid_dl.dataset.labels_str
    return valid_dl, preprocessing, classes


def get_pretrained_model():
    # get pretrained model
    model = get_model(cfg)
    print(f'Trying to load checkpoint from epoch {cfg["CAM"]["epoch_to_load"]}...')
    checkpoint = torch.load(cfg['CAM']['checkpoints_dir'] + f'checkpoint_{cfg["CAM"]["epoch_to_load"]}.pth')
    load_state_dict = checkpoint['model']
    model.load_state_dict(load_state_dict)
    print(f'Successfully loaded checkpoint from epoch {cfg["CAM"]["epoch_to_load"]}.')
    model.eval()

    activations = []

    def save_activation(module, input, output):
        activations.append(output.data.cpu().numpy())

    model._modules.get('layer4').register_forward_hook(save_activation)
    fc = np.squeeze(list(model.parameters())[-2].data.numpy())
    return model, fc, activations


def get_cam_for_image(path, preprocessing, model, fc, activations, save_cam=False, path_to_save=''):
    """
    Gets CAM for single image.
    :return: image cam
    """
    img = Image.open(path).convert('RGB')
    input_tensor = preprocessing(img).unsqueeze(0)
    output = f.softmax(model(input_tensor), dim=1).data.squeeze()
    softmax_outs, labels_idxs = output.sort(0, True)
    softmax_outs, labels_idxs = softmax_outs.numpy(), labels_idxs.numpy()

    sorted_labels_idxs_ids = np.argsort(labels_idxs)
    sorted_labels_idxs = labels_idxs[sorted_labels_idxs_ids]
    sorted_softmax_outs = softmax_outs[sorted_labels_idxs_ids]

    _, channels, h, w = activations[0].shape

    img = cv2.imread(path)
    height, width, _ = img.shape

    cams, heatmaps = [], []
    for idx_ in labels_idxs:
        cam = fc[idx_].dot(activations[0].reshape((channels, h * w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        heatmap = cv2.applyColorMap(cv2.resize(cam_img, (width, height)), cv2.COLORMAP_JET)
        cams.append(cam_img)
        heatmaps.append(heatmap)

    if save_cam:
        main_heatmap = heatmaps[0]
        result = main_heatmap * 0.5 + img * 0.5
        cv2.imwrite(path_to_save, result)
    return heatmaps, img, sorted_softmax_outs


def get_predictions(cams, img, softmax_outs, save_img_with_bb=False):
    """
    формирование предсказаний (боксы, уверенности и номера классов) для картинки
    :return:
    """
    probabilities = [[] for _ in range(len(cams))]  # уверенности
    # classes_ids = np.arange(len(cams))  # номера классов
    boxes = [[] for _ in range(len(cams))]
    for i, cam in enumerate(cams):
        gray_heatmap = np.uint8(255 * cam)
        gray_heatmap = cv2.cvtColor(gray_heatmap, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray_heatmap, 0, 255, cv2.THRESH_OTSU)[1]
        # Find contours
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        bb_img = np.uint8(255 * img.copy())
        r, g, b = cv2.split(bb_img)
        bb_img = cv2.merge([b, g, r])
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(bb_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.rectangle(gray_heatmap, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # boxes[i].append((x, y, x + w, y + h))
            bb_area_on_heatmap = np.asarray(Image.fromarray(gray_heatmap).crop((x, y, x + w - 1, y + h - 1)))
            if bb_area_on_heatmap.shape != () and np.max(bb_area_on_heatmap) != 0:
                bb_area_on_heatmap_normalized = (bb_area_on_heatmap - np.min(bb_area_on_heatmap)) / (np.max(bb_area_on_heatmap) - np.min(bb_area_on_heatmap))
                final_prob = np.mean(bb_area_on_heatmap_normalized) * softmax_outs[i]
                probabilities[i].append(final_prob)
                boxes[i].append((x, y, x + w, y + h))

        if save_img_with_bb:
            cv2.imwrite('bb_img.jpg', bb_img)
            cv2.imwrite('gray_heatmap.jpg', gray_heatmap)

    return boxes, probabilities  # , classes_ids


def get_predicted_boxes(valid_dl, preprocessing, model, fc, activations):
    if cfg['localization']['save_bboxes']:
        # previously saving predicted boxes to pickle
        start_time = time.time()
        info = {'boxes': [], 'probabilities': [], 'labels': [], 'gt': [], 'img_path': []}
        len_dl = len(valid_dl)
        for k, batch in enumerate(valid_dl):

            # if k % 100 == 0:
            print(f'{k}/{len_dl}')

            _, img_file = batch
            labels, gt = valid_dl.dataset.path_to_label_and_bb[img_file[0]]

            cams, img, softmax_outs = get_cam_for_image(img_file[0], preprocessing, model, fc, activations) #, save_cam=True, path_to_save='cam.jpg')
            boxes, probabilities = get_predictions(cams, img, softmax_outs) #, save_img_with_bb=True)
            # match_gt_and_bbs(boxes, probabilities, classes_ids, gt, label)

            info['boxes'].append(boxes)
            info['probabilities'].append(probabilities)
            info['labels'].append(labels)
            info['gt'].append(gt)
            info['img_path'].append(img_file[0])

        with open(cfg['localization']['pickle_path'], 'wb') as f:
            pickle.dump(info, f)

        print(f'Total time: {np.round((time.time() - start_time) / 60, 3)} min')
    else:
        with open(cfg['localization']['pickle_path'], 'rb') as f:
            info = pickle.load(f)
    return info


def get_iou(bb1, bb2):
    x1, y1 = max(bb1[0], bb2[0]), max(bb1[1], bb2[1])
    x2, y2 = min(bb1[2], bb2[2]), min(bb1[3], bb2[3])
    intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    bb1_area = (bb1[2] - bb1[0] + 1) * (bb1[3] - bb1[1] + 1)
    bb2_area = (bb2[2] - bb2[0] + 1) * (bb2[3] - bb2[1] + 1)
    union = float(bb1_area + bb2_area - intersection)
    iou = intersection / union
    return iou


def match_gt_and_bbs(boxes, probabilities, classes_ids, gt, label):
    """
    Matches gt with bbs for single image (for all 80 classes).
    :param boxes: predicted boxes
    :param probabilities:
    :param classes_ids:
    :return:
    """
    # gt_dict = {l: [] for l in range(len(boxes))}
    # for i, l in enumerate(np.unique(label)):
    #     idx = np.where(label == l)[0]
    #     gt_dict[l].extend(np.asarray(gt)[idx])

    nb_classes = len(boxes)
    matched_bbs = [[] for _ in range(nb_classes)]

    for i in range(nb_classes):
        cur_class_boxes = boxes[i]  # берем боксы для данного изображения, для данного класса
        for b_id, bb in enumerate(cur_class_boxes):
            for gt_ in gt:
                iou = get_iou(bb, gt_)
                if iou >= 0.5:
                    matched_bbs[i].append([bb, iou])


def compute_ap(rec, prec):
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def get_mAP(predictions, classes):
    predicted_boxes_all = predictions['boxes']
    probabilities_all = predictions['probabilities']
    labels_all = predictions['labels']
    gt_all = predictions['gt']
    img_paths_all = predictions['img_path']

    nb_iters = len(img_paths_all)
    nb_classes = len(classes)

    predictions_list = []
    gt_dict = []
    for i in range(nb_iters):

        predicted_boxes, probs, classes_ids = predicted_boxes_all[i], probabilities_all[i], np.arange(nb_classes)  # predictions
        labels, gt_boxes, img_filename = labels_all[i], gt_all[i], img_paths_all[i]  # gt

        for cl_id in range(nb_classes):
            predictions_list.append((img_filename, predicted_boxes[cl_id], classes_ids[cl_id], probs[cl_id], gt_boxes, labels))
        gt_dict.append((img_filename, gt_boxes, labels))

    ap_values, pr_curves = [], {}
    for cl_id in range(nb_classes):
        class_str = classes[cl_id]
        cur_class_ids = [k for k in range(len(predictions_list)) if predictions_list[k][2] == cl_id]
        cur_class_img_paths = np.array([predictions_list[k][0] for k in range(len(predictions_list))])[cur_class_ids]
        cur_class_probs = np.array([predictions_list[k][3] for k in range(len(predictions_list))])[cur_class_ids]
        cur_class_bboxes = np.array([predictions_list[k][1] for k in range(len(predictions_list))])[cur_class_ids]

        cur_class_gt_boxes = np.array([predictions_list[k][4] for k in range(len(predictions_list))])[cur_class_ids]
        cur_class_gt_boxes_labels = np.array([predictions_list[k][5] for k in range(len(predictions_list))])[cur_class_ids]

        cur_class_nb_images = len(cur_class_ids)
        tp = np.zeros(cur_class_nb_images)
        fp = np.zeros(cur_class_nb_images)

        n_positives = 0
        for j, img_id in enumerate(cur_class_ids):
            cur_prob = cur_class_probs[j]
            cur_bb = cur_class_bboxes[j]  # [np.argmax(cur_prob)]
            cur_gt_ids = np.where(np.asarray([c[0] for c in cur_class_gt_boxes_labels[j]]) == cl_id)
            cur_gt = list(np.asarray(cur_class_gt_boxes[j])[cur_gt_ids])
            cur_gt = np.asarray([tuple(c) for c in cur_gt])
            n_positives += len(cur_gt)

            if len(cur_gt) > 0:
                for cur_gt_ in cur_gt:
                    ious = []
                    for cur_bb_ in cur_bb:
                        iou = get_iou(cur_bb_, cur_gt_)
                        ious.append(iou)
                    if np.max(ious) > 0.5:
                        tp[j] = 1.
                    else:
                        fp[j] = 1.

        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / np.maximum(float(n_positives), np.finfo(np.float64).eps)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = compute_ap(rec, prec)
        ap_values.append(ap)
        print(class_str + f' AP: {ap * 100} %')

    mAP = np.mean(ap_values)
    print(f'mAP: {mAP * 100} %')


if __name__ == '__main__':

    valid_dl, preprocessing, classes = get_valid_dataset()
    model, fc, activations = get_pretrained_model()

    predictions = get_predicted_boxes(valid_dl, preprocessing, model, fc, activations)
    get_mAP(predictions, classes)
