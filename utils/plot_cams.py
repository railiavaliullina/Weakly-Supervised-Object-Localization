from PIL import Image
from torch.nn import functional as f
from data.coco_80_dataset import CocoClsDataset
from torch.utils.data import DataLoader
import numpy as np
import cv2
import torch
from torchvision import transforms
from models.resnet50 import get_model
from configs.config import cfg


def get_dataset():
    test_set = CocoClsDataset(root_dir=cfg['data']['dataset']['root_path'],
                              ann_file='annotations/instances_val2017.json',
                              img_dir='val2017/val2017',
                              phase='val',
                              less_sample=True,
                              get_cropped_with_bb_images=False)
    test_dl = DataLoader(test_set, batch_size=1, shuffle=True)
    classes = {int(key): value for (key, value) in enumerate(list(test_set.id2cat.values()))}
    preprocessing = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])
    return test_dl, classes, preprocessing


def get_pretrained_model():
    model = get_model(cfg)
    print(f'Trying to load checkpoint from epoch {cfg["CAM"]["epoch_to_load"]}...')
    checkpoint = torch.load(cfg['CAM']['checkpoints_dir'] + f'checkpoint_{cfg["CAM"]["epoch_to_load"]}.pth')
    load_state_dict = checkpoint['model']
    model.load_state_dict(load_state_dict)
    print(f'Successfully loaded checkpoint from epoch {cfg["CAM"]["epoch_to_load"]}.')
    model.eval()
    return model


def save_activation(module, input, output):
    activations.append(output.data.cpu().numpy())


def get_cams():
    for k, batch in enumerate(dl):
        (_, path), (label, label_name), gt = batch
        img = Image.open(path[0]).convert('RGB')
        print(f'target: {label_name}')

        input_tensor = preprocessing(img).unsqueeze(0)
        output = f.softmax(model(input_tensor), dim=1).data.squeeze()
        softmax_outs, labels_idxs = output.sort(0, True)
        softmax_outs, labels_idxs = softmax_outs.numpy(), labels_idxs.numpy()
        _, channels, h, w = activations[0].shape
        cam_ = []
        for idx_ in labels_idxs:
            cam = fc[idx_].dot(activations[0].reshape((channels, h * w)))
            cam = cam.reshape(h, w)
            cam = cam - np.min(cam)
            cam_img = cam / np.max(cam)
            cam_img = np.uint8(255 * cam_img)
            cam_.append(cam_img)

        img = cv2.imread(path[0])
        height, width, _ = img.shape
        heatmap = cv2.applyColorMap(cv2.resize(cam_[0], (width, height)), cv2.COLORMAP_JET)
        result = heatmap * 0.5 + img * 0.5
        cv2.imwrite('cam.jpg', result)
        # if k == 10:
        #     break
        get_bb(heatmap, img, gt)


def get_bb(cam, rgb_img, gt):
    gray_heatmap = np.uint8(255 * cam)
    gray_heatmap = cv2.cvtColor(gray_heatmap, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray_heatmap, 0, 255, cv2.THRESH_OTSU)[1]
    # Find contours
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    bb_img = np.uint8(255 * rgb_img.copy())
    r, g, b = cv2.split(bb_img)
    bb_img = cv2.merge([b, g, r])
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(bb_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(gray_heatmap, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imwrite('bb_img.jpg', bb_img)
    cv2.imwrite('gray_heatmap.jpg', gray_heatmap)
    # cv2.imshow("bb", bb)


if __name__ == '__main__':
    model = get_pretrained_model()
    dl, classes, preprocessing = get_dataset()

    activations = []
    model._modules.get('layer4').register_forward_hook(save_activation)
    fc = np.squeeze(list(model.parameters())[-2].data.numpy())
    get_cams()
