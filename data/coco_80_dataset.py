from torch.utils import data
from PIL import Image
from pycocotools.coco import COCO
from torchvision import transforms
from pprint import pprint
from torch.utils.data import DataLoader
import os


class CocoClsDataset(data.Dataset):
    def __init__(self, root_dir, ann_file, img_dir, phase, less_sample=False, get_cropped_with_bb_images=True):
        self.ann_file = os.path.join(root_dir, ann_file)
        self.img_dir = os.path.join(root_dir, img_dir)
        self.coco = COCO(self.ann_file)
        self.dataset_type = phase
        self.get_cropped_with_bb_images = get_cropped_with_bb_images

        # self.bg_anns = self._load_bg_anns()

        if phase == 'train':
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
            ])

        cat_ids = self.coco.getCatIds()
        categories = self.coco.dataset['categories']
        self.id2cat = dict()
        for category in categories:
            self.id2cat[category['id']] = category['name']
        self.id2label = {category['id']: (label, category['name']) for label, category in enumerate(categories)}
        # self.id2label_name = {category['name']: label for label, category in enumerate(categories)}
        self.label2id = {v: k for k, v in self.id2label.items()}

        tmp_ann_ids = self.coco.getAnnIds()
        self.ann_ids = []
        for ann_id in tmp_ann_ids:
            ann = self.coco.loadAnns([ann_id])[0]
            x, y, w, h = ann['bbox']
            x, y, w, h = int(x), int(y), int(w), int(h)
            if ann['area'] <= 0 or w < 1 or h < 1 or ann['iscrowd']:
                continue
            self.ann_ids.append(ann_id)

        self._cal_num_dict()

        if phase == 'train' and less_sample:
            self.ann_ids = self._mining_sample()

        print('total_length of dataset:', len(self))

    def _cal_num_dict(self):
        self.num_dict = {}
        for ann_id in self.ann_ids:
            ann = self.coco.loadAnns([ann_id])[0]
            cat = self.id2cat[ann['category_id']]
            num = self.num_dict.get(cat, 0)
            self.num_dict[cat] = num + 1

    def _mining_sample(self):
        self.num_dict = {}
        tmp_ann_ids = []
        for ann_id in self.ann_ids:
            ann = self.coco.loadAnns([ann_id])[0]
            cat = self.id2cat[ann['category_id']]
            num = self.num_dict.get(cat, 0)
            if num >= 20000:
                continue
            self.num_dict[cat] = num + 1
            tmp_ann_ids.append(ann_id)
        return tmp_ann_ids

    def _load_bg_anns(self):
        assert os.path.exists(self.bg_bboxes_file)
        bg_anns = []
        with open(self.bg_bboxes_file, 'r') as f:
            line = f.readline()
            while line:
                if line.strip() == '':
                    break
                file_name, num = line.strip().split()
                for _ in range(int(num)):
                    bbox = f.readline()
                    bbox = bbox.strip().split()
                    bbox = [float(i) for i in bbox]
                    w = bbox[2] - bbox[0] + 1
                    h = bbox[3] - bbox[1] + 1
                    bbox[2], bbox[3] = w, h
                    ann = dict(
                        file_name=file_name,
                        bbox=bbox)
                    bg_anns.append(ann)
                line = f.readline()
        return bg_anns

    def __len__(self):
        return len(self.ann_ids)

    def __getitem__(self, idx):
        ann = self.coco.loadAnns([self.ann_ids[idx]])[0]

        cat_id = ann['category_id']
        label = self.id2label[cat_id]

        img_meta = self.coco.loadImgs(ann['image_id'])[0]
        img_path = os.path.join(self.img_dir, img_meta['file_name'])

        img = Image.open(img_path).convert('RGB')
        x, y, w, h = ann['bbox']
        x, y, w, h = int(x), int(y), int(w), int(h)
        if self.get_cropped_with_bb_images:
            img = img.crop((x, y, x + w - 1, y + h - 1))

        transformed_img = self.transform(img)
        return (transformed_img, img_path), label, (x, y, w, h)


def get_data(cfg):
    """
    Gets data and returns train, test dataloaders
    :param cfg: cfg['data'] part of config
    :return: train, test dataloaders
    """
    print(f'Getting train set...')
    train_set = CocoClsDataset(root_dir=cfg['dataset']['root_path'],
                               ann_file='annotations/instances_train2017.json',
                               img_dir='train2017/train2017',
                               phase='train',
                               less_sample=True)
    print('length: ', len(train_set))
    pprint(train_set.num_dict)
    train_dl = DataLoader(train_set, batch_size=cfg['dataloader']['batch_size'], shuffle=True, drop_last=True)

    print(f'Getting test set...')
    test_set = CocoClsDataset(root_dir=cfg['dataset']['root_path'],
                              ann_file='annotations/instances_val2017.json',
                              img_dir='val2017/val2017',
                              phase='val',
                              less_sample=True)
    print('length: ', len(test_set))
    pprint(test_set.num_dict)
    test_dl = DataLoader(test_set, batch_size=cfg['dataloader']['batch_size'])

    return train_dl, test_dl

