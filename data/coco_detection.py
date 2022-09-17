from torch.utils import data
from PIL import Image
from pycocotools.coco import COCO
from torchvision import transforms
from pprint import pprint
from torch.utils.data import DataLoader
import os

from configs.config import cfg


class CocoClsDataset(data.Dataset):
    def __init__(self, root_dir, dataType, annFile):
        self.root_dir = root_dir
        self.dataType = dataType
        self.annFile = root_dir + annFile
        self.coco = COCO(self.annFile)
        self.cats = self.coco.loadCats(self.coco.getCatIds())
        self.labels_str = [cat['name'] for cat in self.cats]
        print('COCO categories: \n{}\n'.format(' '.join(self.labels_str)))
        self.imgIds = self.coco.getImgIds()
        self.images = self.coco.loadImgs(self.imgIds)
        self.root_dir = self.root_dir + f'{dataType}/{dataType}/'

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ])
        self.path_to_label_and_bb = {}
        categories = self.coco.dataset['categories']
        self.id2label = {category['id']: (label, category['name']) for label, category in enumerate(categories)}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        im = self.images[idx]
        img_file_name = im['file_name']
        img_path = os.path.join(self.root_dir, img_file_name)
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        annIds = self.coco.getAnnIds(imgIds=im['id'], iscrowd=None)
        anns = self.coco.loadAnns(annIds)

        label, bb = [], []
        for i in range(len(anns)):
            # print(img_path, (anns[i]['bbox'][0], anns[i]['bbox'][1], anns[i]['bbox'][0] + anns[i]['bbox'][2],
            #                  anns[i]['bbox'][1] + anns[i]['bbox'][3]), anns[i]['category_id'])
            cat_id = anns[i]['category_id']
            label.append(self.id2label[cat_id])
            bb.append((anns[i]['bbox'][0], anns[i]['bbox'][1], anns[i]['bbox'][0] + anns[i]['bbox'][2],
                             anns[i]['bbox'][1] + anns[i]['bbox'][3]))

        self.path_to_label_and_bb[img_path] = (label, bb)
        return img, img_path  #, label, bb
