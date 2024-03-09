
import os
import sys
import random
import torch
import datetime
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from ptflops import get_model_complexity_info


def test_latency(args, create_model):
    device = torch.device(args.test_device)
    print(f"using {device} device.")

    if args.seed:
        seed = 1234
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    val_data_path = args.test_path
    val_images_path, val_images_label = read_val_data(val_data_path)

    img_size = args.test_image_size
    data_transform = {
        "val": transforms.Compose([transforms.Resize(int(img_size)),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 实例化验证数据集
    val_dataset = DataSet(images_path=val_images_path,
                          images_class=val_images_label,
                          transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    model = create_model(num_classes=args.num_classes).to(device)

    flops, params = get_model_complexity_info(model, (3, img_size, img_size), as_strings=True, print_per_layer_stat=True)
    print("flops: {}, params: {}.".format(flops, params))

    # validate
    model.eval()
    mean_latency = latency(model=model, data_loader=val_loader, device=device, mode=args.mode)
    print("test one image time (mean latency): {}".format(mean_latency))

    print("params: ", params)
    print("flops: ", flops)

    return mean_latency, params, flops


@torch.no_grad()
def latency(model, data_loader, device, mode):
    model.eval()
    if mode == 'cla':
        accu_num = torch.zeros(1).to(device)

        sample_num = 0
        all_time = 0.0
        data_loader = tqdm(data_loader, file=sys.stdout)
        for step, data in enumerate(data_loader):
            images, labels = data
            sample_num += images.shape[0]
            a = images.shape[0]
            test_start_time = datetime.datetime.now()
            pred = model(images.to(device))
            pred_classes = torch.max(pred, dim=1)[1]
            test_end_time = datetime.datetime.now()
            one_time = test_end_time - test_start_time
            one_time = datetime.timedelta.total_seconds(one_time)
            all_time += one_time
            # Latency = (one_time / a) * 1000
            # print("  one image latency on {} (ms): {} / {} = {}".format(device, one_time, a, Latency))
            accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        Mean_latency = (all_time / sample_num) * 1000
        print("\n", "\n", "all time: {}     sample num: {}".format(all_time, sample_num))
        print("mean one image latency on {} (ms):  {} / {} = {}"
              .format(device, all_time, sample_num, Mean_latency))

        return Mean_latency

    elif mode == 'det_seg':
        sample_num = 0
        all_time = 0.0
        data_loader = tqdm(data_loader, file=sys.stdout)
        for step, data in enumerate(data_loader):
            images, labels = data
            sample_num += images.shape[0]
            a = images.shape[0]
            test_start_time = datetime.datetime.now()
            pred = model(images.to(device))
            test_end_time = datetime.datetime.now()
            one_time = test_end_time - test_start_time
            one_time = datetime.timedelta.total_seconds(one_time)
            all_time += one_time
            Latency = (one_time / a) * 1000
            print("  one image latency on {} (ms): {} / {} = {}".format(device, one_time, a, Latency))
        Mean_latency = (all_time / sample_num) * 1000
        print("\n", "\n", "all time: {}     sample num: {}".format(all_time, sample_num))
        print("mean one image latency on {} (ms):  {} / {} = {}"
              .format(device, all_time, sample_num, Mean_latency))
        return Mean_latency

    else:
        print("the mode is error.")


def read_val_data(root: str):
    random.seed(0)
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)
    imagenet_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    imagenet_class.sort()
    class_indices = dict((k, v) for v, k in enumerate(imagenet_class))

    val_images_path = []
    val_images_label = []
    every_class_num = []
    supported = [".jpg", ".JPG", ".png", ".PNG", ".JPEG"]
    for cla in imagenet_class:
        cla_path = os.path.join(root, cla)
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        image_class = class_indices[cla]
        every_class_num.append(len(images))

        for img_path in images:
            val_images_path.append(img_path)
            val_images_label.append(image_class)

    print("{} images for test latency.".format(len(val_images_path)))
    assert len(val_images_path) > 0, "not find data for train."

    return val_images_path, val_images_label


class DataSet(Dataset):
    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        if img.mode != 'RGB':
            img = img.convert("RGB")
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
