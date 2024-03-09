import datetime
import os
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from DecoupleNet import DecoupleNet_D0_1662_e32_k9 as create_model
from utils import train_one_epoch, evaluate, sample, MyDataSet, test2, create_lr_scheduler, read_train_data, read_val_data
from test_latency import test_latency
from ptflops import get_model_complexity_info


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--version', type=str,
                        default='DecoupleNet_D0_1662_e32_k9/{}'
                        .format(datetime.datetime.now().strftime("%y.%m.%d-%H:%M")))

    parser.add_argument('--num_classes', type=int, default=46)
    parser.add_argument('--epochs', type=int, default=300)
    # batchsize和学习率的关系
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--wd', type=float, default=0.05)
    parser.add_argument('--seed', type=bool, default=True)
    # 数据集所在根目录
    parser.add_argument('--data-path', type=str, default="")
    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    # 是否继续训练
    parser.add_argument('--resume', type=bool, default=False)
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    # 训练所用设备
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    # parser.add_argument('--devices', default=True, help='device id (i.e. 0 or 0,1 or cpu)')

    # test latency
    parser.add_argument('--test-latency', type=bool, default=False)
    parser.add_argument('--mode', type=str, default='cla')
    parser.add_argument('--test-image-size', type=int, default=224)
    # 测试延迟数据集路径
    parser.add_argument('--test-path', type=str, default="")
    # 测试延迟设备
    parser.add_argument('--test-device', default='cpu', help='device id')
    args = parser.parse_args()
    return args


def main(args):

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")
    version = args.version
    time = "results_{}.txt".format(datetime.datetime.now().strftime("%y%m%d-%H%M"))
    output = os.path.join('./output', version)
    results_file = os.path.join(output, time)
    if os.path.exists(output) is False:
        os.makedirs(output)

    tensorboard = os.path.join(output, 'runs')
    tb_writer = SummaryWriter(tensorboard)

    if args.seed:
        seed = 1234
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.deterministic = False

    train_data_path = args.data_path + "train/"
    val_data_path = args.data_path + "val/"
    train_images_path, train_images_label = read_train_data(train_data_path)
    val_images_path, val_images_label = read_val_data(val_data_path)

    img_size = 224
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandAugment(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    model = create_model(num_classes=args.num_classes).to(device)

    # if torch.cuda.device_count() > 1 and args.devices == True:
    #     device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    #     print("Use", torch.cuda.device_count(), 'gpus')
    #     model = nn.DataParallel(model).to(device)

    with open(results_file, "a") as f:
        info = f"args: {args}\n"
        f.write(info + "\n")
    # 加载预训练权重，并删除你想删除的部分
    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)["model"]
        # 删除有关分类类别的权重
        # for k in list(weights_dict.keys()):
        # #     # if "head" in k:
        # #     #     print("delete:", k)
        # #     #     del weights_dict[k]
        #     if "downsample" in k:
        #         print("delete:", k)
        #         del weights_dict[k]
        #     if "patch_embed" in k:
        #         print("delete:", k)
        #         del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    # 是否冻结权重或解冻
    if args.freeze_layers:
        for name, para in model.named_parameters():
            para.requires_grad_(False)
        # 解冻并训练自己需要的权重数据
        for name, para in model.named_parameters():
            # if "head" in name:
            #     para.requires_grad_(True)
            #     print("training {}".format(name))
            if "downsample" in name:
                para.requires_grad_(True)
                print("training {}".format(name))
            if "patch_embed" in name:
                para.requires_grad_(True)
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=args.wd)
    scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True, warmup_epochs=2)

    best_acc = 0.0
    best_epoch = 0
    # 是否断点继续训练
    start_epoch = -1
    if args.resume:
        path_checkpoint = "last-val_acc.pth"                # 断点路径
        checkpoint = torch.load(path_checkpoint)            # 加载断点
        model.load_state_dict(checkpoint['model'])          # 加载模型可学习参数
        optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
        start_epoch = checkpoint['epoch']                   # 设置开始的epoch


    # test latency
    if args.test_latency:
        print("test cpu latency")
        mean_latency, params, flops = test_latency(args, create_model)

        print("test gpu latency")
        sample_num = sample(data_loader=val_loader)
        test_all_time1 = test2(model=model, data_loader=val_loader, device=device)
        test_all_time2 = test2(model=model, data_loader=val_loader, device=device)
        test_all_time3 = test2(model=model, data_loader=val_loader, device=device)
        test_all_time = (test_all_time1 + test_all_time2 + test_all_time3) / 3.0
        one_time1 = (test_all_time / float(sample_num)) * 1000
        FPS1 = sample_num / test_all_time
        print("sample num: ", sample_num)
        print("test one image time on gpu (ms): {} ", one_time1)
        print("test FPS on gpu (images/s): ", FPS1)
        print("test mean latency on cpu (ms): ", mean_latency)
        print("params: {}, flops: {}".format(params, flops))

        with open(results_file, "a") as f:
            info = f"test one image time on gpu (ms): {one_time1:.4f} \n "\
                   f"test one image time on cpu (ms): {mean_latency:.4f}  \n "\
                   f"test FPS  on gpu(images/s): {FPS1:.4f}  \n "\
                   f"params: {params} , flops: {flops} "
            f.write(info + "\n")
    else:
        mean_latency = 0
        FPS = 0
        flops, params = get_model_complexity_info(model, (3, img_size, img_size), as_strings=True,
                                                  print_per_layer_stat=True)
        print("flops: {}, params: {}.".format(flops, params))

    start_time = datetime.datetime.now()

    print("start_time:", start_time)

    for epoch in range(start_epoch + 1, args.epochs):
        # train
        model.train()
        train_loss, train_acc = train_one_epoch(model=model, optimizer=optimizer, scheduler=scheduler,
                                                data_loader=train_loader, device=device, epoch=epoch)

        # validate
        model.eval()
        val_loss, val_acc = evaluate(model=model, data_loader=val_loader,
                                     device=device, epoch=epoch)

        # save checkpoint
        save_path = os.path.join(output, "weights")
        if epoch >= 0:
            checkpoint = {
                "model": model.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": epoch}
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            torch.save(checkpoint, './output/{}/weights/last-val_acc.pth'.format(version))

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            checkpoint = {
                "model": model.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": epoch}
            torch.save(checkpoint, './output/{}/weights/best-val_acc.pth'.format(version))

        if epoch == 299 and args.test_latency:
            sample_num = sample(data_loader=val_loader)

            test_all_time1 = test2(model=model, data_loader=val_loader, device=device)
            test_all_time2 = test2(model=model, data_loader=val_loader, device=device)
            test_all_time3 = test2(model=model, data_loader=val_loader, device=device)
            test_all_time = (test_all_time1 + test_all_time2 + test_all_time3) / 3.0
            one_time = (test_all_time / float(sample_num)) * 1000
            FPS = sample_num / test_all_time
            mean_latency, params, flops = test_latency(args, create_model)
            print("test all time (s): ", test_all_time)
            print("sample num: ", sample_num)
            print("test one image time (Latency): {} ms".format(one_time))
            print("test FPS (images/s): ", FPS)
        else:
            mean_latency = 0
            FPS1 = 0

        with open(results_file, "a") as f:
            info = f"[epoch: {epoch}]  "\
                   f"train_acc: {train_acc:.4f}  " \
                   f"train_loss: {train_loss:.4f}  " \
                   f"val_acc: {val_acc:.4f}  "\
                   f"val_loss: {val_loss:.4f}  "
            f.write(info + "\n")

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "best_acc", "CPU_mean_latency", "GPU_FPS", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], best_acc, epoch)
        tb_writer.add_scalar(tags[5], mean_latency, epoch)
        tb_writer.add_scalar(tags[6], FPS1, epoch)
        tb_writer.add_scalar(tags[7], optimizer.param_groups[0]["lr"], epoch)

    # train time
    end_time = datetime.datetime.now()
    print("start_time:", start_time)
    print("end_time:", end_time)
    all_time = end_time - start_time
    print("all_time:", all_time)

    with open(results_file, "a") as f:
        info = f"params: {params} , flops: {flops} \n "\
               f"best_epoch: {best_epoch}\n" \
               f"best_acc: {best_acc:.4f}\n" \
               f"test one image time on gpu (Latency): {one_time}\n"\
               f"test one image time on cpu (ms): {mean_latency:.4f}  \n "\
               f"test FPS on gpu (images/s): {FPS}\n" \
               f"start_time: {start_time}\n" \
               f"end_time: {end_time}\n" \
               f"train_and_val all_time: {all_time}\n"\
               f"params: {params} , flops: {flops} "
        f.write(info + "\n\n")


if __name__ == '__main__':
    args = parse_args()

    main(args)
