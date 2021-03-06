import math
import os
import datetime
import csv
import time

import numpy as np
import requests
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import config
from datasets import CategoriesSampler, DataSet
from models.ici import ICI
from utils import get_embedding, mean_confidence_interval, setup_seed


def train_embedding(args):
    """
    Train the feature extractor and save the model parameters.
    """
    start_time = time.time()
    setup_seed(2333)
    trained_root = os.path.join('./trained', args.dataset)
    os.makedirs(trained_root, exist_ok=True)
    data_root = os.path.join(args.folder, args.dataset)
    from datasets import EmbeddingDataset
    source_set = EmbeddingDataset(data_root, args.img_size, 'train')
    source_loader = DataLoader(
        source_set, num_workers=args.num_workers, batch_size=64, shuffle=True)
    test_set = EmbeddingDataset(data_root, args.img_size, 'val')
    test_loader = DataLoader(test_set, num_workers=args.num_workers, batch_size=32, shuffle=False)

    if args.dataset == 'cub':
        num_classes = 100
    elif args.dataset == 'tieredImageNet':
        num_classes = 351
    else:
        num_classes = 64
    from models.resnet12 import resnet12
    model = resnet12(num_classes).to(args.device)
    model = model.to(args.device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    from torch.optim.lr_scheduler import StepLR
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    with open('log/train_miniImageNet.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Train-loss", "Train-acc", "Valid-acc"])
        file.flush()
        # Origin is 120
        for epoch in range(120):
            model.train()
            scheduler.step(epoch)
            loss_list = []
            train_acc_list = []
            for images, labels in tqdm(source_loader, ncols=0):
                predictions = model(images.to(args.device))
                loss = criterion(predictions, labels.to(args.device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())
                train_acc_list.append(predictions.max(1)[1].cpu().eq(labels).float().mean().item())
            acc = []
            model.eval()
            for images, labels in test_loader:
                predictions = model(images.to(args.device)).detach().cpu()
                predictions = torch.argmax(predictions, 1).reshape(-1)
                labels = labels.reshape(-1)
                acc += (predictions == labels).tolist()
            acc = np.mean(acc)
            log_info = [epoch, str(np.mean(loss_list))[:6], str(np.mean(train_acc_list))[:6], str(acc)[:6]]
            writer.writerow(log_info)
            file.flush()
            print(log_info)
            if 'MESSAGE_PUSH_URL' in os.environ:
                requests.get(f"{os.environ['MESSAGE_PUSH_URL']}{log_info}")
            if acc > best_acc:
                best_acc = acc
                save_path = os.path.join(
                    trained_root, "res12_epoch{}.pth.tar".format(epoch))
                torch.save(model.state_dict(), save_path)
                torch.save(model.state_dict(), os.path.join(trained_root, 'res12_best.pth.tar'))
    end_time = time.time()
    if 'MESSAGE_PUSH_URL' in os.environ:
        requests.get(f"{os.environ['MESSAGE_PUSH_URL']}Training done, time used: {(end_time - start_time) / 3600}h.")


def train_with_ICI(args):
    with open(args.log_filename, 'a') as file:
        file.write(f"{str(datetime.datetime.now())} : {str(args)}\n")
        file.flush()
    setup_seed(2333)
    import warnings
    warnings.filterwarnings('ignore')
    if args.dataset == 'CUB':
        num_classes = 100
    elif args.dataset == 'tieredImageNet':
        num_classes = 351
    else:
        num_classes = 64
    # Load our ResNet model's parameter from the given path specified by args.resume.
    assert args.resume is not None
    from models.resnet12 import resnet12
    model = resnet12(num_classes).to(args.device)
    state_dict = torch.load(args.resume)
    model.load_state_dict(state_dict)
    model.to(args.device)
    # Change the PyTorch model to evaluation mode.
    model.eval()
    ici = ICI(classifier=args.classifier, num_class=args.num_test_ways,
              step=args.step, reduce=args.embed, d=args.dim)
    # Load the specified dataset.
    data_root = os.path.join(args.folder, args.dataset)
    # DataSet is used to offer data which can be accessed by index
    test_dataset = DataSet(data_root, 'test', args.img_size)
    # sample is used to offer indexes
    sampler = CategoriesSampler(test_dataset.label, args.num_batches,
                                args.num_test_ways, (args.num_shots, 15, args.unlabeled))
    test_loader = DataLoader(test_dataset, batch_sampler=sampler, num_workers=args.num_workers, pin_memory=True)
    k = args.num_shots * args.num_test_ways
    loader = tqdm(test_loader, ncols=0)
    iterations = math.ceil(args.unlabeled / args.step) + 2 if args.unlabeled != 0 else math.ceil(15 / args.step) + 2
    acc_list = [[] for _ in range(iterations)]
    for data, indicator in loader:
        targets = torch.arange(args.num_test_ways).repeat(args.num_shots + 15 + args.unlabeled).long()[
            indicator[:args.num_test_ways * (args.num_shots + 15 + args.unlabeled)] != 0]
        data = data[indicator != 0].to(args.device)
        train_inputs = data[:k]
        train_targets = targets[:k].cpu().numpy()
        test_inputs = data[k:k + 15 * args.num_test_ways]
        test_targets = targets[k:k + 15 * args.num_test_ways].cpu().numpy()
        train_embeddings = get_embedding(model, train_inputs, args.device)
        ici.fit(train_embeddings, train_targets)
        test_embeddings = get_embedding(model, test_inputs, args.device)
        if args.unlabeled != 0:
            unlabeled_inputs = data[k + 15 * args.num_test_ways:]
            unlabeled_embeddings = get_embedding(model, unlabeled_inputs, args.device)
        else:
            unlabeled_embeddings = None
        acc = ici.predict(test_embeddings, unlabeled_embeddings, True, test_targets, disable_ici=args.disable_ici)
        for i in range(min(iterations - 1, len(acc))):
            acc_list[i].append(acc[i])
        acc_list[-1].append(acc[-1])
    mean_list = []
    ci_list = []
    for item in acc_list:
        mean, ci = mean_confidence_interval(item)
        mean_list.append(mean)
        ci_list.append(ci)

    result = "Test Acc Mean {}".format(' '.join([str(i * 100)[:5] for i in mean_list]))
    result += "\tTest Acc ci {}".format(' '.join([str(i * 100)[:5] for i in ci_list]))
    with open(args.log_filename, 'a') as file:
        file.write(f"{str(datetime.datetime.now())} : {str(result)}\n")
        file.flush()
    print(result)


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(args)
    if args.mode == 'train':
        train_embedding(args)
    elif args.mode == 'test':
        train_with_ICI(args)
    else:
        raise NameError


if __name__ == '__main__':
    main(config())
