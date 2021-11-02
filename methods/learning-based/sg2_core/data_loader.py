"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

from pathlib import Path
from itertools import chain
import os
import random

from munch import Munch
from PIL import Image
import numpy as np

import torch
from torch.utils import data
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder


def listdir(dname):
    fnames = list(chain(*[list(Path(dname).rglob('*.' + ext))
                          for ext in ['png', 'jpg', 'jpeg', 'JPG']]))
    return fnames


class DefaultDataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.samples = listdir(root)
        self.samples.sort()
        self.transform = transform
        self.targets = None

    def __getitem__(self, index):
        fname = self.samples[index]
        img = Image.open(fname).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)


class ReferenceDataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.samples, self.targets = self._make_dataset(root)
        self.transform = transform

    def _make_dataset(self, root):
        domains = os.listdir(root)
        fnames, fnames2, labels = [], [], []
        for idx, domain in enumerate(sorted(domains)):
            class_dir = os.path.join(root, domain)
            cls_fnames = listdir(class_dir)
            fnames += cls_fnames
            fnames2 += random.sample(cls_fnames, len(cls_fnames))
            labels += [idx] * len(cls_fnames)
        return list(zip(fnames, fnames2)), labels

    def __getitem__(self, index):
        fname, fname2 = self.samples[index]
        label = self.targets[index]
        img = Image.open(fname).convert('RGB')
        img2 = Image.open(fname2).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
            img2 = self.transform(img2)
        return img, img2, label

    def __len__(self):
        return len(self.targets)


def _make_balanced_sampler(labels):
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    weights = class_weights[labels]
    return WeightedRandomSampler(weights, len(weights))


def get_train_loader(root, which='source', img_size=256,
                     batch_size=8, prob=0.5, num_workers=4):
    print('Preparing DataLoader to fetch %s images '
          'during the training phase...' % which)

    crop = transforms.RandomResizedCrop(
        img_size, scale=[0.8, 1.0], ratio=[0.9, 1.1])
    rand_crop = transforms.Lambda(
        lambda x: crop(x) if random.random() < prob else x)

    transform = transforms.Compose([
        rand_crop,
        transforms.Resize([img_size, img_size]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    if which == 'source':
        dataset = ImageFolder(root, transform)
    elif which == 'reference':
        dataset = ReferenceDataset(root, transform)
    else:
        raise NotImplementedError

    sampler = _make_balanced_sampler(dataset.targets)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           sampler=sampler,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=True)


def get_eval_loader(root, img_size=256, batch_size=4,
                    imagenet_normalize=True, shuffle=False,
                    num_workers=0, drop_last=False):
    print('Preparing DataLoader for the evaluation phase...')
    if imagenet_normalize:
        height, width = 299, 299
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        height, width = img_size, img_size
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.Resize([height, width]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    dataset = DefaultDataset(root, transform=transform)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=drop_last)


def get_test_loader(root, img_size=256, batch_size=32,
                    shuffle=True, num_workers=4):
    print('Preparing DataLoader for the generation phase...')
    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    dataset = ImageFolder(root, transform)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True)


class InputFetcher:
    def __init__(self, loader, loader_ref=None, latent_dim=16, mode=''):
        self.loader = loader
        self.loader_ref = loader_ref
        self.latent_dim = latent_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode = mode

    def _fetch_inputs(self):
        try:
            x, y = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            x, y = next(self.iter)
        return x, y

    def _fetch_refs(self):
        try:
            x, x2, y = next(self.iter_ref)
        except (AttributeError, StopIteration):
            self.iter_ref = iter(self.loader_ref)
            x, x2, y = next(self.iter_ref)
        return x, x2, y

    def __next__(self):
        x, y = self._fetch_inputs()
        if self.mode == 'train':
            x_ref, x_ref2, y_ref = self._fetch_refs()
            z_trg = torch.randn(x.size(0), self.latent_dim)
            z_trg2 = torch.randn(x.size(0), self.latent_dim)
            inputs = Munch(x_src=x, y_src=y, y_ref=y_ref,
                           x_ref=x_ref, x_ref2=x_ref2,
                           z_trg=z_trg, z_trg2=z_trg2)
        elif self.mode == 'val':
            x_ref, y_ref = self._fetch_inputs()
            inputs = Munch(x_src=x, y_src=y,
                           x_ref=x_ref, y_ref=y_ref)
        elif self.mode == 'test':
            inputs = Munch(x=x, y=y)
        else:
            raise NotImplementedError

        return Munch({k: v.to(self.device)
                      for k, v in inputs.items()})
      
# =============================================================================

class DatasetFC2(data.Dataset):
  def __init__(self, data_dir, style_dir, temp_dir, transform, num_dom=4):
    self.data_dir = data_dir
    self.style_dir = style_dir
    self.temp_dir = temp_dir
    
    self.transform = transform
    self.num_dom = num_dom
    self.dataset = []
    self.styles = []
    
    self.preprocess()
    self.num_images = len(self.dataset)
    print(self.num_images)

  def __getitem__(self, index):
    file, src_lbl, ref_lbl = self.dataset[index]

    src_path = self.style_dir + self.styles[src_lbl] + file
    src_path2 = self.temp_dir + self.styles[src_lbl] + file[:-4] + "_2.jpg"
    ref_path = self.style_dir + self.styles[ref_lbl] + file
    
    src_img =  self.transform(Image.open(src_path))
    src_img2 = self.transform(Image.open(src_path2))
    ref_img =  self.transform(Image.open(ref_path))
    
    np_data = np.load(self.data_dir + file[:-4] + ".npy")[0]
    mask = torch.from_numpy(np.moveaxis(np_data[:,:,6:7], 2, 0))
    flow = torch.from_numpy(np.moveaxis(np_data[:,:,7:9], 2, 0))
    
    src_lbl = torch.Tensor([src_lbl])[0].type(torch.LongTensor)
    ref_lbl = torch.Tensor([ref_lbl])[0].type(torch.LongTensor)

    return src_img, src_img2, src_lbl, ref_img, ref_lbl, mask, flow
  
  def __len__(self):
    return self.num_images
  
  def preprocess(self, base_len=22208):
    src_list = os.listdir(self.data_dir)
    src_list.sort()
    
    style_list = os.listdir(self.style_dir)
    style_list.sort()
    style_list = style_list[:self.num_dom]
    
    #check if there is a style image for every source image for all styles
    for sty in style_list:
      sty_list = os.listdir(self.style_dir + sty)
      sty_list.sort()
      #assert len(src_list) == len(sty_list)
      print(base_len, len(sty_list))
      assert base_len == len(sty_list)
      self.styles.append(sty)

    #for src in sorted(os.listdir(self.style_dir + style_list[0])):
    #  self.dataset.append([src, 0, src, 0])
    #  
    #  for i, sty in enumerate(style_list[1:]):
    #    for ref in sorted(os.listdir(self.style_dir + sty)):
    #      self.dataset.append([src, 0, ref, i + 1])
    #      self.dataset.append([ref, i + 1, src, 0])
    #      self.dataset.append([ref, i + 1, ref, i + 1])
          
    for img in sorted(os.listdir(self.style_dir + style_list[0])):
      file = "/" + img
      self.dataset.append([file, 0, 0])
      
      for i, sty in enumerate(style_list[1:]):
        self.dataset.append([file, 0, i + 1])
        self.dataset.append([file, i + 1, 0])
        self.dataset.append([file, i + 1, i + 1])
    
    print("Dataset Len:", len(self.dataset))
    
    random.seed(1234)
    random.shuffle(self.dataset)
  
def get_loaderFC2(data_dir, style_dir, temp_dir, batch_size=4, num_workers=0, num_dom=2, mode="train"):
  transform = []
  #if mode == "train":
  #  transform.append(T.RandomHorizontalFlip())
  #transform.append(T.CenterCrop(crop_size))
  #transform.append(T.Resize(image_size))
  transform.append(transforms.ToTensor())
  transform.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
  transform = transforms.Compose(transform)

  full_dataset = DatasetFC2(data_dir, style_dir, temp_dir, transform, num_dom)
  
  train_size = int(0.97 * len(full_dataset))
  eval_size = len(full_dataset) - train_size
  print("training size", train_size)
  print("eval size", eval_size)
  train_dataset, eval_dataset = torch.utils.data.random_split(full_dataset, [train_size, eval_size])
  
  #if mode == "train":
  train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
  eval_loader = data.DataLoader(dataset=eval_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
  
  return train_loader, eval_loader

# =============================================================================

class FC2Fetcher:
  def __init__(self, loader, loader_ref=None, latent_dim=16, mode=''):
    self.loader = loader
    #self.loader_ref = loader_ref
    self.latent_dim = latent_dim
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.mode = mode
    
  def _fetch_inputs_fc2(self):
    try:
      src_img, src_img2, src_lbl, ref_img, ref_lbl, mask, flow = next(self.iter)
    except (AttributeError, StopIteration):
      self.iter = iter(self.loader)
      src_img, src_img2, src_lbl, ref_img, ref_lbl, mask, flow = next(self.iter)

    return src_img, src_img2, src_lbl, ref_img, ref_lbl, mask, flow

  def __next__(self):
    src_img, src_img2, src_lbl, ref_img, ref_lbl, mask, flow = self._fetch_inputs_fc2()
    z_trg = torch.randn(src_img.size(0), self.latent_dim)
    z_trg2 = torch.randn(src_img.size(0), self.latent_dim)
    inputs = Munch(x_src=src_img, x2_src=src_img2, y_src=src_lbl,
                   x_ref=ref_img, y_ref=ref_lbl,
                   mask=mask, flow=flow,
                   z_trg=z_trg, z_trg2=z_trg2)

    return Munch({k: v.to(self.device)
                  for k, v in inputs.items()})
