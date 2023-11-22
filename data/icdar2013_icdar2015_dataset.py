import os
import random
from PIL import Image
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset
import numpy as np

class CustomDataset(BaseDataset):
    def name(self):
        return 'CustomDataset'

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.isTrain = opt.isTrain
        self.data_A, self.data_B = self.load_data()  # Separate data for each domain

        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.shuffle_indices()

    def load_data(self):
        data_A = self.load_year_data('icdar2013')
        data_B = self.load_year_data('icdar2015')
        return data_A, data_B

    def load_year_data(self, year):
        data = []
        folder_path = os.path.join(self.root, year, 'train' if self.isTrain else 'test')
        labels_file = os.path.join(folder_path, 'labels.txt')

        with open(labels_file, 'r') as file:
            labels = file.read().splitlines()

        for label in labels:
            image_name, image_label = label.split()
            image_path = os.path.join(folder_path, 'images', image_name + '.png')
            data.append((image_path, int(image_label)))

        return data

    def shuffle_indices(self):
        self.indices = list(range(max(len(self.data_A), len(self.data_B))))
        if not self.opt.serial_batches:
            random.shuffle(self.indices)

    def __getitem__(self, index):
        if index == 0:
            self.shuffle_indices()

        A_index = index % len(self.data_A)
        B_index = index % len(self.data_B)

        A_image_path, A_label = self.data_A[A_index]
        B_image_path, B_label = self.data_B[B_index]

        A_image = Image.open(A_image_path).convert('RGB')
        A_image = self.transform(A_image)

        B_image = Image.open(B_image_path).convert('RGB')
        B_image = self.transform(B_image)

        # Include labels in the returned item
        item = {'A': A_image, 'B': B_image, 
                'A_paths': A_image_path, 'B_paths': B_image_path, 
                'A_label': A_label, 'B_label': B_label}
        return item


    def __len__(self):
        return max(len(self.data_A), len(self.data_B))

