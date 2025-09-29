"""
數據加載和預處理模組
支持圖像數據的加載、預處理和數據增強
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10
import numpy as np
import pandas as pd
from PIL import Image
import os
import cv2
from config import Config
from multi_domain_loader import MultiDomainDataset, MultiDomainDataProcessor
import json

class CustomDataset(Dataset):
    """自定義數據集類 - 支持多種數據格式"""
    
    def __init__(self, data_path, transform=None, is_training=True, data_type="auto"):
        """
        自定義數據集類 - 支持多種數據格式
        
        Args:
            data_path: 數據路径，可以是CSV文件、JSON文件或圖像文件夾
            transform: 數據變換
            is_training: 是否為訓練模式
            data_type: 數據類型 ("image", "text", "multi_domain", "auto")
        """
        self.data_path = data_path
        self.transform = transform
        self.is_training = is_training
        self.data_type = data_type
        self.data = []
        self.labels = []
        
        self.load_data()
    
    def load_data(self):
        """加載數據"""
        # 自動檢測數據類型
        if self.data_type == "auto":
            self.data_type = self._detect_data_type()
        
        if os.path.exists(self.data_path):
            if self.data_type == "multi_domain":
                # 使用多領域數據加載器
                self.multi_domain_dataset = MultiDomainDataset(self.data_path)
                self.data = [item['input'] for item in self.multi_domain_dataset.data]
                self.labels = [item['output'] for item in self.multi_domain_dataset.data]
            # 如果是CSV文件
            elif self.data_path.endswith('.csv'):
                df = pd.read_csv(self.data_path)
                if 'input' in df.columns and 'output' in df.columns:
                    # 文本數據格式
                    self.data = df['input'].values
                    self.labels = df['output'].values
                    self.data_type = "text"
                else:
                    # 傳統特徵-標籤格式
                    self.data = df.iloc[:, :-1].values  # 除最後一列外的所有列作為特徵
                    self.labels = df.iloc[:, -1].values  # 最後一列作為標籤
            # 如果是JSON文件
            elif self.data_path.endswith('.json'):
                self.load_json_data()
            # 如果是圖像文件夾
            elif os.path.isdir(self.data_path):
                for class_name in os.listdir(self.data_path):
                    class_path = os.path.join(self.data_path, class_name)
                    if os.path.isdir(class_path):
                        for img_name in os.listdir(class_path):
                            img_path = os.path.join(class_path, img_name)
                            if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                                self.data.append(img_path)
                                self.labels.append(int(class_name))
        else:
            print(f"警告: 數據路徑 {self.data_path} 不存在，將使用示例數據")
            self.create_sample_data()
    
    def _detect_data_type(self):
        """自動檢測數據類型"""
        if os.path.isdir(self.data_path):
            return "image"
        elif self.data_path.endswith('.json'):
            return "multi_domain"
        elif self.data_path.endswith('.csv'):
            # 檢查CSV內容
            try:
                df = pd.read_csv(self.data_path, nrows=1)
                if 'input' in df.columns and 'output' in df.columns:
                    return "multi_domain"
                else:
                    return "tabular"
            except:
                return "tabular"
        else:
            return "text"
    
    def load_json_data(self):
        """加載JSON數據"""
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            if isinstance(json_data, list):
                for item in json_data:
                    if isinstance(item, dict):
                        input_text = item.get('input', item.get('question', ''))
                        output_text = item.get('output', item.get('answer', ''))
                        if input_text and output_text:
                            self.data.append(input_text)
                            self.labels.append(output_text)
            
            self.data_type = "text"
        except Exception as e:
            print(f"加載JSON數據時出錯: {e}")
    
    def create_sample_data(self):
        """創建示例數據用於演示"""
        # 創建隨機數據
        np.random.seed(42)
        self.data = np.random.randn(1000, Config.INPUT_SIZE).astype(np.float32)
        self.labels = np.random.randint(0, Config.OUTPUT_SIZE, 1000)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if isinstance(self.data[idx], str):  # 圖像路徑
            image = Image.open(self.data[idx]).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, self.labels[idx]
        else:  # 數值數據
            data = torch.FloatTensor(self.data[idx])
            label = torch.LongTensor([self.labels[idx]])[0]
            return data, label

class DataProcessor:
    """數據處理器"""
    
    def __init__(self):
        self.train_transform = self.get_train_transforms()
        self.val_transform = self.get_val_transforms()
    
    def get_train_transforms(self):
        """獲取訓練數據變換"""
        transforms_list = [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ]
        
        if Config.USE_DATA_AUGMENTATION:
            augmentation_transforms = [
                transforms.RandomRotation(Config.ROTATION_DEGREES),
                transforms.ColorJitter(brightness=Config.BRIGHTNESS_FACTOR,
                                     contrast=Config.CONTRAST_FACTOR),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
            transforms_list = augmentation_transforms + transforms_list
        
        return transforms.Compose(transforms_list)
    
    def get_val_transforms(self):
        """獲取驗證數據變換"""
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def get_data_loaders(self):
        """獲取數據加載器"""
        # 訓練數據
        train_dataset = CustomDataset(
            Config.TRAIN_DATA_PATH, 
            transform=self.train_transform, 
            is_training=True
        )
        train_loader = DataLoader(
            train_dataset, 
            batch_size=Config.BATCH_SIZE, 
            shuffle=True,
            num_workers=0  # Windows上設為0避免多進程問題
        )
        
        # 驗證數據
        val_dataset = CustomDataset(
            Config.VAL_DATA_PATH, 
            transform=self.val_transform, 
            is_training=False
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=Config.BATCH_SIZE, 
            shuffle=False,
            num_workers=0
        )
        
        # 測試數據
        test_dataset = CustomDataset(
            Config.TEST_DATA_PATH, 
            transform=self.val_transform, 
            is_training=False
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=Config.BATCH_SIZE, 
            shuffle=False,
            num_workers=0
        )
        
        return train_loader, val_loader, test_loader
    
    def load_mnist_data(self):
        """加載MNIST數據集作為示例"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = MNIST(root='./mnist_data', train=True, 
                             download=True, transform=transform)
        test_dataset = MNIST(root='./mnist_data', train=False, 
                            download=True, transform=transform)
        
        train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, 
                                shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, 
                               shuffle=False, num_workers=0)
        
        return train_loader, test_loader

def preprocess_image(image_path):
    """預處理單張圖像"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"無法讀取圖像: {image_path}")
    
    # 轉換為RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 調整大小
    image = cv2.resize(image, (32, 32))
    
    # 正規化
    image = image.astype(np.float32) / 255.0
    
    # 轉換為tensor
    image = torch.FloatTensor(image).permute(2, 0, 1)
    
    return image.unsqueeze(0)  # 添加batch維度

if __name__ == "__main__":
    # 測試數據處理器
    processor = DataProcessor()
    
    try:
        train_loader, val_loader, test_loader = processor.get_data_loaders()
        print(f"訓練數據批次數: {len(train_loader)}")
        print(f"驗證數據批次數: {len(val_loader)}")
        print(f"測試數據批次數: {len(test_loader)}")
        
        # 測試一個批次
        for batch_idx, (data, target) in enumerate(train_loader):
            print(f"數據形狀: {data.shape}")
            print(f"標籤形狀: {target.shape}")
            break
            
    except Exception as e:
        print(f"使用自定義數據時出錯: {e}")
        print("嘗試使用MNIST數據集...")
        
        train_loader, test_loader = processor.load_mnist_data()
        print(f"MNIST訓練數據批次數: {len(train_loader)}")
        print(f"MNIST測試數據批次數: {len(test_loader)}")
    
    print("數據處理模組測試完成！")