"""
神經網絡模型定義
包含多種模型架構：簡單神經網絡、卷積神經網絡、殘差網絡等
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

class SimpleNN(nn.Module):
    """簡單的全連接神經網絡"""
    
    def __init__(self, input_size=784, 
                 hidden_size=128, 
                 output_size=10,
                 num_classes=None):
        super(SimpleNN, self).__init__()
        
        # 如果提供了num_classes，使用它作為output_size
        if num_classes is not None:
            output_size = num_classes
            
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)  # 使用LayerNorm替代BatchNorm
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)  # 使用LayerNorm替代BatchNorm
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = torch.relu(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.ln2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class CNN(nn.Module):
    """卷積神經網絡"""
    
    def __init__(self, num_classes=Config.OUTPUT_SIZE):
        super(CNN, self).__init__()
        
        # 卷積層
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # 批次正規化
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        # 池化層
        self.pool = nn.MaxPool2d(2, 2)
        
        # 全連接層
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(Config.DROPOUT_RATE)
        
    def forward(self, x):
        # 第一個卷積塊
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # 第二個卷積塊
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # 第三個卷積塊
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全連接層
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class ResidualBlock(nn.Module):
    """殘差塊"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                              kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 
                              kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(residual)
        out = F.relu(out)
        
        return out

class ResNet(nn.Module):
    """簡化版ResNet"""
    
    def __init__(self, num_classes=Config.OUTPUT_SIZE):
        super(ResNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # 殘差層
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

class AutoEncoder(nn.Module):
    """自編碼器"""
    
    def __init__(self, input_size=Config.INPUT_SIZE, encoding_dim=64):
        super(AutoEncoder, self).__init__()
        
        # 編碼器
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, encoding_dim),
            nn.ReLU()
        )
        
        # 解碼器
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_size),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        x = x.view(x.size(0), -1)
        return self.encoder(x)

def get_model(model_name='SimpleNN', **kwargs):
    """模型工廠函數"""
    models = {
        'SimpleNN': SimpleNN,
        'CNN': CNN,
        'ResNet': ResNet,
        'AutoEncoder': AutoEncoder
    }
    
    if model_name not in models:
        raise ValueError(f"未知的模型類型: {model_name}")
    
    model = models[model_name](**kwargs)
    return model

def count_parameters(model):
    """計算模型參數數量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initialize_weights(model):
    """初始化模型權重"""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

if __name__ == "__main__":
    # 測試所有模型
    device = Config.DEVICE
    
    models_to_test = ['SimpleNN', 'CNN', 'ResNet', 'AutoEncoder']
    
    for model_name in models_to_test:
        print(f"\n測試模型: {model_name}")
        
        try:
            model = get_model(model_name)
            model = model.to(device)
            
            # 初始化權重
            initialize_weights(model)
            
            # 計算參數數量
            param_count = count_parameters(model)
            print(f"參數數量: {param_count:,}")
            
            # 測試前向傳播
            if model_name in ['CNN', 'ResNet']:
                test_input = torch.randn(1, 3, 32, 32).to(device)
            else:
                test_input = torch.randn(1, Config.INPUT_SIZE).to(device)
            
            with torch.no_grad():
                output = model(test_input)
                print(f"輸入形狀: {test_input.shape}")
                print(f"輸出形狀: {output.shape}")
                
        except Exception as e:
            print(f"模型 {model_name} 測試失敗: {e}")
    
    print("\n模型定義測試完成！")