"""
AI模型訓練器
包含訓練、驗證、評估和模型保存功能
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os
import time
from tqdm import tqdm
import json
from datetime import datetime

from config import Config
from models import get_model, count_parameters, initialize_weights
from data_loader import DataProcessor, CustomDataset
from multi_domain_loader import MultiDomainDataProcessor

class EarlyStopping:
    """早停機制"""
    
    def __init__(self, patience=Config.EARLY_STOPPING_PATIENCE, 
                 min_delta=Config.MIN_DELTA):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

class Trainer:
    """AI模型訓練器"""
    
    def __init__(self, model_name='SimpleNN', **model_kwargs):
        self.device = Config.DEVICE
        self.model_name = model_name
        
        # 創建模型
        self.model = get_model(model_name, **model_kwargs)
        initialize_weights(self.model)
        self.model = self.model.to(self.device)
        
        # 優化器和損失函數
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY
        )
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        # 早停
        self.early_stopping = EarlyStopping()
        
        # 記錄
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        # TensorBoard
        self.writer = SummaryWriter(Config.LOG_DIR)
        
        # 數據處理器
        self.data_processor = DataProcessor()
        
        # 多領域數據處理器
        self.multi_domain_processor = MultiDomainDataProcessor()
        
        print(f"模型: {model_name}")
        print(f"參數數量: {count_parameters(self.model):,}")
        print(f"使用設備: {self.device}")
    
    def train_multi_domain(self, data_paths, epochs=10, batch_size=32):
        """
        多領域聯合訓練
        
        Args:
            data_paths: 各領域數據路徑字典 {"domain": "path"}
            epochs: 訓練輪數
            batch_size: 批次大小
        """
        print("開始多領域訓練...")
        
        # 創建多領域數據加載器
        dataloaders = self.multi_domain_processor.create_dataloaders(
            data_paths, batch_size=batch_size
        )
        
        if not dataloaders:
            print("沒有可用的數據加載器")
            return
        
        # 獲取所有訓練和驗證數據加載器
        train_loaders = {k: v for k, v in dataloaders.items() if k.endswith('_train')}
        val_loaders = {k: v for k, v in dataloaders.items() if k.endswith('_val')}
        
        print(f"訓練領域: {list(train_loaders.keys())}")
        
        for epoch in range(epochs):
            print(f"\n=== Epoch {epoch+1}/{epochs} ===")
            
            # 訓練階段
            train_loss, train_acc = self._train_epoch_multi_domain(train_loaders)
            
            # 驗證階段
            val_loss, val_acc = self._validate_epoch_multi_domain(val_loaders)
            
            # 記錄指標
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # TensorBoard記錄
            self.writer.add_scalar('Loss/Train_MultiDomain', train_loss, epoch)
            self.writer.add_scalar('Loss/Val_MultiDomain', val_loss, epoch)
            self.writer.add_scalar('Accuracy/Train_MultiDomain', train_acc, epoch)
            self.writer.add_scalar('Accuracy/Val_MultiDomain', val_acc, epoch)
            
            print(f"訓練損失: {train_loss:.4f}, 訓練準確率: {train_acc:.4f}")
            print(f"驗證損失: {val_loss:.4f}, 驗證準確率: {val_acc:.4f}")
            
            # 學習率調度
            self.scheduler.step(val_loss)
            
            # 早停檢查
            if self.early_stopping(val_loss):
                print("早停觸發，停止訓練")
                break
            
            # 保存檢查點
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch + 1, is_multi_domain=True)
    
    def _train_epoch_multi_domain(self, train_loaders):
        """多領域訓練一個epoch"""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        # 輪流從各個領域採樣數據
        all_batches = []
        for domain, loader in train_loaders.items():
            for batch in loader:
                all_batches.append((domain, batch))
        
        # 隨機打亂批次順序
        np.random.shuffle(all_batches)
        
        progress_bar = tqdm(all_batches, desc="多領域訓練")
        
        for domain, batch in progress_bar:
            # 處理批次數據
            loss, correct, batch_size = self._process_multi_domain_batch(batch, training=True)
            
            total_loss += loss
            total_correct += correct
            total_samples += batch_size
            
            # 更新進度條
            current_acc = total_correct / total_samples if total_samples > 0 else 0
            progress_bar.set_postfix({
                'loss': f'{total_loss/len(all_batches):.4f}',
                'acc': f'{current_acc:.4f}',
                'domain': domain.replace('_train', '')
            })
        
        avg_loss = total_loss / len(all_batches) if all_batches else 0
        avg_acc = total_correct / total_samples if total_samples > 0 else 0
        
        return avg_loss, avg_acc
    
    def _validate_epoch_multi_domain(self, val_loaders):
        """多領域驗證一個epoch"""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        batch_count = 0
        
        with torch.no_grad():
            for domain, loader in val_loaders.items():
                for batch in loader:
                    loss, correct, batch_size = self._process_multi_domain_batch(batch, training=False)
                    
                    total_loss += loss
                    total_correct += correct
                    total_samples += batch_size
                    batch_count += 1
        
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        avg_acc = total_correct / total_samples if total_samples > 0 else 0
        
        return avg_loss, avg_acc
    
    def _process_multi_domain_batch(self, batch, training=True):
        """處理多領域批次數據"""
        inputs = batch['inputs']
        outputs = batch['outputs']
        types = batch['types']
        
        # 簡單的文本特徵提取（示例實現）
        features = []
        labels = []
        
        for inp, out, typ in zip(inputs, outputs, types):
            # 提取基本特徵
            feature = [
                len(inp),  # 輸入長度
                len(out),  # 輸出長度
                hash(typ) % 1000,  # 類型哈希
                inp.count(' '),  # 空格數量
                out.count(' '),  # 輸出空格數量
            ]
            features.append(feature)
            
            # 簡單的標籤生成（示例）
            if typ == 'math':
                label = 0
            elif typ == 'programming':
                label = 1
            elif typ == 'dialogue':
                label = 2
            elif typ == 'writing':
                label = 3
            else:
                label = 4  # 其他類型
            
            labels.append(label)
        
        # 轉換為張量
        features_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)
        labels_tensor = torch.tensor(labels, dtype=torch.long).to(self.device)
        
        if training:
            self.optimizer.zero_grad()
        
        # 前向傳播
        predictions = self.model(features_tensor)
        loss = self.criterion(predictions, labels_tensor)
        
        if training:
            # 反向傳播
            loss.backward()
            self.optimizer.step()
        
        # 計算準確率
        _, predicted = torch.max(predictions.data, 1)
        correct = (predicted == labels_tensor).sum().item()
        
        return loss.item(), correct, len(inputs)
    
    def train_epoch(self, train_loader):
        """訓練一個epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='訓練中')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # 前向傳播
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # 反向傳播
            loss.backward()
            self.optimizer.step()
            
            # 統計
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # 更新進度條
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, val_loader):
        """驗證一個epoch"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                running_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, num_epochs=Config.NUM_EPOCHS):
        """完整訓練流程"""
        print("開始訓練...")
        
        # 獲取數據加載器
        try:
            train_loader, val_loader, test_loader = self.data_processor.get_data_loaders()
        except:
            print("使用MNIST數據集進行訓練...")
            train_loader, val_loader = self.data_processor.load_mnist_data()
            test_loader = val_loader
        
        best_val_loss = float('inf')
        start_time = time.time()
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # 訓練
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # 驗證
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # 記錄
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # TensorBoard記錄
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Validation', val_loss, epoch)
            self.writer.add_scalar('Accuracy/Train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/Validation', val_acc, epoch)
            self.writer.add_scalar('Learning_Rate', 
                                 self.optimizer.param_groups[0]['lr'], epoch)
            
            # 學習率調整
            self.scheduler.step(val_loss)
            
            print(f"訓練損失: {train_loss:.4f}, 訓練準確率: {train_acc:.2f}%")
            print(f"驗證損失: {val_loss:.4f}, 驗證準確率: {val_acc:.2f}%")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model('best_model.pth')
                print("保存最佳模型")
            
            # 早停檢查
            if self.early_stopping(val_loss):
                print(f"早停於第 {epoch+1} epoch")
                break
        
        training_time = time.time() - start_time
        print(f"\n訓練完成！總時間: {training_time:.2f}秒")
        
        # 最終評估
        self.evaluate(test_loader)
        
        # 繪製訓練曲線
        self.plot_training_curves()
        
        # 關閉TensorBoard
        self.writer.close()
    
    def evaluate(self, test_loader):
        """評估模型"""
        print("\n開始評估...")
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc='評估中'):
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                test_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        test_loss /= len(test_loader)
        test_acc = 100. * correct / total
        
        print(f"測試損失: {test_loss:.4f}")
        print(f"測試準確率: {test_acc:.2f}%")
        
        # 分類報告
        print("\n分類報告:")
        print(classification_report(all_targets, all_predictions))
        
        # 混淆矩陣
        self.plot_confusion_matrix(all_targets, all_predictions)
        
        return test_loss, test_acc
    
    def plot_training_curves(self):
        """繪製訓練曲線"""
        plt.figure(figsize=(15, 5))
        
        # 損失曲線
        plt.subplot(1, 3, 1)
        plt.plot(self.train_losses, label='訓練損失')
        plt.plot(self.val_losses, label='驗證損失')
        plt.title('損失曲線')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # 準確率曲線
        plt.subplot(1, 3, 2)
        plt.plot(self.train_accuracies, label='訓練準確率')
        plt.plot(self.val_accuracies, label='驗證準確率')
        plt.title('準確率曲線')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        
        # 學習率曲線
        plt.subplot(1, 3, 3)
        lr_history = []
        for epoch in range(len(self.train_losses)):
            # 這裡簡化處理，實際應該記錄每個epoch的學習率
            lr_history.append(Config.LEARNING_RATE * (0.5 ** (epoch // 5)))
        plt.plot(lr_history)
        plt.title('學習率變化')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(Config.LOG_DIR, 'training_curves.png'))
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """繪製混淆矩陣"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('混淆矩陣')
        plt.xlabel('預測標籤')
        plt.ylabel('真實標籤')
        plt.savefig(os.path.join(Config.LOG_DIR, 'confusion_matrix.png'))
        plt.show()
    
    def save_model(self, filename):
        """保存模型"""
        os.makedirs(Config.MODEL_SAVE_DIR, exist_ok=True)
        filepath = os.path.join(Config.MODEL_SAVE_DIR, filename)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_name': self.model_name,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
        }, filepath)
        
        print(f"模型已保存至: {filepath}")
    
    def load_model(self, filename):
        """加載模型"""
        filepath = os.path.join(Config.MODEL_SAVE_DIR, filename)
        
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # 加載訓練歷史
            self.train_losses = checkpoint.get('train_losses', [])
            self.val_losses = checkpoint.get('val_losses', [])
            self.train_accuracies = checkpoint.get('train_accuracies', [])
            self.val_accuracies = checkpoint.get('val_accuracies', [])
            
            print(f"模型已從 {filepath} 加載")
        else:
            print(f"模型文件 {filepath} 不存在")
    
    def predict(self, data):
        """預測"""
        self.model.eval()
        with torch.no_grad():
            if isinstance(data, np.ndarray):
                data = torch.FloatTensor(data)
            
            data = data.to(self.device)
            if len(data.shape) == 3:  # 單張圖像
                data = data.unsqueeze(0)
            
            output = self.model(data)
            probabilities = torch.softmax(output, dim=1)
            predictions = torch.argmax(output, dim=1)
            
            return predictions.cpu().numpy(), probabilities.cpu().numpy()

if __name__ == "__main__":
    # 創建必要目錄
    from config import create_directories
    create_directories()
    
    # 測試訓練器
    print("開始測試AI訓練器...")
    
    # 使用簡單神經網絡進行測試
    trainer = Trainer(model_name='SimpleNN')
    
    # 開始訓練
    trainer.train(num_epochs=5)  # 測試時使用較少的epoch
    
    print("訓練器測試完成！")