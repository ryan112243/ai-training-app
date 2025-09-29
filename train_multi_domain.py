"""
直接執行多領域模型訓練腳本
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
from datetime import datetime
from multi_domain_loader import MultiDomainDataProcessor
from models import SimpleNN
from config import Config

class MultiDomainTrainer:
    """多領域模型訓練器"""
    
    def __init__(self, config=None):
        self.config = config or Config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用設備: {self.device}")
        
    def train_model(self, data_paths, epochs=5, batch_size=2, learning_rate=0.001):
        """
        訓練多領域模型
        
        Args:
            data_paths: 數據路徑字典
            epochs: 訓練輪數
            batch_size: 批次大小
            learning_rate: 學習率
        """
        try:
            print("開始多領域模型訓練...")
            
            # 創建數據處理器
            processor = MultiDomainDataProcessor(self.config)
            
            # 創建數據加載器
            dataloaders = processor.create_dataloaders(
                data_paths, 
                batch_size=batch_size,
                train_split=0.8
            )
            
            if not dataloaders:
                print("錯誤: 沒有可用的數據加載器")
                return None
            
            # 創建模型
            model = SimpleNN(
                input_size=512,  # 假設輸入特徵維度
                hidden_size=256,
                output_size=512,  # 修改輸出維度與輸入匹配
                num_classes=512  # 領域數量
            ).to(self.device)
            
            # 定義損失函數和優化器
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            
            # 訓練循環
            training_history = []
            
            for epoch in range(epochs):
                epoch_loss = 0.0
                batch_count = 0
                
                print(f"\n第 {epoch + 1}/{epochs} 輪訓練:")
                
                # 遍歷所有訓練數據加載器
                for loader_name, loader in dataloaders.items():
                    if 'train' not in loader_name:
                        continue
                        
                    domain = loader_name.replace('_train', '')
                    print(f"  訓練 {domain} 領域...")
                    
                    for batch_idx, batch in enumerate(loader):
                        # 簡化的特徵提取（實際應用中需要更複雜的處理）
                        inputs = self._extract_features(batch['inputs'])
                        targets = self._extract_features(batch['outputs'])
                        
                        inputs = inputs.to(self.device)
                        targets = targets.to(self.device)
                        
                        # 前向傳播
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        
                        # 計算損失
                        loss = criterion(outputs, targets)
                        
                        # 反向傳播
                        loss.backward()
                        optimizer.step()
                        
                        epoch_loss += loss.item()
                        batch_count += 1
                        
                        if batch_idx % 10 == 0:
                            print(f"    批次 {batch_idx}, 損失: {loss.item():.4f}")
                
                avg_loss = epoch_loss / max(batch_count, 1)
                training_history.append({
                    'epoch': epoch + 1,
                    'loss': avg_loss
                })
                
                print(f"  平均損失: {avg_loss:.4f}")
            
            # 保存模型（未加密）
            model_path = self._save_model(model, training_history)
            
            print(f"\n訓練完成！模型已保存至: {model_path}")
            return model_path
            
        except Exception as e:
            print(f"訓練過程中出錯: {e}")
            return None
    
    def _extract_features(self, texts):
        """
        簡化的特徵提取（將文本轉換為數值特徵）
        實際應用中應該使用更複雜的文本編碼方法
        """
        features = []
        for text in texts:
            # 簡單的字符級特徵提取
            feature = [ord(c) % 256 for c in text[:512]]
            # 填充或截斷到固定長度
            if len(feature) < 512:
                feature.extend([0] * (512 - len(feature)))
            else:
                feature = feature[:512]
            features.append(feature)
        
        return torch.FloatTensor(features)
    
    def _save_model(self, model, training_history):
        """保存模型（未加密格式）"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = "models/multi_domain"
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, f"multi_domain_model_{timestamp}.pth")
        
        # 保存完整的模型狀態（未加密）
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_architecture': {
                'input_size': 512,
                'hidden_size': 256,
                'output_size': 512,
                'num_classes': 512
            },
            'training_history': training_history,
            'timestamp': timestamp,
            'pytorch_version': torch.__version__
        }, model_path)
        
        # 同時保存為可讀的JSON格式配置
        config_path = os.path.join(model_dir, f"model_config_{timestamp}.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump({
                'model_path': model_path,
                'architecture': {
                    'input_size': 512,
                    'hidden_size': 256,
                    'output_size': 512,
                    'num_classes': 512
                },
                'training_history': training_history,
                'timestamp': timestamp,
                'usage_instructions': {
                    'load_model': f"torch.load('{model_path}')",
                    'description': "未加密的PyTorch模型，可直接在其他程式中載入使用"
                }
            }, f, ensure_ascii=False, indent=2)
        
        print(f"模型配置已保存至: {config_path}")
        return model_path

def main():
    """主函數 - 直接執行訓練"""
    print("=== 多領域AI模型直接訓練 ===")
    
    # 數據路徑配置
    data_paths = {
        "math": "data/samples/math_sample.json",
        "programming": "data/samples/programming_sample.json",
        "dialogue": "data/samples/dialogue_sample.json"
    }
    
    # 檢查數據文件是否存在
    missing_files = []
    for domain, path in data_paths.items():
        if not os.path.exists(path):
            missing_files.append(f"{domain}: {path}")
    
    if missing_files:
        print("錯誤: 以下數據文件不存在:")
        for file in missing_files:
            print(f"  - {file}")
        print("\n請先創建示例數據或檢查文件路徑")
        return
    
    # 創建訓練器
    trainer = MultiDomainTrainer()
    
    # 開始訓練
    model_path = trainer.train_model(
        data_paths=data_paths,
        epochs=3,  # 較少的輪數用於快速測試
        batch_size=1,  # 小批次適合示例數據
        learning_rate=0.001
    )
    
    if model_path:
        print(f"\n✅ 訓練成功完成！")
        print(f"📁 模型文件: {model_path}")
        print(f"🔓 模型未加密，可直接在其他程式中使用")
        print(f"\n使用方法:")
        print(f"import torch")
        print(f"model_data = torch.load('{model_path}')")
        print(f"model_state = model_data['model_state_dict']")
    else:
        print("❌ 訓練失敗")

if __name__ == "__main__":
    main()