"""
多領域數據加載器
支持數學、程式設計、對話、寫作、模擬聯合國等多種數據類型的統一加載和處理
"""

import torch
from torch.utils.data import Dataset, DataLoader
import json
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple, Any
import re
from config import Config

class MultiDomainDataset(Dataset):
    """多領域統一數據集類"""
    
    def __init__(self, data_path: str, domain_type: str = "auto", max_length: int = 512):
        """
        初始化多領域數據集
        
        Args:
            data_path: 數據文件路徑
            domain_type: 數據領域類型 ("math", "programming", "dialogue", "writing", "mun", "auto")
            max_length: 文本最大長度
        """
        self.data_path = data_path
        self.domain_type = domain_type
        self.max_length = max_length
        self.data = []
        
        self.load_data()
    
    def load_data(self):
        """加載數據"""
        if not os.path.exists(self.data_path):
            print(f"警告: 數據路徑 {self.data_path} 不存在")
            return
        
        # 根據文件擴展名選擇加載方式
        if self.data_path.endswith('.json'):
            self._load_json_data()
        elif self.data_path.endswith('.csv'):
            self._load_csv_data()
        elif self.data_path.endswith('.txt'):
            self._load_text_data()
        else:
            print(f"不支持的文件格式: {self.data_path}")
    
    def _load_json_data(self):
        """加載JSON格式數據"""
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            if isinstance(raw_data, list):
                for item in raw_data:
                    processed_item = self._process_item(item)
                    if processed_item:
                        self.data.append(processed_item)
            elif isinstance(raw_data, dict):
                processed_item = self._process_item(raw_data)
                if processed_item:
                    self.data.append(processed_item)
                    
        except Exception as e:
            print(f"加載JSON數據時出錯: {e}")
    
    def _load_csv_data(self):
        """加載CSV格式數據"""
        try:
            df = pd.read_csv(self.data_path)
            
            for _, row in df.iterrows():
                item = {
                    'input': str(row.get('input', row.get('question', ''))),
                    'output': str(row.get('output', row.get('answer', ''))),
                    'type': self.domain_type if self.domain_type != "auto" else row.get('type', 'general')
                }
                
                processed_item = self._process_item(item)
                if processed_item:
                    self.data.append(processed_item)
                    
        except Exception as e:
            print(f"加載CSV數據時出錯: {e}")
    
    def _load_text_data(self):
        """加載純文本數據"""
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 假設每兩行為一對（問題-答案）
            for i in range(0, len(lines)-1, 2):
                item = {
                    'input': lines[i].strip(),
                    'output': lines[i+1].strip() if i+1 < len(lines) else '',
                    'type': self.domain_type if self.domain_type != "auto" else 'general'
                }
                
                processed_item = self._process_item(item)
                if processed_item:
                    self.data.append(processed_item)
                    
        except Exception as e:
            print(f"加載文本數據時出錯: {e}")
    
    def _process_item(self, item: Dict) -> Dict:
        """處理單個數據項"""
        try:
            # 提取輸入和輸出
            input_text = str(item.get('input', item.get('question', item.get('problem', ''))))
            output_text = str(item.get('output', item.get('answer', item.get('solution', ''))))
            
            if not input_text or not output_text:
                return None
            
            # 自動檢測數據類型
            if self.domain_type == "auto":
                detected_type = self._detect_domain_type(input_text, output_text)
            else:
                detected_type = self.domain_type
            
            # 根據領域類型進行特殊處理
            processed_input = self._preprocess_by_domain(input_text, detected_type)
            processed_output = self._preprocess_by_domain(output_text, detected_type)
            
            return {
                'input': processed_input,
                'output': processed_output,
                'type': detected_type,
                'metadata': item.get('metadata', {})
            }
            
        except Exception as e:
            print(f"處理數據項時出錯: {e}")
            return None
    
    def _detect_domain_type(self, input_text: str, output_text: str) -> str:
        """自動檢測數據領域類型"""
        text = (input_text + " " + output_text).lower()
        
        # 數學關鍵詞
        math_keywords = ['solve', 'equation', 'calculate', '計算', '解', '數學', 'math', 'algebra', 'geometry']
        if any(keyword in text for keyword in math_keywords):
            return 'math'
        
        # 程式設計關鍵詞
        programming_keywords = ['def ', 'function', 'class ', 'import', 'return', '程式', '代碼', 'code', 'algorithm']
        if any(keyword in text for keyword in programming_keywords):
            return 'programming'
        
        # 對話關鍵詞
        dialogue_keywords = ['hello', 'hi', 'how are you', '你好', '對話', 'conversation', 'chat']
        if any(keyword in text for keyword in dialogue_keywords):
            return 'dialogue'
        
        # 寫作關鍵詞
        writing_keywords = ['essay', 'article', 'write', '寫作', '文章', '作文', 'composition']
        if any(keyword in text for keyword in writing_keywords):
            return 'writing'
        
        # 模擬聯合國關鍵詞
        mun_keywords = ['resolution', 'delegate', 'united nations', '聯合國', '外交', 'diplomatic']
        if any(keyword in text for keyword in mun_keywords):
            return 'mun'
        
        return 'general'
    
    def _preprocess_by_domain(self, text: str, domain_type: str) -> str:
        """根據領域類型進行預處理"""
        # 基本清理
        text = text.strip()
        
        if domain_type == 'math':
            # 數學文本預處理：保留數學符號
            text = re.sub(r'\s+', ' ', text)  # 合併多個空格
            
        elif domain_type == 'programming':
            # 程式碼預處理：保留縮進和格式
            pass  # 保持原格式
            
        elif domain_type == 'dialogue':
            # 對話預處理：標準化對話格式
            text = re.sub(r'\s+', ' ', text)
            
        elif domain_type == 'writing':
            # 寫作預處理：段落格式化
            text = re.sub(r'\n\s*\n', '\n\n', text)  # 標準化段落分隔
            
        elif domain_type == 'mun':
            # 模擬聯合國預處理：正式文檔格式
            text = re.sub(r'\s+', ' ', text)
        
        # 長度限制
        if len(text) > self.max_length:
            text = text[:self.max_length]
        
        return text
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'input': item['input'],
            'output': item['output'],
            'type': item['type'],
            'metadata': item.get('metadata', {})
        }

class MultiDomainDataProcessor:
    """多領域數據處理器"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
    
    def create_dataloaders(self, 
                          data_paths: Dict[str, str], 
                          batch_size: int = 32,
                          train_split: float = 0.8) -> Dict[str, DataLoader]:
        """
        創建多領域數據加載器
        
        Args:
            data_paths: 各領域數據路徑字典 {"domain": "path"}
            batch_size: 批次大小
            train_split: 訓練集比例
            
        Returns:
            數據加載器字典
        """
        dataloaders = {}
        
        for domain, path in data_paths.items():
            if os.path.exists(path):
                # 創建數據集
                dataset = MultiDomainDataset(path, domain_type=domain)
                
                if len(dataset) == 0:
                    print(f"警告: {domain} 領域數據集為空")
                    continue
                
                # 分割訓練和驗證集
                train_size = int(len(dataset) * train_split)
                val_size = len(dataset) - train_size
                
                # 確保至少有一個樣本用於訓練和驗證
                if train_size == 0:
                    train_size = 1
                    val_size = max(0, len(dataset) - 1)
                elif val_size == 0:
                    val_size = 1
                    train_size = max(0, len(dataset) - 1)
                
                train_dataset, val_dataset = torch.utils.data.random_split(
                    dataset, [train_size, val_size]
                )
                
                # 創建數據加載器
                train_loader = DataLoader(
                    train_dataset, 
                    batch_size=batch_size, 
                    shuffle=True,
                    collate_fn=self._collate_fn
                )
                
                val_loader = DataLoader(
                    val_dataset, 
                    batch_size=batch_size, 
                    shuffle=False,
                    collate_fn=self._collate_fn
                )
                
                dataloaders[f"{domain}_train"] = train_loader
                dataloaders[f"{domain}_val"] = val_loader
                
                print(f"已創建 {domain} 領域數據加載器: 訓練集 {train_size}, 驗證集 {val_size}")
            else:
                print(f"警告: {domain} 領域數據路徑不存在: {path}")
        
        return dataloaders
    
    def _collate_fn(self, batch):
        """自定義批次整理函數"""
        inputs = [item['input'] for item in batch]
        outputs = [item['output'] for item in batch]
        types = [item['type'] for item in batch]
        metadata = [item['metadata'] for item in batch]
        
        return {
            'inputs': inputs,
            'outputs': outputs,
            'types': types,
            'metadata': metadata
        }
    
    def create_sample_data(self, output_dir: str = "data/samples"):
        """創建示例數據文件"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 數學示例數據
        math_data = [
            {
                "input": "解方程 2x + 5 = 13",
                "output": "x = 4",
                "type": "math",
                "metadata": {"difficulty": "easy", "category": "algebra"}
            },
            {
                "input": "計算 15 × 23",
                "output": "345",
                "type": "math", 
                "metadata": {"difficulty": "easy", "category": "arithmetic"}
            }
        ]
        
        # 程式設計示例數據
        programming_data = [
            {
                "input": "寫一個函數計算階乘",
                "output": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)",
                "type": "programming",
                "metadata": {"difficulty": "medium", "language": "python"}
            }
        ]
        
        # 對話示例數據
        dialogue_data = [
            {
                "input": "你好，今天天氣怎麼樣？",
                "output": "你好！今天天氣很不錯，陽光明媚，適合外出活動。",
                "type": "dialogue",
                "metadata": {"style": "friendly", "topic": "weather"}
            }
        ]
        
        # 保存示例數據
        datasets = {
            "math": math_data,
            "programming": programming_data,
            "dialogue": dialogue_data
        }
        
        for domain, data in datasets.items():
            file_path = os.path.join(output_dir, f"{domain}_sample.json")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"已創建 {domain} 示例數據: {file_path}")

# 使用示例
if __name__ == "__main__":
    # 創建數據處理器
    processor = MultiDomainDataProcessor()
    
    # 創建示例數據
    processor.create_sample_data()
    
    # 數據路徑配置
    data_paths = {
        "math": "data/samples/math_sample.json",
        "programming": "data/samples/programming_sample.json", 
        "dialogue": "data/samples/dialogue_sample.json"
    }
    
    # 創建數據加載器
    dataloaders = processor.create_dataloaders(data_paths, batch_size=2)
    
    # 測試數據加載
    for name, loader in dataloaders.items():
        print(f"\n測試 {name} 數據加載器:")
        for batch in loader:
            print(f"  批次大小: {len(batch['inputs'])}")
            print(f"  輸入示例: {batch['inputs'][0][:100]}...")
            print(f"  輸出示例: {batch['outputs'][0][:100]}...")
            break