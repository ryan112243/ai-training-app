"""
AI訓練項目配置文件
包含模型參數、訓練設置和數據路徑等配置
"""

import torch
import os

class Config:
    # 設備配置
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 數據配置
    DATA_DIR = 'data'
    TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'train')
    VAL_DATA_PATH = os.path.join(DATA_DIR, 'val')
    TEST_DATA_PATH = os.path.join(DATA_DIR, 'test')
    
    # 模型配置
    MODEL_NAME = 'SimpleNN'
    INPUT_SIZE = 784  # 28x28 for MNIST-like data
    HIDDEN_SIZE = 128
    OUTPUT_SIZE = 10  # 10 classes
    DROPOUT_RATE = 0.2
    
    # 訓練配置
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    WEIGHT_DECAY = 1e-4
    
    # 保存配置
    MODEL_SAVE_DIR = 'models'
    CHECKPOINT_DIR = 'checkpoints'
    LOG_DIR = 'logs'
    
    # 早停配置
    EARLY_STOPPING_PATIENCE = 10
    MIN_DELTA = 0.001
    
    # 數據增強配置
    USE_DATA_AUGMENTATION = True
    ROTATION_DEGREES = 10
    BRIGHTNESS_FACTOR = 0.2
    CONTRAST_FACTOR = 0.2
    
    # Web界面配置
    WEB_HOST = '127.0.0.1'
    WEB_PORT = 5000
    WEB_DEBUG = True

# 為了向後兼容，定義全局變量
DATA_DIR = Config.DATA_DIR
MODELS_DIR = Config.MODEL_SAVE_DIR
PLOTS_DIR = 'plots'
LOGS_DIR = Config.LOG_DIR

# 創建必要的目錄
def create_directories():
    """創建項目所需的目錄"""
    dirs = [
        Config.DATA_DIR,
        Config.TRAIN_DATA_PATH,
        Config.VAL_DATA_PATH,
        Config.TEST_DATA_PATH,
        Config.MODEL_SAVE_DIR,
        Config.CHECKPOINT_DIR,
        Config.LOG_DIR,
        PLOTS_DIR  # 添加plots目錄
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"目錄已創建: {dir_path}")

if __name__ == "__main__":
    create_directories()
    print(f"使用設備: {Config.DEVICE}")
    print("配置初始化完成！")