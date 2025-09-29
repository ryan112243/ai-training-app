# AI模型訓練平台

一個專注於AI模型訓練和管理的Web應用程序，支持多種神經網絡架構和數據集。

## 功能特色

- 🧠 **多種模型架構**：支持CNN、ResNet、DenseNet等
- 📊 **實時監控**：訓練過程可視化和性能監控
- 🎯 **智能預測**：訓練完成後的模型預測功能
- 📈 **數據分析**：訓練曲線和性能指標分析
- 🔧 **模型管理**：模型保存、載入和刪除

## 支持的數據集

- **MATH**：數學問題數據集
- **Google DeepMind Mathematics**：高級數學推理
- **The Stack**：程式碼數據集
- **CodeContests**：程式設計競賽
- **TACO**：對話數據集
- **ConvAI2**：對話AI數據集

## 支持的模型架構

- **CNN**：卷積神經網絡
- **ResNet**：殘差網絡
- **DenseNet**：密集連接網絡
- **Multi-Domain**：多域學習模型

## 本地運行

```bash
# 安裝依賴
pip install -r requirements.txt

# 啟動應用
python app.py
```

訪問 http://localhost:5000

## 多域訓練

支持同時訓練多個領域的AI模型：

```bash
# 訓練多域模型
python train_multi_domain.py --epochs 10 --batch_size 32 --learning_rate 0.001
```

## 部署到Render

1. 將代碼推送到GitHub
2. 在Render中創建新的Web Service
3. 連接GitHub倉庫
4. Render會自動使用render.yaml配置進行部署

## 技術棧

- **後端**: Flask, Python
- **前端**: HTML5, CSS3, JavaScript, Bootstrap 5
- **機器學習**: PyTorch, NumPy
- **可視化**: Matplotlib
- **部署**: Render, Gunicorn

## API端點

- `POST /api/start_training` - 開始訓練
- `GET /api/training_status` - 獲取訓練狀態
- `POST /api/predict` - 模型預測
- `GET /api/get_training_plot` - 獲取訓練圖表
- `POST /api/delete_model` - 刪除模型
- `POST /train_multi_domain` - 多域訓練
- `POST /create_sample_data` - 創建樣本數據