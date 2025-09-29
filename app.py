"""
AI訓練項目Web界面
使用Flask提供模型訓練、預測和管理的Web界面
"""

from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
import torch
import numpy as np
import os
import json
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from PIL import Image
import threading
import time

from config import Config
from trainer import Trainer
from models import get_model
from data_loader import preprocess_image

app = Flask(__name__)
app.secret_key = 'ai_training_secret_key'

# 全局變量
current_trainer = None
training_status = {
    'is_training': False,
    'current_epoch': 0,
    'total_epochs': 0,
    'train_loss': 0.0,
    'val_loss': 0.0,
    'train_acc': 0.0,
    'val_acc': 0.0,
    'message': '準備就緒'
}

@app.route('/')
def index():
    """主頁"""
    return render_template('index.html')

@app.route('/train')
def train_page():
    """訓練頁面"""
    return render_template('train.html')

@app.route('/predict')
def predict_page():
    """預測頁面"""
    return render_template('predict.html')

@app.route('/models')
def models_page():
    """模型管理頁面"""
    # 獲取已保存的模型列表
    model_files = []
    if os.path.exists(Config.MODEL_SAVE_DIR):
        for file in os.listdir(Config.MODEL_SAVE_DIR):
            if file.endswith('.pth'):
                filepath = os.path.join(Config.MODEL_SAVE_DIR, file)
                size = os.path.getsize(filepath)
                mtime = os.path.getmtime(filepath)
                model_files.append({
                    'name': file,
                    'size': f"{size / 1024 / 1024:.2f} MB",
                    'modified': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))
                })
    
    return render_template('models.html', models=model_files)

@app.route('/api/start_training', methods=['POST'])
def start_training():
    """開始訓練API"""
    global current_trainer, training_status
    
    if training_status['is_training']:
        return jsonify({'success': False, 'message': '訓練正在進行中'})
    
    try:
        data = request.get_json()
        model_name = data.get('model_name', 'SimpleNN')
        epochs = int(data.get('epochs', 10))
        learning_rate = float(data.get('learning_rate', Config.LEARNING_RATE))
        batch_size = int(data.get('batch_size', Config.BATCH_SIZE))
        
        # 更新配置
        Config.LEARNING_RATE = learning_rate
        Config.BATCH_SIZE = batch_size
        
        # 創建訓練器
        current_trainer = Trainer(model_name=model_name)
        
        # 重置訓練狀態
        training_status.update({
            'is_training': True,
            'current_epoch': 0,
            'total_epochs': epochs,
            'train_loss': 0.0,
            'val_loss': 0.0,
            'train_acc': 0.0,
            'val_acc': 0.0,
            'message': '開始訓練...'
        })
        
        # 在新線程中開始訓練
        training_thread = threading.Thread(
            target=train_model_async, 
            args=(current_trainer, epochs)
        )
        training_thread.daemon = True
        training_thread.start()
        
        return jsonify({'success': True, 'message': '訓練已開始'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'訓練啟動失敗: {str(e)}'})

def train_model_async(trainer, epochs):
    """異步訓練模型"""
    global training_status
    
    try:
        # 獲取數據加載器
        try:
            train_loader, val_loader, test_loader = trainer.data_processor.get_data_loaders()
        except:
            train_loader, val_loader = trainer.data_processor.load_mnist_data()
            test_loader = val_loader
        
        # 訓練循環
        for epoch in range(epochs):
            training_status['current_epoch'] = epoch + 1
            training_status['message'] = f'訓練第 {epoch + 1} epoch...'
            
            # 訓練一個epoch
            train_loss, train_acc = trainer.train_epoch(train_loader)
            val_loss, val_acc = trainer.validate_epoch(val_loader)
            
            # 更新狀態
            training_status.update({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc
            })
            
            # 記錄
            trainer.train_losses.append(train_loss)
            trainer.val_losses.append(val_loss)
            trainer.train_accuracies.append(train_acc)
            trainer.val_accuracies.append(val_acc)
            
            # 保存檢查點
            if (epoch + 1) % 5 == 0:
                trainer.save_model(f'checkpoint_epoch_{epoch + 1}.pth')
        
        # 保存最終模型
        trainer.save_model('final_model.pth')
        
        training_status.update({
            'is_training': False,
            'message': '訓練完成！'
        })
        
    except Exception as e:
        training_status.update({
            'is_training': False,
            'message': f'訓練失敗: {str(e)}'
        })

@app.route('/api/training_status')
def get_training_status():
    """獲取訓練狀態API"""
    return jsonify(training_status)

@app.route('/api/predict', methods=['POST'])
def predict_api():
    """預測API"""
    try:
        # 檢查是否有上傳的文件
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': '沒有上傳文件'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'message': '沒有選擇文件'})
        
        # 加載模型
        model_name = request.form.get('model_name', 'SimpleNN')
        model_file = request.form.get('model_file', 'best_model.pth')
        
        # 創建預測器
        predictor = Trainer(model_name=model_name)
        predictor.load_model(model_file)
        
        # 處理圖像
        if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # 保存臨時文件
            temp_path = os.path.join('temp', file.filename)
            os.makedirs('temp', exist_ok=True)
            file.save(temp_path)
            
            # 預處理圖像
            processed_image = preprocess_image(temp_path)
            
            # 預測
            predictions, probabilities = predictor.predict(processed_image)
            
            # 清理臨時文件
            os.remove(temp_path)
            
            # 準備結果
            result = {
                'prediction': int(predictions[0]),
                'probabilities': probabilities[0].tolist(),
                'confidence': float(np.max(probabilities[0]))
            }
            
            return jsonify({'success': True, 'result': result})
        
        else:
            return jsonify({'success': False, 'message': '不支持的文件格式'})
            
    except Exception as e:
        return jsonify({'success': False, 'message': f'預測失敗: {str(e)}'})

@app.route('/api/get_training_plot')
def get_training_plot():
    """獲取訓練曲線圖"""
    global current_trainer
    
    if current_trainer is None or not current_trainer.train_losses:
        return jsonify({'success': False, 'message': '沒有訓練數據'})
    
    try:
        # 創建圖表
        plt.figure(figsize=(12, 4))
        
        # 損失曲線
        plt.subplot(1, 2, 1)
        plt.plot(current_trainer.train_losses, label='訓練損失')
        if current_trainer.val_losses:
            plt.plot(current_trainer.val_losses, label='驗證損失')
        plt.title('損失曲線')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # 準確率曲線
        plt.subplot(1, 2, 2)
        plt.plot(current_trainer.train_accuracies, label='訓練準確率')
        if current_trainer.val_accuracies:
            plt.plot(current_trainer.val_accuracies, label='驗證準確率')
        plt.title('準確率曲線')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # 保存到內存
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
        img_buffer.seek(0)
        
        # 轉換為base64
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return jsonify({
            'success': True, 
            'image': f'data:image/png;base64,{img_base64}'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'生成圖表失敗: {str(e)}'})

@app.route('/api/delete_model', methods=['POST'])
def delete_model():
    """刪除模型API"""
    try:
        data = request.get_json()
        model_name = data.get('model_name')
        
        if not model_name:
            return jsonify({'success': False, 'message': '模型名稱不能為空'})
        
        model_path = os.path.join(Config.MODEL_SAVE_DIR, model_name)
        
        if os.path.exists(model_path):
            os.remove(model_path)
            return jsonify({'success': True, 'message': '模型已刪除'})
        else:
            return jsonify({'success': False, 'message': '模型文件不存在'})
            
    except Exception as e:
        return jsonify({'success': False, 'message': f'刪除失敗: {str(e)}'})

# 多領域訓練路由
@app.route('/train_multi_domain', methods=['POST'])
def train_multi_domain():
    """多領域訓練接口"""
    try:
        data = request.get_json()
        
        # 獲取訓練參數
        data_paths = data.get('data_paths', {})
        epochs = data.get('epochs', 10)
        batch_size = data.get('batch_size', 32)
        model_type = data.get('model_type', 'SimpleNN')
        
        if not data_paths:
            return jsonify({
                'success': False,
                'message': '請提供數據路徑'
            })
        
        # 創建訓練器
        trainer = Trainer(model_name=model_type, input_size=5, num_classes=5)
        
        # 開始多領域訓練
        trainer.train_multi_domain(
            data_paths=data_paths,
            epochs=epochs,
            batch_size=batch_size
        )
        
        return jsonify({
            'success': True,
            'message': '多領域訓練完成',
            'train_losses': trainer.train_losses,
            'val_losses': trainer.val_losses,
            'train_accuracies': trainer.train_accuracies,
            'val_accuracies': trainer.val_accuracies
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'多領域訓練失敗: {str(e)}'
        })

# 創建示例數據路由
@app.route('/create_sample_data', methods=['POST'])
def create_sample_data():
    """創建示例數據"""
    try:
        from multi_domain_loader import MultiDomainDataProcessor
        
        processor = MultiDomainDataProcessor()
        processor.create_sample_data()
        
        return jsonify({
            'success': True,
            'message': '示例數據創建成功',
            'data_paths': {
                'math': 'data/samples/math_sample.json',
                'programming': 'data/samples/programming_sample.json',
                'dialogue': 'data/samples/dialogue_sample.json'
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'創建示例數據失敗: {str(e)}'
        })

# 數據集信息路由
@app.route('/dataset_info')
def dataset_info():
    """數據集資訊頁面"""
    return render_template('dataset_info.html')

# 聊天功能已移至獨立的聊天應用

# 多領域訓練相關的輔助函數
def get_available_datasets():
    """獲取可用的數據集列表"""
    datasets = []
    data_dir = os.path.join(os.getcwd(), 'data', 'samples')
    
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if file.endswith('.json'):
                dataset_name = file.replace('_sample.json', '')
                datasets.append({
                    'name': dataset_name,
                    'file': file,
                    'path': os.path.join(data_dir, file)
                })
    
    return datasets

if __name__ == '__main__':
    # 創建必要目錄
    from config import create_directories
    create_directories()
    
    print("AI訓練Web界面啟動中...")
    
    # 從環境變數獲取配置，如果沒有則使用默認值
    import os
    host = os.environ.get('WEB_HOST', Config.WEB_HOST)
    port = int(os.environ.get('PORT', Config.WEB_PORT))
    debug = os.environ.get('WEB_DEBUG', str(Config.WEB_DEBUG)).lower() == 'true'
    
    print(f"訪問地址: http://{host}:{port}")
    
    app.run(
        host=host,
        port=port,
        debug=debug
    )