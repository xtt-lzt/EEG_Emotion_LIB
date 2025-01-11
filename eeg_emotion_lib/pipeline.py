import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from eegemotionlib.data_loader import load_eeg_data  # 假设你有一个数据加载模块

class EEGEmotionPipeline:
    def __init__(self, model=None):
        """
        初始化 EEG 情感识别管道。
        
        :param model: 使用的机器学习模型，默认为 SVM。
        """
        if model is None:
            self.model = SVC(kernel='linear')
        else:
            self.model = model
        
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),  # 数据标准化
            ('classifier', self.model)     # 分类器
        ])

    def load_data(self, data_path):
        """
        加载 EEG 数据。
        
        :param data_path: 数据路径。
        :return: X (特征), y (标签)
        """
        self.X, self.y = load_eeg_data(data_path)
        return self.X, self.y

    def preprocess_data(self, test_size=0.2, random_state=42):
        """
        数据预处理，包括划分训练集和测试集。
        
        :param test_size: 测试集比例。
        :param random_state: 随机种子。
        :return: X_train, X_test, y_train, y_test
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        return self.X_train, self.X_test, self.y_train, self.y_test

    def train(self):
        """
        训练模型。
        """
        self.pipeline.fit(self.X_train, self.y_train)

    def evaluate(self):
        """
        评估模型性能。
        """
        y_pred = self.pipeline.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred)
        
        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(report)

    def save_model(self, model_path):
        """
        保存训练好的模型。
        
        :param model_path: 模型保存路径。
        """
        import joblib
        joblib.dump(self.pipeline, model_path)

    def load_model(self, model_path):
        """
        加载已训练的模型。
        
        :param model_path: 模型路径。
        """
        import joblib
        self.pipeline = joblib.load(model_path)

    def predict(self, X):
        """
        使用训练好的模型进行预测。
        
        :param X: 输入数据。
        :return: 预测结果。
        """
        return self.pipeline.predict(X)

# 示例用法
if __name__ == "__main__":
    pipeline = EEGEmotionPipeline()
    
    # 加载数据
    X, y = pipeline.load_data("path/to/eeg_data.csv")
    
    # 数据预处理
    X_train, X_test, y_train, y_test = pipeline.preprocess_data()
    
    # 训练模型
    pipeline.train()
    
    # 评估模型
    pipeline.evaluate()
    
    # 保存模型
    pipeline.save_model("eeg_emotion_model.pkl")
    
    # 加载模型并进行预测
    pipeline.load_model("eeg_emotion_model.pkl")
    predictions = pipeline.predict(X_test)
    print("Predictions:", predictions)