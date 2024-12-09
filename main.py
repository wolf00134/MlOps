from flask import Flask, render_template
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# mlflow server --host 127.0.0.1 --port 8080

# Tạo ứng dụng Flask
app = Flask(__name__)

# Tải mô hình tốt nhất đã lưu
best_model = joblib.load('./mlruns/model.pkl')
print(best_model)

# Tạo dữ liệu kiểm thử giả
X, y = make_classification(
    # same as the previous section
    n_samples=1000, n_features=5, n_informative=3, n_classes=2,
    # flip_y - high value to add more noise
    flip_y=0.1,
    # class_sep - low value to reduce space between classes
    class_sep=0.5
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Dự đoán từ mô hình tốt nhất
y_pred = best_model.predict(X_test)

# Tính toán các chỉ số đánh giá
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

@app.route('/')
def home():
    # Lấy thông tin mô hình
    model_name = best_model.__class__.__name__
    model_params = best_model.get_params()

    return render_template('index.html', 
        model_name=model_name, 
        model_params=model_params, 
        accuracy=accuracy, 
        precision=precision, 
        recall=recall, 
        f1=f1)

if __name__ == '__main__':
    app.run(debug=True)
