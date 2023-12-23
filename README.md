# seminar-demo-onnx

Cần tải dữ liệu MNIST vào thư mục data/MNIST

## Demo 1

1. Chọn môi trường
    
    ```python
    conda activate onnx_1
    ```
    
2. Khởi tạo và train model với `scikit-learn`
    
    ```python
    python train.py
    ```
    
3. Chuyển đổi model sang định dạng `.onnx`
    
    ```python
    python convert.py
    ```
    
4. Chạy inference với ONNX model
    
    Không dùng tới `scikit-learn`
    
    ```python
    python infer.py
    ```
    

## Demo 2

1. Chọn môi trường
    
    ```python
    conda activate onnx_2
    ```
    
2. Tải model về từ ONNX model zoo
    
    ```python
    wget --no-check-certificate https://www.cntk.ai/OnnxModels/mnist/opset_7/mnist.tar.gz
    ```
    
    ```python
    tar xvf mnist.tar.gz
    ```
    
3. Sử dụng model để chạy inference
    
    ```python
    python infer.py test.png
    ```
    

## Demo 3

1. Chọn môi trường
    
    ```python
    conda activate onnx_3
    ```
    
2. Khởi tạo và train mô hình bằng `torch`
    
    ```python
    python train.py
    ```
    
    Đã chạy và thu được kết quả
    
    ```python
    Test set: Average loss: 0.0304, Accuracy: 9907/10000 (99%)
    ```
    
3. Chuyển `model.pt` thành `model.onnx`
    
    ```python
    python torch2onnx.py
    ```
    
4. Chuyển từ `model.onnx` thành `model.pb`
    
    ```python
    python onnx2tf.py
    ```
