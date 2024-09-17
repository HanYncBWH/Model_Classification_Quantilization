import ktc
import numpy as np
import onnx
from PIL import Image
import os
from os import walk

# 設置圖片和模型路徑
images_dir = "/docker_mount/img_align_celeba"
image_path = "/docker_mount/img_align_celeba/000001.jpg"
onnx_model_path = "/docker_mount/FCmodel_simplified.onnx"

# 確認 ONNX 模型的輸入名稱
def check_onnx_model_input_names(onnx_file_path):
    model = onnx.load(onnx_file_path)
    print("Model Input Names:")
    for input in model.graph.input:
        print(f"Input name: {input.name}")
    return [input.name for input in model.graph.input]

# 圖像預處理函數
def preprocess(input_file):
    image = Image.open(input_file)
    image = image.convert("RGB")
    img_data = np.array(image.resize((224, 224), Image.BILINEAR)) / 255
    img_data = np.transpose(img_data, (2, 0, 1))
    img_data = np.expand_dims(img_data, 0)
    return img_data

# 打印 ONNX 模型層
def print_onnx_layers(onnx_file_path):
    model = onnx.load(onnx_file_path)
    print("ONNX Model Layers:")
    for i, node in enumerate(model.graph.node):
        print(f"Layer {i}: {node.op_type}, Inputs: {node.input}, Outputs: {node.output}")

# 打印模型結構
print_onnx_layers(onnx_model_path)

# 獲取輸入名稱
input_names = check_onnx_model_input_names(onnx_model_path)

# 加載模型
opt_onnx = onnx.load(onnx_model_path)

# 創建模型配置
km = ktc.ModelConfig(32770, "8b28", "720", onnx_model=opt_onnx)

# 評估模型
eval_result = km.evaluate()
print("\nNpu performance evaluation result:\n" + eval_result)

# 預處理單張圖片
input_data = preprocess(image_path)

# 進行推理
inf_results = ktc.kneron_inference([input_data], onnx_file=onnx_model_path, input_names=input_names)

# 預處理所有圖片
raw_images = os.listdir(images_dir)
input_images = [preprocess(os.path.join(images_dir, image_name)) for image_name in raw_images]

# 分析模型並進行優化
input_mapping = {input_names[0]: input_images}
bie_model_path = km.analysis(input_mapping, threads=4)

# 進行推理並獲取固定點結果
fixed_results = ktc.kneron_inference([input_data], bie_file=bie_model_path, input_names=input_names, platform=720)

# 編譯成 NEF 模型
nef_model_path = ktc.compile([km])

# 使用 NEF 模型進行推理
hw_results = ktc.kneron_inference([input_data], nef_file=nef_model_path, input_names=input_names, platform=720)
