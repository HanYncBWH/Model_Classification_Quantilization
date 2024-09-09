import ktc
import numpy as np
import onnx
from PIL import Image
import os


def check_onnx_model_input_names(onnx_file_path):
    model = onnx.load(onnx_file_path)
    print("Model Input Names:")
    for input in model.graph.input:
        print(f"Input name: {input.name}")
    return [input.name for input in model.graph.input]


def preprocess(input_file):
    image = Image.open(input_file)
    image = image.convert("RGB")
    img_data = np.array(image.resize((224, 224), Image.BILINEAR)) / 255
    img_data = np.transpose(img_data, (2, 0, 1))
    img_data = np.expand_dims(img_data, 0)
    return img_data


def print_onnx_layers(onnx_file_path):
    model = onnx.load(onnx_file_path)
    print("ONNX Model Layers:")
    for i, node in enumerate(model.graph.node):
        print(f"Layer {i}: {node.op_type}, Inputs: {node.input}, Outputs: {node.output}")


onnx_model_path = "/docker_mount/optimized_MobileNetV2.onnx"

print_onnx_layers(onnx_model_path)

input_names = check_onnx_model_input_names(onnx_model_path)

image_folder_path = "/docker_mount/image_ddata"

for image_file in os.listdir(image_folder_path):
    if image_file.startswith('._'):
        continue
    if image_file.endswith(('.jpeg', '.jpg', '.png', '.webp')):
        image_path = os.path.join(image_folder_path, image_file)
        print(f"Processing image: {image_path}")
        
        try:
            input_data = [preprocess(image_path)]
            inf_results = ktc.kneron_inference(input_data, onnx_file=onnx_model_path, input_names=input_names)
            
            print(f"Inference Results for {image_file}: {inf_results}")
        
        except Exception as e:
            print(f"Error occurred while processing {image_file}: {str(e)}")
