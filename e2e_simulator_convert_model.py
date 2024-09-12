import ktc
import numpy as np
import onnx
from PIL import Image
import os
from os import walk

images_dir = "/mnt/kneron/image_ddata"
image_path = "/mnt/kneron/image_ddata/002.jpg"



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


# onnx_model_path = "/docker_mount/optimized_MobileNetV2.onnx"
# onnx_model_path = "/data1/MobileNetV2_optimized.onnx"
onnx_model_path = "/mnt/kneron/optimized_MobileNetV2.onnx"


print_onnx_layers(onnx_model_path)

input_names = check_onnx_model_input_names(onnx_model_path)

opt_onnx = onnx.load("/mnt/kneron/optimized_MobileNetV2.onnx")

km = ktc.ModelConfig(32770, "8b28", "720", onnx_model=opt_onnx)
eval_result = km.evaluate()
print("\nNpu performance evaluation result:\n" + eval_result)

input_data = preprocess(image_path)
inf_results = ktc.kneron_inference([input_data], onnx_file=onnx_model_path, input_names=["input.1"])

raw_images = os.listdir(images_dir)
input_images = [preprocess(images_dir + image_name) for image_name in raw_images]


input_mapping = {"input.1": input_images}
bie_model_path = km.analysis(input_mapping, threads=4)

fixed_results = ktc.kneron_inference(input_data, bie_file=bie_model_path, input_names=["input.1"], platform=720)

nef_model_path = ktc.compile([km])

hw_results = ktc.kneron_inference([input_data], nef_file=nef_model_path, input_names=["input.1"], platform=720)