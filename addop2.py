import numpy as np
import ktc

nef_model_path = "/docker_mount/models_720.nef"
image_path = "/docker_mount/img_align_celeba/000001.jpg"

def preprocess(input_file):
    from PIL import Image
    image = Image.open(input_file)
    image = image.convert("RGB")
    img_data = np.array(image.resize((160, 160), Image.BILINEAR)) / 255
    img_data = np.transpose(img_data, (2, 0, 1))
    img_data = np.expand_dims(img_data, 0)
    return img_data

input_data = preprocess(image_path)

input_names = ["input.1"]
hw_results = ktc.kneron_inference([input_data], nef_file=nef_model_path, input_names=input_names, platform=720)

# inference的output張量
output_tensor = hw_results[0]

# 保存inference result成.npy 
np.save('/docker_mount/output_tensor.npy', output_tensor)
