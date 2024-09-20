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


output_tensor = hw_results[0]

# postprocess
# 1. ReduceL2
l2_norm = np.linalg.norm(output_tensor, ord=2, axis=1, keepdims=True)

# 2. Expand
expanded_l2_norm = np.broadcast_to(l2_norm, output_tensor.shape)

# 3. Div
normalized_output = output_tensor / expanded_l2_norm

# 4. Clip
final_output = np.clip(normalized_output, a_min=-10.0, a_max=10.0)

print(final_output)
