import onnx
import onnxruntime as rt
import ktc
import numpy as np
from onnxsim import simplify
from PIL import Image
import os

# Load the ONNX model
exported_clip = onnx.load('/docker_mount/ViT-B-32_0827.onnx')

# Simplify the model with inference check disabled entirely
try:
    # Simplify the model without performing inference check (set check_n=False)
    optimized_clip, check_ok = simplify(exported_clip, skip_shape_inference=True, perform_optimization=False, check_n=False)
    
    if not check_ok:
        print("Simplification did not produce a valid model, proceeding without it.")
        optimized_clip = exported_clip  # Use the original if simplification fails
except Exception as e:
    print(f"Simplification failed: {e}")
    optimized_clip = exported_clip  # Use the original model if simplification fails

# Save the model after simplification
onnx.save(optimized_clip, '/docker_mount/CLIPModel_gen_simplified.onnx')
print("Saved simplified model.")

# Proceed with the ONNX optimization flow
try:
    optimized_clip = ktc.onnx_optimizer.torch_exported_onnx_flow(optimized_clip)
    # Save the model after torch_exported_onnx_flow
    onnx.save(optimized_clip, '/docker_mount/CLIPModel_gen_after_torch_exported.onnx')
    print("Saved model after torch_exported_onnx_flow.")

    optimized_clip = ktc.onnx_optimizer.onnx2onnx_flow(optimized_clip, eliminate_tail=False, opt_matmul=False)
    # Save the model after onnx2onnx_flow
    onnx.save(optimized_clip, '/docker_mount/CLIPModel_gen_after_onnx2onnx.onnx')
    print("Saved model after onnx2onnx_flow.")
except Exception as e:
    print(f"ONNX optimization flow failed: {e}")
    optimized_clip = exported_clip  # Fallback to original if optimization fails

# Save the final optimized model
onnx.save(optimized_clip, '/docker_mount/CLIPModel_gen_final.onnx')
print("Saved final optimized model.")

# Initialize the model configuration
km = ktc.ModelConfig(32770, "8b28", "720", onnx_model=optimized_clip)

# Preprocess the image data
def preprocess_clip_model(image_folder):
    img_list = []
    for img_path in os.listdir(image_folder):
        # Load the image
        img = Image.open(os.path.join(image_folder, img_path)).convert('RGB')
        # Resize to the required input size [224, 224]
        img = img.resize((224, 224))
        # Convert the image to a numpy array and normalize to [0, 1]
        img = np.array(img).astype(np.float32) / 255.0
        # Transpose the dimensions to match the model's input [1, 3, 224, 224]
        img = np.transpose(img, (2, 0, 1))  # Change from (H, W, C) to (C, H, W)
        # Expand dimensions to add batch size [1, 3, 224, 224]
        processed_img = np.expand_dims(img, axis=0)
        img_list.append(processed_img)
    return img_list

clip_input_list = preprocess_clip_model('/docker_mount/image_ddata')

# Quantization (Fix-point analysis)
input_mapping = {"images": clip_input_list}
bie_model_path = km.analysis(input_mapping, threads=4, quantize_mode="default")
print("\nFixed-point analysis done. Save bie model to '" + str(bie_model_path) + "'")

# Compile to NEF model
nef_model_path = ktc.compile([km])
print("\nCompile done. Save Nef file to '" + str(nef_model_path) + "'")
