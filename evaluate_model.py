import onnx
import ktc

optimized_model_path = "/docker_mount/optimized_MobileNetV2.onnx"
optimized_model = onnx.load(optimized_model_path)

km = ktc.ModelConfig(32769, "8b28", "720", onnx_model=optimized_model)

eval_result = km.evaluate(output_dir="/docker_mount/evaluation_results")

print(f"IP Evaluation Result:\n{eval_result}")
