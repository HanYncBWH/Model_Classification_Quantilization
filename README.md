說明:
步驟一： CLIP onnx opset 18 模型下載網址：https://drive.google.com/file/d/1TiSM0k6EP9wi1bpDgLM-JUYtMc0D6CL1/view?usp=sharing
步驟二：image_ddata下載網址: https://drive.google.com/drive/folders/1jdmhlpNxYAb5wZXF-R_HOtPU8hNv9wos?usp=drive_link
步驟三：CLIP模型和image_ddata請放到同一個資料夾並放到
步驟四：使用“docker run --rm -it -v "你放CLIP onnx和圖片image_ddata的路徑:/docker_mount" -v "你放optimize_onnx2NEF.py的路徑（不包含optimize_onnx2NEF.py):/scripts" kneron/toolchain:latest”
ex:docker run --rm -it -v "/Volumes/One Touch/Kneron:/docker_mount" -v "/Users/hanyanc/Documents/ppthon/python-training:/scripts" kneron/toolchain:latest
步驟五：使用cd /docker_mount和ls -l去檢查有沒有掛載到docker裡面。正確的話，話應該要透過ls成功顯示“image_ddata"、"ViT-B-32_0827.onnx"和"optimize_onnx2NEF.py”。
步驟六：接著輸入“python /docker_mount/optimize_onnx2NEF.py”(如果你cd沒有在mount裡面則是“python optimize_onnx2NEF.py”)，即可跑optimize_onnx2NEF.py這份程式碼。
