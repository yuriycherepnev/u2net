import onnxruntime as ort
import numpy as np

def test_onnx_model(onnx_model_path, input_tensor):
    ort_session = ort.InferenceSession(onnx_model_path)

    # Выполните предсказание
    ort_inputs = {ort_session.get_inputs()[0].name: input_tensor}
    ort_outs = ort_session.run(None, ort_inputs)

    return ort_outs

# Например, сгенерируйте тестовые данные:
dummy_input = np.random.randn(1, 3, 320, 320).astype(np.float32)

# Протестируйте ONNX модель:
output = test_onnx_model('/home/yuriy/projects/u2net/saved_models/u2net/u2net_2.onnx', dummy_input)
print(output)