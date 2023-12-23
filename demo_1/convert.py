from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import joblib

# load model with weights
model = joblib.load("output/model.pkl")

# create onnx model
initial_type = [('float_input', FloatTensorType([None, 4]))]
onx = convert_sklearn(model, initial_types=initial_type)

# save onnx model
with open("output/model.onnx", "wb") as f:
    f.write(onx.SerializeToString())