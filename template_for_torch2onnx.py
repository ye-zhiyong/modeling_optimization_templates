import torch
from model import UNet
import onnx

# load model and data
u_net = UNet(1, 2)
image_data = torch.randn(4, 1, 480, 320)

# export
torch.onnx.export(
    u_net,
    image_data,
    'model.onnx',
    export_params=True,
    opset_version=8
)

# increase dimensional information
model_file = 'model.onnx'
onnx_model = onnx.load(model_file)
onnx.save(onnx.shape_inference.infer_shapes(onnx_model), model_file)
