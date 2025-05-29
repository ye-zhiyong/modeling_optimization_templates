import torch
from model import UNet
import onnx

#构建模型与数据
image_data = torch.randn(4, 1, 480, 320)
u_net = UNet(1, 2)

#导出
torch.onnx.export(
    u_net,
    image_data,
    'model.onnx',
    export_params=True,
    opset_version=8
)

# 增加维度信息
model_file = 'model.onnx'
onnx_model = onnx.load(model_file)
onnx.save(onnx.shape_inference.infer_shapes(onnx_model), model_file)
