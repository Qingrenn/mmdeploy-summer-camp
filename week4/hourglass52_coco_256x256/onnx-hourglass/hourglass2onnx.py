from mmpose.models import HourglassNet
import torch

model = HourglassNet()
dummy_input = torch.rand(1, 3, 256, 256)
dummy_output = model(dummy_input)
model_script = torch.jit.script(model)
# model_trace = torch.jit.trace(model, dummy_input)

torch.onnx.export(
    model_script,
    dummy_input,
    'hourglass_script.onnx',
    example_outputs=dummy_output
)
