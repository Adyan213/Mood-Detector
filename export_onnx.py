# Run once on your laptop, in the folder that has best_model.pth:
#   pip install onnx onnxscript
#   python export_onnx.py
# Then keep the generated model.onnx next to index.html.
import os
import onnx
import torch
import torch.nn as nn
from torchvision import models

model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, 7)
model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
model.eval()

torch.onnx.export(
    model, torch.randn(1, 3, 224, 224), "model.onnx",
    input_names=["input"], output_names=["logits"], opset_version=13,
)

# Newer PyTorch versions put the weights in a separate model.onnx.data file,
# which the browser can't load. Merge everything into one model.onnx.
merged = onnx.load("model.onnx")
onnx.save_model(merged, "model.onnx", save_as_external_data=False)
if os.path.exists("model.onnx.data"):
    os.remove("model.onnx.data")

print("Saved model.onnx (%.1f MB)" % (os.path.getsize("model.onnx") / 1e6))
