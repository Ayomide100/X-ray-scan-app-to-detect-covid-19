import json
import torch
from common import get_model, get_tensor

model = get_model()

def diagnosis_type(image_bytes):
	classes = ('corona', 'normal')
	tensor = get_tensor(image_bytes)
	outputs = model(tensor)
	_, prediction = torch.max(outputs, 1)
	category = prediction.item()
	name = classes[category]
	return  name