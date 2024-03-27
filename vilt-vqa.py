from transformers import ViltProcessor, ViltForQuestionAnswering
import requests
from PIL import Image
import torch

# prepare image + question
image_path = "vizwiz_dataset/test/VizWiz_test_00000000.jpg"
image = Image.open(image_path).convert("RGB")
text = "What is the text?"

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# prepare inputs
encoding = processor(image, text, return_tensors="pt")

# forward pass
outputs = model(**encoding)
logits = outputs.logits
idx = logits.argmax(-1).item()
print("Predicted answer:", model.config.id2label[idx])

# Specify the directory where you want to save the model and processor
save_directory_model = "ViLT/vilt-vqa-model"
save_directory_processor = "ViLT/vilt-vqa-processor"

# Save the model
model.save_pretrained(save_directory_model)

# Save the processor
processor.save_pretrained(save_directory_processor)
print("Model and processor are Saved!")
