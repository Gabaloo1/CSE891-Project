from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
import json, os, csv
from tqdm import tqdm
import torch

# Set the path to your test data directory
test_imgs_dir = os.path.join("vizwiz_dataset", "test")
test_data_json = os.path.join("vizwiz_dataset", "annotations", "test.json")

processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to("cuda")

# Create a list to store the results
results = []

# Read the json file
with open(test_data_json, "r") as json_file:
    data = json.load(json_file)

# Iterate over the test data
for item in tqdm(data, desc="Processing test data"):
    image_filename = item["image"]
    question = item["question"]

    # Read the corresponding image
    image_path = os.path.join(test_imgs_dir, f"{image_filename}")
    image = Image.open(image_path).convert("RGB")

    # prepare inputs
    encoding = processor(image, question, return_tensors="pt").to("cuda:0", torch.float16)

    out = model.generate(**encoding)
    generated_text = processor.decode(out[0], skip_special_tokens=True)

    results.append((image_filename, question, generated_text))

# Write the results to a CSV file
# csv_file_path = "blip/results/results-zero-shot.csv"
# with open(csv_file_path, mode="w", newline="") as csv_file:
#     csv_writer = csv.writer(csv_file)
#     csv_writer.writerow(["Image", "Question", "Answer"])  # Write header
#     csv_writer.writerows(results)

# Write the results to a JSON file [{image:..., "answer":...}, ...]
json_file_path = "blip/results/results-zero-shot.json"
with open(json_file_path, "w") as json_file:
    entries = [{"image": image, "answer": answer} for image, _, answer in results]
    json.dump(entries, json_file)

print(f"Results saved!")