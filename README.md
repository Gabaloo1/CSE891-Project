# VQA with Multimodal Transformers

This repo contains the dataset & code for exploring multimodal *fusion-type* transformer models for the task of visual question answering.

## 🗂️ Dataset Used: [DAQUAR Dataset](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/vision-and-language/visual-turing-challenge/)

Based on the work [VQA-With-Multimodal-Transformers](https://github.com/tezansahu/VQA-With-Multimodal-Transformers).

## Requirements

Python 3.8 is required to run the code.

Create a virtual environment & install the required packages using `pip install -r requirements.txt`:

Install `torch` separetely using the following command (this will enable GPU support):
```bash
pip install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111
```

## Running the Code

The `src/` folder contains all the scripts necessary for data processing & model training. All the configs & hyperparameters are specified in the `params.yaml` file.

### Model Training

In order to train and evaluate the model, run the following command. Note that the `params.yaml` file contains the necessary hyperparameters for training the model.

```bash
python src/main.py --config=params.yaml
```

Model training will be done on the GPU if available. The model will be saved in the `checkpoints/` directory and the metrics will be saved in the `metrics/` directory.

_Note: Training took around 2h20min on a single RTX 3070Ti GPU._

### Inference

For inferencing, run `python src/inference.py --config=params.yaml --img_path=<path-to-image> --question=<question>`

## Results

The training process was done using the following transformer models, retrieved from the HuggingFace model hub:

- Text Transformers (for encoding questions):
    - RoBERTa (Robustly Optimized BERT Pretraining Approach): `'roberta-base'`
- Image Transformers (for encoding images):
    - BEiT (Bidirectional Encoder representation from Image Transformers): `'microsoft/beit-base-patch16-224-pt22k-ft22k'`

The model achieved a Wu & Palmer similarity score of 0.308, an accuracy of 0.261, and an F1 score of 0.033. The model has 211M trainable parameters.


