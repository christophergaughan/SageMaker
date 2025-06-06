> 🛡️ **Disclaimer:** This repository was created as part of my independent learning in machine learning and MLOps.  
While I consulted various public resources along the way, including open tutorials and documentation, all code and commentary in this repo were written and developed by me.  
No proprietary, paid, or licensed content has been copied or redistributed.



## Educational Notebooks

- [**Positional Encoding Explained**](Positional_Encodings.ipynb)  
  A clean walkthrough of how transformer models (like BERT) encode word positions using sinusoidal patterns — includes visualizations and code examples. Great for anyone trying to understand the inner workings of attention mechanisms.

# SageMaker ML Pipeline: NLP Classification & MLOps Platform Engineering

This repo demonstrates the full pipeline for deploying NLP models using AWS SageMaker — from data ingestion and preprocessing to model training, deployment, and inference. It’s built with real-world, cloud-native practices in mind, and designed for integration into an internal ML platform like the one described in Genmab's AI & Analytics Engineering team.

---

## Project Goals

- Classify news headlines into topics using a fine-tuned DistilBERT model
- Train and deploy the model using Amazon SageMaker
- Enable real-time inference via a containerized API (ready for SageMaker hosting)
- Showcase MLOps: containerization, CI/CD automation, infrastructure-as-code
- Document core transformer theory (positional encodings) for team enablement

---

## Tech Stack

| Layer               | Tools & Frameworks                                      |
|--------------------|----------------------------------------------------------|
| **Language**        | Python 3.10                                              |
| **Modeling**        | PyTorch, HuggingFace Transformers                        |
| **Data**            | Pandas, S3-hosted corpus (newsCorpora)                  |
| **Training**        | Custom PyTorch class, `script.py`                        |
| **Inference**       | `inference.py` using `model_fn` / `predict_fn` / etc.   |
| **Infrastructure**  | AWS SageMaker, IAM, S3                                   |
| **MLOps**           | Docker, GitHub Actions (CI), Terraform (infra)          |

---

## Core Features

### Multiclass Text Classification with DistilBERT
- Four target categories: **Business**, **Entertainment**, **Science**, **Health**
- Tokenization and padding via HuggingFace
- Dropout + fully connected classification head

### Containerized Inference
- `inference.py` defines a SageMaker-compatible model serving interface
- Supports JSON-based inputs
- Returns probabilities + predicted label

### 🐳 Docker-Ready

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "inference.py"]
```
### Infrastructure-as-Code
Provisioned via Terraform:
```
resource "aws_sagemaker_notebook_instance" "sagemaker_notebook" {
  name          = "nlp-model-notebook"
  instance_type = "ml.t2.medium"
  role_arn      = aws_iam_role.sagemaker_execution_role.arn
}
```
Also provisions IAM roles with `AmazonSageMakerFullAccess`

### CI/CD Automation
GitHub Actions included in `.github/workflows/train.yml`:

* Lint checks with flake8

* Install dependencies

* (Optional) run training or test workflow

## Educational Notebooks

- [**Positional Encoding Explained**](Positional_Encodings.ipynb)  
  Walkthrough of how transformer models encode position information using sinusoidal functions — includes visuals + annotated code.

- [**TrainingNotebook.ipynb**](TrainingNotebook.ipynb)  
  Experimental training loop with dropout tuning, batch sizes, and activation function exploration.

## Weights, Biases, and Core ML Concepts

This repo includes notebooks that explore how loss functions, optimizers, and activation functions affect model performance — especially in neural networks like DistilBERT.

For anyone looking to build rock-solid intuition on these core ideas, I highly recommend [StatQuest with Josh Starmer](https://www.youtube.com/user/joshstarmer) on YouTube.  
His explanations of **gradient descent**, **activation functions**, and **loss landscapes** are some of the clearest out there.


## Rough Folder Guide (has been moved around)
```
├── script.py                       # Core model training script
├── inference.py                   # SageMaker-style inference pipeline
├── Positional_Encodings.ipynb     # Transformer theory explainer
├── Dockerfile                     # Container config for inference
├── requirements.txt               # All Python dependencies
├── main.tf                        # Terraform infra for SageMaker
└── .github/workflows/train.yml    # CI/CD with GitHub Actions
```

## Author
**Christopher Gaughan, Ph.D.**

*Machine Learning Engineer | Biochemical Engineer | Cloud Architect*

##  License
MIT License — free to use, modify, and deploy.

[![View in nbviewer](https://img.shields.io/badge/View%20Notebook-nbviewer-orange?logo=jupyter)](https://nbviewer.org/github/christophergaughan/SageMaker/blob/main/Positional_Encodings.ipynb)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/christophergaughan/SageMaker/blob/main/Positional_Encodings.ipynb)

> ⚠️ **Note:** This repository is actively being organized to support fully runnable notebooks from end to end.  
Currently, some notebooks (e.g., `TrainingNotebook.ipynb`) may require running from specific directories or setting paths manually.  
*A streamlined execution flow and directory structure are in progress* as **time allows**.



