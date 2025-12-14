IRIS ML Pipeline

This repository contains a complete **end-to-end MLOps pipeline** for the classic **Iris classification dataset**.  
It demonstrates how to integrate **data validation, model training, evaluation, and inference** into a version-controlled, test-driven and automated CI/CD workflow using **DVC**, **Pytest**, **GitHub Actions**, and **CML**.

---

## Project Overview

| Component | Description |
|------------|-------------|
| **Data Validation** | Ensures input dataset meets schema and quality expectations. |
| **Model Training** | Trains a Random Forest classifier on the Iris dataset. |
| **Evaluation** | Computes model accuracy on validation data. |
| **Inference** | Predicts species for new flower measurements. |
| **Version Control** | Dataset and model versioned using DVC. |
| **Testing** | Unit tests written with Pytest. |
| **CI/CD** | Automated testing and reporting with GitHub Actions + CML. |

---
