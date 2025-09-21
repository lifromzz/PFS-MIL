# PFS-MIL: Pathological Feature Stratification Multi-Instance Learning

This repository contains the implementation of **PFS-MIL**, a Teacher-Student based multi-instance learning framework for pathology image analysis.

## Overview

The training procedure consists of two stages:

1. **Teacher Model Training**  
   The Teacher model is first trained to learn attention scores for all instances, which are later used to guide the Student model.  

2. **Student Model Training**  
   The Student model is then trained using the attention-based stratification provided by the Teacher model, enabling more effective learning from significant and latent instances.

---

## Repository Structure
```
├── Teacher_model_train
│ └── train
│ └── 5X_CPTAC_SSL_L3.py # Script to train the Teacher model
├── Student_model_train
│ └── train
│ └── SSL_L3_CPTAC_512_1_64.py # Script to train the Student model
├── data # Path to datasets
├── utils # Utility functions
└── README.md
```

---

## Training Instructions

### 1. Train the Teacher Model

Navigate to the Teacher model training directory:
Run the training script for the Teacher model:
```
python 5X_CPTAC_SSL_L3.py
```
This will train the Teacher model and save the attention scores for all instances, which are required for the Student model.

### 2. Train the Student Model

After the Teacher model finishes training, navigate to the Student model training directory:

```
cd Student_model_train/train
```

Run the training script for the Student model:

```
python SSL_L3_CPTAC_512_1_64.py
```

The Student model will use the Teacher model’s attention scores for pathological feature stratification during training.

```bash
cd Teacher_model_train/train
