Automated Detection of Intracranial Hemorrhage from CT Scans
This repository contains the final project for the CS-7830 Machine Learning course at Wright State University, completed by Diwakar Khadka, Anbesha Thapa, and Barsha Thapa under the guidance of Professor Tanvi Banerjee.

Final Presentation
A complete video walkthrough of our project, methodology, and results can be found here:
https://www.youtube.com/watch?v=g-zDToDl_20

Project Overview
This project explores the development and evaluation of machine learning models to automatically detect signs of intracranial hemorrhage from brain CT scans. The goal was to build a reliable tool that could assist radiologists by flagging potentially hemorrhagic slices, thereby speeding up diagnosis in critical situations.

We systematically built and compared a range of models, from simple baselines to advanced deep learning architectures. Our final, best-performing model is an Ensemble of a custom CNN and a fine-tuned VGG16 model, which achieved an F1-Score of 0.82 for hemorrhage detection.

Key Results
The following chart summarizes the performance of all models tested. The F1-Score for the "Hemorrhage" class was used as the primary metric for comparison.

Model

F1-Score (Hemorrhage)

Precision

Recall

Logistic Regression

0.75

0.76

0.74

Multi-Layer Perceptron (MLP)

0.62

0.77

0.51

Custom CNN

0.78

0.77

0.79

Fine-Tuned VGG16

0.74

0.74

0.74

Ensemble Model (Best)

0.82

0.83

0.81

How to Run This Project
1. Prerequisites
Python 3.9+

Git

2. Clone the Repository
git clone https://github.com/khadkadiwakar/Intercranial-Hemorrhage-Detection-
cd Intracranial-Hemorrhage-Detection

3. Install Dependencies
Install all the required Python packages using the requirements.txt file.

pip install -r requirements.txt

4. Download the Dataset
Download the dataset from the link below and place the contents (Patients_CT folder, hemorrhage_diagnosis.csv, etc.) into the data/raw/ directory.

Dataset: Computed Tomography Images for Intracranial Hemorrhage Detection, Link : https://www.kaggle.com/datasets/vbookshelf/computed-tomography-ct-images

5. Run the Data Pipeline
The following scripts must be run in order to prepare the data for training.

a. Preprocess the Data
This script will resize and normalize the raw images, saving the processed versions into data/processed/.

python src/data/preprocess.py

b. Split the Data
This script will split the processed data into training, validation, and test sets.

python src/data/split.py

6. Explore the Models
All model development, training, and evaluation were conducted in the Jupyter Notebook. To see our experiments and results, open and run the cells in:
notebooks/2.0-model-prototyping.ipynb

Project Structure
├── data/
│   ├── raw/          # Original dataset files
│   └── processed/    # Processed images and data splits
├── notebooks/
│   ├── 1.0-data-exploration.ipynb
│   └── 2.0-model-prototyping.ipynb
├── src/
│   ├── data/
│   │   ├── preprocess.py
│   │   └── split.py
│   └── utils/
│       └── evaluation.py
├── assets/           # Saved plots and visuals for presentation
├── reports/          # Final presentation slides
├── requirements.txt  # Project dependencies
└── README.md         # This file

Dataset Citation
This project would not have been possible without the publicly available dataset provided by Hssayeni et al.

Hssayeni, M. (2020). Computed Tomography Images for Intracranial Hemorrhage Detection and Segmentation, Version 2. [Dataset]. PhysioNet. https://doi.org/10.13026/42e8-0x57.

License
This project is licensed under the MIT License. See the LICENSE file for details.
