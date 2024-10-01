# Cassava Leaf Disease Classification
GitHub repository for the semester project on Cassava Leaf Disease Classification using deep learning models.

##### Run the code in the Jupyter Notebook (`Cassava_Leaf_Disease_Classification.ipynb`) for each step of the analysis.

## Project Overview

This project focuses on classifying cassava leaf diseases using advanced machine learning techniques and deep learning models. The dataset used is from the Kaggle Cassava Leaf Disease Classification competition, where images of cassava leaves are labeled with their corresponding disease types. The project explores various methods for data exploration, visualization, and model training to achieve high accuracy in disease classification.

### Key Features:

- **Data Preprocessing**: Includes loading the dataset, reading the labels, and performing exploratory data analysis (EDA).
- **Image Visualization**: Visualizes random images from the dataset along with their respective class labels.
- **Deep Learning Models**: Utilizes EfficientNet, a state-of-the-art convolutional neural network model, for image classification.
- **Loss Function**: Implements Bi-Tempered Logistic Loss, which is robust to label noise and imbalanced datasets.
- **Training and Validation**: The model is trained on the dataset with techniques to handle overfitting, and validation is performed using a holdout dataset.

## Methodology

The following steps were performed in the project:

1. **Data Loading and Exploration**: 
   - Kaggle competition dataset is loaded using Kaggle API.
   - A JSON file is used to map numeric labels to disease names.
   - Data is visualized using bar charts to understand the distribution of classes.

2. **Image Visualization**: 
   - A sample of images from the training dataset is visualized using OpenCV and Matplotlib.
   - The images are displayed with corresponding disease labels.

3. **Model Training**:
   - EfficientNet architecture is used for the classification task.
   - The model is compiled with an appropriate loss function and optimizer.
   - Training is conducted on the cassava leaf images with regular monitoring of validation accuracy.

4. **Loss Function**:
   - Bi-Tempered Logistic Loss is utilized to combat the issue of imbalanced classes and label noise.

5. **Evaluation**:
   - The model’s performance is evaluated using standard metrics such as accuracy and loss curves.
   - Visualizations of training and validation metrics are provided to monitor overfitting or underfitting.

## Libraries Used

- [Pandas](https://pandas.pydata.org/): For data manipulation and reading CSV files.
- [Matplotlib](https://matplotlib.org/): Visualization library used to plot class distribution and training curves.
- [Seaborn](https://seaborn.pydata.org/): Used for enhanced data visualizations.
- [OpenCV](https://opencv.org/): For image reading and processing.
- [EfficientNet](https://github.com/qubvel/efficientnet): Deep learning architecture used for image classification.
- [BiTemperedLogisticLoss](https://github.com/google/bi-tempered-loss): Used to improve robustness against noisy labels.

## Results

The model successfully classified cassava leaf diseases with high accuracy. Key metrics such as accuracy and loss were monitored, and the final model is capable of predicting diseases with reasonable success.

### Future Work

- Further tuning of the model hyperparameters to improve classification accuracy.
- Experimenting with different deep learning architectures like ResNet or DenseNet.
- Augmenting the dataset with additional images to further generalize the model.
  
## References
- Kaggle competition dataset: [Cassava Leaf Disease Classification](https://www.kaggle.com/c/cassava-leaf-disease-classification)
- EfficientNet paper: [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
- Bi-Tempered Logistic Loss: [A robust loss function for imbalanced classification](https://arxiv.org/abs/1906.03361)

## License

This project is licensed under the MIT License


