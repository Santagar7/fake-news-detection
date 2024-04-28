
# Fake News Detection

## Introduction
This repository contains the implementation of a fake news detection system using machine learning and natural language processing techniques. The project utilizes models like LSTM, BERT, and other algorithms to classify news articles as real or fake.

## Project Setup
1. **Clone the Repository**
   ```
   git clone https://github.com/Santagar7/fake-news-detection.git
   ```
2. **Install Dependencies**
   Navigate to the project directory and install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

## Data Preparation
1. **Download the Dataset**
   - Download the dataset from [Kaggle: Fake News Detection](https://www.kaggle.com/datasets/bhavikjikadara/fake-news-detection).
   - Ensure you have a Kaggle account and accept the competition rules to access the dataset.
2. **Prepare the Data**
   - Place the downloaded data in the `data/` directory within the project root.
   - Extract the contents, ensuring the CSV files are accessible.

## Model Training

To train the model on the fake news detection dataset, follow these steps:

1. **Prepare the Environment**:
   - Ensure you have Python 3.10+ installed on your system.
   - Install the required dependencies using `pip install -r requirements.txt`.

2. **Download the Dataset**:
   - Download the dataset from Kaggle by instructions above.
   
3. **Train the Model**:
   - Run the training process by executing the `training/main.py` script.
   - The training process involves several steps managed by the script, including data loading, model initialization, training epochs, and validation.
   - Training parameters like epochs, batch size, and model configuration are set in `training/config.py`.

4. **Monitoring Training Progress**:
   - Training and validation metrics such as loss and accuracy are printed after each epoch.
   - The model with the best validation accuracy is saved as `best_model_state.bin`.

5. **Evaluate the Model**:
   - After training, the model's performance is evaluated on the test set to calculate the final accuracy and generate the confusion matrix.

By following these steps, you will be able to train the model and evaluate its performance on the task of detecting fake news in textual content. The code is structured in a modular way to allow easy customization and scalability for future enhancements.


## Usage
To run the fake news detection model on new data:
```
python main.py
```

Then go to Telegram bot [Bert Detector](https://t.me/f_news_detection_bot)

## Technologies Used
- Python 3.10+
- Libraries: NumPy, Pandas, TensorFlow, Transformers

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.
