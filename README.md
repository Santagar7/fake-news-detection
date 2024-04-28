
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
Navigate to the `train/` directory within the project:
**Model Training**
   - Train the model using the training script. This will use the preprocessed data and save the trained model.
   ```
   fake_news_detection_bert.ipynb
   ```

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
