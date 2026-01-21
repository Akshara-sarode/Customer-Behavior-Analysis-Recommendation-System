# 🛒 CUSTOMER BEHAVIOR ANALYSIS & RECOMMENDATION SYSTEM

Using SVD, Matrix Factorization, Stacked LSTM & Attention-Based Autoencoder

## 📌 PROJECT OVERVIEW

Personalized recommendation systems are a core component of modern e-commerce platforms.
This project presents a comprehensive customer behavior analysis and recommendation system built using traditional collaborative filtering and advanced deep learning models.

The system analyzes user–item interactions, temporal behavior, and latent representations to predict ratings and recommend relevant products from the Amazon Fine Food Reviews dataset.

## 🎯 PROJECT OBJECTIVES

Analyze customer behavior using historical interaction data

Compare classical recommendation algorithms with deep learning models

Capture temporal dependencies in user preferences

Improve recommendation accuracy and personalization

Address challenges such as data sparsity and cold-start

## 🧠 MODELS & TECHNIQUES USED

### 🔹 Traditional & Machine Learning Models

Cosine Similarity (Distance-Based Collaborative Filtering)

Random Forest Regressor

Singular Value Decomposition (SVD)

Matrix Factorization (MF)

Probabilistic Matrix Factorization (PMF)

### 🔹 Deep Learning Models

Convolutional Neural Networks (CNN)

Stacked Long Short-Term Memory (LSTM)

Attention-Based Autoencoder

Captures temporal dependencies

Learns contextual importance

Handles sparse and cold-start scenarios

## 📊 DATASET DETAILS

Amazon Fine Food Reviews Dataset

Over 568,000 reviews (filtered to ~64,000 for efficiency)

Key attributes:

UserId

ProductId

Rating

Time

Interaction frequency features

## Dataset Source:
https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews

## ⚙️ METHODOLOGY

### 1️⃣ Data Preprocessing

Removed sparse users and products

Encoded user–item interactions

Time-aware feature engineering

### 2️⃣ Model Training & Comparison

Trained multiple recommendation models

Compared regression and classification performance

### 3️⃣ Deep Learning Pipeline

Sequential modeling with stacked LSTM

Attention mechanism for behavioral relevance

Autoencoder-based latent representation learning

### 4️⃣ Recommendation Strategy

Predict user ratings

Select Top-N recommended products

Filter outdated items using time-based thresholds

## 📈 RESULTS & PERFORMANCE

Model	Accuracy
Cosine Similarity	~44%
Random Forest Regressor	~76%
SVD	~94%
Matrix Factorization	~86%
Probabilistic MF	~83%
CNN	~72%
Stacked LSTM + Attention Autoencoder	~99%

SVD and Matrix Factorization performed best among classical models, while the Stacked LSTM with Attention Autoencoder achieved the highest accuracy by effectively modeling temporal user behavior.

## 🧪 EVALUATION METRICS

Mean Squared Error (MSE)

Classification Accuracy

Confusion Matrix

Ratings > 4 treated as positive recommendations

## 🛠️ TECH STACK

# Programming & Tools

Python

NumPy, Pandas, Scikit-learn

TensorFlow / Keras

Matplotlib, Seaborn

Jupyter Notebook

## 📁 PROJECT STRUCTURE

├── Amazon_pmf_svd_Stacked_LSTM_Autoencoder.ipynb
├── data/
│   └── amazon_fine_food_reviews.csv
├── figures/
│   └── evaluation_plots.png
├── report/
│   └── Customer_Behavior_Analysis_Paper.pdf
├── README.md


## 🚀 HOW TO RUN THE PROJECT

git clone https://github.com/your-username/customer-behavior-recommendation.git
cd customer-behavior-recommendation
pip install -r requirements.txt
jupyter notebook Amazon_pmf_svd_Stacked_LSTM_Autoencoder.ipynb


## 🔮 FUTURE ENHANCEMENTS

Integrate sentiment analysis from review text

Add user and product metadata

Improve explainability of deep learning models

Deploy as a real-time recommendation system

## 👩‍💻 AUTHOR

Akshara Avinash Sarode

MS in Computer Science
Data Analytics | Machine Learning | Recommender Systems

## LinkedIn:
https://www.linkedin.com/in/akshara-avinash-sarode/

## ⭐ ACKNOWLEDGMENT

If you find this project useful, consider starring ⭐ the repository!
