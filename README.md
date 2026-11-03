# AI-Based Unified Content Verification System

An AI-powered content verification application that detects **fake news, AI-generated text, and deepfake images** using machine learning and deep learning techniques.

The project integrates three specialized detection models into a single web application built with Streamlit. Users can select the type of content they want to verify, submit text or an image, and receive a prediction with a confidence score.

---

## 1. Project Overview

With the rapid growth of artificial intelligence and digital media, misleading information and artificially generated content have become increasingly difficult to identify.

Fake news articles can spread misinformation, AI-generated text can resemble human writing, and synthetic face images can appear realistic.

Checking these different forms of content often requires separate tools and techniques.

The **AI Based Unified Content Verification System** addresses this problem by combining three independent AI detection modules into one application.

The system includes:

* **Fake News Detection:** Identifies whether a news article is fake or real.
* **AI-Generated Text Detection:** Determines whether a text sample is human-written or AI-generated.
* **Deepfake Image Detection:** Classifies face images as real or fake.

Each module uses a dedicated trained model and returns a prediction along with its confidence score.

---

## 2. Project Objectives

The main objectives of this project are:

1. To develop a machine learning model for fake news detection.
2. To identify AI-generated text using Natural Language Processing techniques.
3. To detect real and AI-generated face images using deep learning.
4. To integrate all three detection modules into a unified system.
5. To develop a simple and interactive web interface using Streamlit.
6. To evaluate the models using appropriate performance metrics.

---

## 3. System Architecture

The application follows a modular architecture in which each detection task has its own trained model.

**Overall workflow:**

```text
                  USER
                    |
                    v
          CONTENT TYPE SELECTION
                    |
                    v
             UNIFIED BACKEND
                    |
          +---------+---------+
          |         |         |
          v         v         v
       FAKE NEWS  AI TEXT  DEEPFAKE
          |         |         |
          v         v         v
        TF-IDF    TF-IDF   EfficientNetB0
          +---------+---------+
                    |
                    v
             RESULT + CONFIDENCE
```

### Workflow Explanation

1. The user opens the Streamlit application.
2. The user selects one of the three verification modules.
3. The user provides the required text or image.
4. The unified backend routes the input to the appropriate model.
5. The selected model processes the input and generates a prediction.
6. The application displays the result and confidence score.

---

## 4. Technologies Used

| Technology          | Purpose                            |
| ------------------- | ---------------------------------- |
| Python              | Main programming language          |
| Pandas              | Dataset handling and preprocessing |
| NumPy               | Numerical operations               |
| Scikit-learn        | Machine learning and evaluation    |
| TensorFlow / Keras  | Deep learning                      |
| TF-IDF              | Text feature extraction            |
| Logistic Regression | Text classification                |
| EfficientNetB0      | Image classification               |
| Matplotlib          | Data visualization                 |
| Streamlit           | Web application                    |
| Google Colab        | Model development and training     |
| Google Drive        | Model and project artifact storage |

---

# 5. Project Modules

## Module 1: Fake News Detection

### Description

The Fake News Detection module identifies whether a given news article is fake or real.

It uses the WELFake dataset, which contains real and fake news articles.

### Dataset

**WELFake Dataset**

* Original dataset: WELFake_Dataset.csv
* Final cleaned records: 62,717
* Input features: Title and article text
* Target variable: News label

### Data Preprocessing

The following preprocessing operations were performed:

1. Loaded the dataset.
2. Removed records with missing article text.
3. Filled missing titles with empty strings.
4. Removed articles with conflicting labels.
5. Removed duplicate article texts.
6. Combined the title and article text into a single feature.
7. Split the dataset into training and testing sets.

### Model Development

**Algorithms used:**

* TF-IDF Vectorization
* Logistic Regression

TF-IDF converts the news text into numerical features based on word importance.

Logistic Regression uses these features to classify the article into one of two categories.

### Classification Labels

| Label | Meaning   |
| ----- | --------- |
| 0     | Fake News |
| 1     | Real News |

### Model Performance

**Test Accuracy: 96.17%**

The model was evaluated using accuracy, precision, recall, F1-score, and a confusion matrix.

---

## Module 2: AI-Generated Text Detection

### Description

The AI-Generated Text Detection module determines whether a given text sample is written by a human or generated by artificial intelligence.

The module uses Natural Language Processing and machine learning techniques to identify patterns in text.

### Dataset

**HC3 – Human ChatGPT Comparison Corpus**

The dataset contains human-written answers and ChatGPT-generated answers.

* Original records: 24,322
* Samples created from human and ChatGPT answers: 85,431
* Final cleaned samples: 79,331
* Input feature: Text
* Target variable: Text origin

### Data Preprocessing

1. Loaded the HC3 JSONL dataset.
2. Extracted human-written answers.
3. Extracted ChatGPT-generated answers.
4. Assigned labels to both categories.
5. Combined the extracted responses.
6. Removed duplicate responses.
7. Split the cleaned dataset into training and testing sets.

### Model Development

**Algorithms and techniques used:**

* TF-IDF Vectorization
* Logistic Regression

The TF-IDF vectorizer converts text into numerical representations using unigram and bigram features.

Logistic Regression classifies the text based on the learned patterns.

### Classification Labels

| Label | Meaning       |
| ----- | ------------- |
| 0     | Human-Written |
| 1     | AI-Generated  |

### Model Performance

**Test Accuracy: 98.05%**

The model was evaluated using accuracy, precision, recall, F1-score, and a confusion matrix.

---

## Module 3: Deepfake Image Detection

### Description

The Deepfake Image Detection module identifies whether a face image is real or artificially generated.

It uses deep learning and transfer learning to identify visual patterns in face images.

### Dataset

**140K Real and Fake Faces Dataset**

The dataset contains 140,000 face images divided into training, validation, and testing folders.

| Dataset Split | Number of Images |
| ------------- | ---------------: |
| Training      |          100,000 |
| Validation    |           20,000 |
| Testing       |           20,000 |
| **Total**     |      **140,000** |

Each split contains real and fake image categories.

### Image Preprocessing

1. Extracted the dataset from the ZIP archive.
2. Organized images into training, validation, and testing folders.
3. Resized images to 224 × 224 pixels.
4. Loaded images in RGB format.
5. Assigned binary labels based on folder names.
6. Created TensorFlow image batches with a batch size of 32.

### Model Development

**Architecture: EfficientNetB0**

The model uses transfer learning with a pretrained EfficientNetB0 network.

The architecture consists of:

1. Input Layer – 224 × 224 × 3.
2. Data Augmentation – Horizontal flipping, rotation, and zoom.
3. EfficientNetB0 – Pretrained feature extraction backbone.
4. Global Average Pooling 2D – Reduces feature maps.
5. Dropout – 0.30, to help reduce overfitting.
6. Dense Layer – One output neuron.
7. Sigmoid Activation – Produces a binary classification probability.

The deeper portion of the pretrained network was fine-tuned for the real/fake face classification task.

### Classification Labels

| Label | Meaning    |
| ----- | ---------- |
| 0     | Fake Image |
| 1     | Real Image |

### Model Performance

**Test Accuracy: 95.47%**

The final model was evaluated on 20,000 test images that were not used for model weight updates.

---

# 6. Model Performance Comparison

The final test accuracy of the three detection modules is shown below.

| Detection Module            | Algorithm                    | Test Accuracy |
| --------------------------- | ---------------------------- | ------------: |
| Fake News Detection         | TF-IDF + Logistic Regression |        96.17% |
| AI-Generated Text Detection | TF-IDF + Logistic Regression |        98.05% |
| Deepfake Image Detection    | EfficientNetB0               |        95.47% |

### Evaluation Metrics

The models were evaluated using the following metrics:

* **Accuracy:** Measures the proportion of correct predictions.
* **Precision:** Measures how many predicted positive samples were correct.
* **Recall:** Measures how many actual positive samples were correctly identified.
* **F1-Score:** Provides a balance between precision and recall.
* **Confusion Matrix:** Displays correct and incorrect predictions for each class.

Each accuracy value represents performance on the corresponding module's own test dataset.

---

# 7. Unified Web Application

The three trained models are integrated into a single application using Streamlit.

### Application Features

* Simple and interactive user interface.
* Three content verification options.
* Text input for fake news detection.
* Text input for AI-generated text detection.
* Image upload for deepfake detection.
* Automatic routing to the selected model.
* Human-readable classification results.
* Confidence score and visual progress indicator.
* Model performance information.

### Application Workflow

```text
             STREAMLIT APPLICATION
                       |
             SELECT DETECTION TYPE
                       |
          +------------+------------+
          |            |            |
          v            v            v
       NEWS TEXT    AI TEXT       IMAGE
          |            |            |
          v            v            v
       NEWS MODEL   AI TEXT MODEL  IMAGE MODEL
          |            |            |
          +------------+------------+
                       |
                       v
               PREDICTION RESULT
                       |
                       v
                CONFIDENCE SCORE
```

---

# 8. Project Structure

The project is organized into separate folders for models, datasets, results, and application files.

```text
AI_Unified_Content_Verification/
│
├── models/
│   ├── fake_news_logistic_regression.pkl
│   ├── fake_news_tfidf.pkl
│   ├── ai_text_logistic_regression.pkl
│   ├── ai_text_tfidf.pkl
│   └── deepfake_efficientnet_retrained.keras
│
├── datasets/
│   ├── fake_news/
│   │   └── WELFake_Dataset.csv
│   │
│   ├── ai_text/
│   │
│   └── deepfake/
│       └── 140k-real-and-fake-faces.zip
│
├── results/
│   ├── fake_news_evaluation.txt
│   ├── fake_news_test_predictions.csv
│   ├── ai_text_evaluation.txt
│   ├── ai_text_test_predictions.csv
│   ├── deepfake_evaluation.txt
│   └── deepfake_test_predictions.csv
│
├── unified_verification.py
├── app.py
├── app_v2.py
├── app_v3.py
└── app_v4.py
```

**Note:** The structure above represents the development project. Large datasets and trained model files may need to be downloaded or provided separately rather than committed directly to GitHub.

---

# 9. Installation and Execution

### Step 1: Clone the Repository

```bash
git clone <your-github-repository-link>
```

### Step 2: Open the Project Folder

```bash
cd AI_Unified_Content_Verification
```

### Step 3: Install Required Libraries

```bash
pip install streamlit tensorflow scikit-learn pandas numpy pillow joblib matplotlib
```

### Step 4: Verify Model Files

Ensure that the five trained model and vectorizer files are available in the `models/` directory.

Also ensure that the model paths configured in `unified_verification.py` point to the correct locations.

### Step 5: Run the Application

```bash
streamlit run app_v4.py
```

### Step 6: Open the Application

Streamlit will provide a local URL in the terminal. Open that URL in your browser to use the application.

---

# 10. Testing and Validation

The project was tested at different stages to verify the individual models and the unified application.

### Model Testing

* Evaluated all three models using their respective held-out test datasets.
* Generated classification reports.
* Generated confusion matrices.
* Saved test predictions and evaluation reports.

### Integration Testing

* Verified that all five model and vectorizer artifacts load successfully.
* Tested the unified backend.
* Checked that each content type is routed to the correct detection module.
* Verified the returned prediction labels and confidence values.

### Application Testing

* Launched the Streamlit V4 application.
* Verified successful application response (HTTP 200).
* Tested text inputs for both text-based modules.
* Tested image uploads for deepfake detection.
* Verified result display and confidence progress bar.

---

# 11. Limitations

Although the system achieved good results on its selected test datasets, some limitations remain:

1. Model performance depends on the quality and diversity of the training datasets.
2. AI-generated content from unseen tools or generators may be misclassified.
3. The fake news module classifies learned text patterns; it does not verify claims against live news sources.
4. The deepfake model was evaluated on face images from the selected dataset and may not generalize to every image-generation method.
5. Confidence scores represent model predictions and should not be interpreted as guaranteed certainty.

The system should be used as an initial verification aid, not as a replacement for human judgment or independent fact-checking.

---

# 12. Future Enhancements

Possible improvements include:

* Integration of live fact-checking and trusted news sources.
* Support for additional AI text generators.
* Training with more diverse synthetic-image datasets.
* Video deepfake detection.
* Multilingual content verification.
* Explainable AI techniques.
* Deployment through a cloud-based platform or API.

---

# 13. Conclusion

The **AI Based Unified Content Verification System** successfully integrates three independent AI detection modules into one web application.

The project combines Natural Language Processing, machine learning, deep learning, and web application development to address three different digital content verification tasks.

The Fake News Detection module achieved 96.17% test accuracy, the AI-Generated Text Detection module achieved 98.05%, and the Deepfake Image Detection module achieved 95.47%.

The unified Streamlit application provides a common interface through which users can submit different types of content and receive predictions with confidence scores.

This modular approach demonstrates how multiple AI models can be combined into a single practical content verification system.

---




*Note: The reported accuracy values belong to the individual models' respective test datasets. Predictions on new, unseen content may vary.*
