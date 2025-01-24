
# MLflow Projects and Tutorial

Welcome to the MLflow repository, where we explore the powerful capabilities of MLflow for both Natural Language Processing (NLP) and Machine Learning (ML) projects. This repository includes two distinct NLP projects and a comprehensive ML tutorial notebook. 

## Repository Structure

- **NLP Projects**
  - **Sentiment Analysis**: Demonstrates sentiment prediction using IMDb reviews.
  - **News Topic Classification**: Classifies news articles into different categories using the AG News dataset.

- **Machine Learning Tutorial Notebook**
  - A detailed guide explaining MLflow's features and how they apply to a range of ML tasks, complete with hands-on coding exercises.

## Getting Started with MLflow

To explore the experiments, models, and metrics tracked using MLflow in these projects, follow the steps below to set up and access the MLflow server interface.

### Prerequisites

- Python 3.x installed on your machine
- Virtual environment tools (e.g., `venv` or `conda`)
- Dependencies listed in `requirements.txt` (if any)

### Setup and Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Vaibhav-Kumar-Yadav/MLflow.git
   cd MLflow
   ```

2. **Create and Activate a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate    # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Projects

#### Sentiment Analysis

1. **Navigate to the Project Directory**:
   ```bash
   cd sentiment-analysis
   ```

2. **Run Experiment Tracking**:
   ```bash
   python scripts/track_experiment.py
   ```

3. **Deploy the Model**:
   ```bash
   python scripts/deploy_model.py
   ```

#### News Topic Classification

1. **Navigate to the Project Directory**:
   ```bash
   cd news-topic-classification
   ```

2. **Run Experiment Tracking**:
   ```bash
   python scripts/track_experiment.py
   ```

3. **Deploy the Model**:
   ```bash
   python scripts/deploy_model.py
   ```

### Running the ML Tutorial Notebook

1. **Navigate to the Notebook Directory**:
   ```bash
   cd src
   ```

2. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

### Setting Up the MLflow Server

1. **Start the MLflow Server**:
   ```bash
   mlflow ui --backend-store-uri sqlite:///mlruns.db
   ```

2. **Access the MLflow UI**: Open [http://localhost:5000](http://localhost:5000) in your web browser to explore the experiments, models, and runs.

## Contributions

Feel free to fork this repository and submit pull requests. Contributions are always welcome!


## Contact

- **GitHub**: [@Vaibhav-Kumar-Yadav](https://github.com/Vaibhav-Kumar-Yadav)

---

Thank you for exploring MLflow with us! If you encounter any issues or have questions, feel free to reach out.
```

