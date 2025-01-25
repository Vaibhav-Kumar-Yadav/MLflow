import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading AG News data...")
data_path = './src/news-topic-classification/data/train.csv'
df = pd.read_csv(data_path)
print(df.head())

X = df['Title'].astype(str) + " " + df['Description'].astype(str)
y = df['Class Index'].values

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = make_pipeline(
    CountVectorizer(),
    LogisticRegression(max_iter=1000)
)

mlflow.set_tracking_uri("sqlite:///mlruns.db")
mlflow.set_experiment("AG_News_Topic_Classification")

print("Tracking experiment with MLflow...")
with mlflow.start_run():
    print("Training the model...")
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')

    print(f"Model accuracy: {accuracy}")

    mlflow.log_param("vectorizer", "CountVectorizer")
    mlflow.log_param("classifier", "LogisticRegression")
    mlflow.log_param("max_iter", 1000)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["World", "Sports", "Business", "Sci/Tech"], yticklabels=["World", "Sports", "Business", "Sci/Tech"])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    plt.close()

    input_example = pd.DataFrame(X_train[:5], columns=["text"])
    mlflow.sklearn.log_model(
        pipeline, 
        "model", 
        input_example=input_example
    )

    model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
    mlflow.register_model(model_uri, "AGNewsClassificationModel")

    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    model_version = client.get_latest_versions("AGNewsClassificationModel", stages=["None"])[0].version
    client.transition_model_version_stage(
        name="AGNewsClassificationModel",
        version=model_version,
        stage="Staging"
    )

    print(f"Model version {model_version} registered and transitioned to Staging.")