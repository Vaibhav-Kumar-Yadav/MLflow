import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

print("Loading AG News data...")
data_path = './src/news-topic-classification/data/train.csv'
df = pd.read_csv(data_path)
print(df.head())

# Combine Title and Description into a single text feature
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

    print(f"Model accuracy: {accuracy}")

    mlflow.log_param("vectorizer", "CountVectorizer")
    mlflow.log_param("classifier", "LogisticRegression")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(pipeline, "model")

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