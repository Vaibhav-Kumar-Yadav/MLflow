import mlflow.pyfunc
import pandas as pd

mlflow.set_tracking_uri("sqlite:///mlruns.db")

model_name = "IMDBSentimentAnalysisModel"
model_version = 2

print(f"Loading model version {model_version} of '{model_name}' for deployment...")
model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")

sample_input = pd.DataFrame(["This movie was fantastic and inspiring!"], columns=["review"])

predictions = model.predict(sample_input)

sentiment_labels = {0: 'negative', 1: 'positive'}

predicted_sentiment = sentiment_labels[predictions[0]]
print(f"Predicted sentiment: {predicted_sentiment}")