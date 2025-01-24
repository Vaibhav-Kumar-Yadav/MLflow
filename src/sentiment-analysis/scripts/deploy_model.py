import mlflow.pyfunc

mlflow.set_tracking_uri("sqlite:///mlruns.db")

model_name = "IMDBSentimentAnalysisModel"
model_version = 1

print(f"Loading model version {model_version} of '{model_name}' for deployment...")
model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")

sample_input = ["This movie was fantastic and inspiring!"]

predictions = model.predict(sample_input)
print(f"Predicted sentiment: {['negative', 'positive'][predictions[0]]}")