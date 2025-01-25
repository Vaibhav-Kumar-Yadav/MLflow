import mlflow.pyfunc
import pandas as pd

mlflow.set_tracking_uri("sqlite:///mlruns.db")

model_name = "AGNewsClassificationModel"
model_version = 2

print(f"Loading model version {model_version} of '{model_name}' for deployment...")
model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")

sample_input = pd.DataFrame({
    "Title": ["Breaking: New technology breakthrough!"],
    "Description": ["In the latest tech news, scientists achieved a revolutionary breakthrough in AI technology."]
})

input_text = sample_input["Title"] + " " + sample_input["Description"]

predictions = model.predict(input_text)

class_labels = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Sci/Tech"
}

print(f"Predicted category: {class_labels[predictions[0]]}")