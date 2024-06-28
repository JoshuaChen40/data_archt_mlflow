# https://mlflow.org/docs/latest/llms/transformers/tutorials/text-generation/text-generation.html

# Disable tokenizers warnings when constructing pipelines
# %env TOKENIZERS_PARALLELISM=false

import warnings
import mlflow
import transformers

# Disable a few less-than-useful UserWarnings from setuptools and pydantic
warnings.filterwarnings("ignore", category=UserWarning)

# Define the task that we want to use (required for proper pipeline construction)
task = "text2text-generation"

# Define the pipeline, using the task and a model instance that is applicable for our task.
generation_pipeline = transformers.pipeline(
    task=task,
    model="declare-lab/flan-alpaca-large",
)

# Define a simple input example that will be recorded with the model in MLflow, giving
# users of the model an indication of the expected input format.
input_example = ["prompt 1", "prompt 2", "prompt 3"]

# Define the parameters (and their defaults) for optional overrides at inference time.
parameters = {"max_length": 512, "do_sample": True, "temperature": 0.4}

# Generate the signature for the model that will be used for inference validation and type checking (as well as validation of parameters being submitted during inference)
signature = mlflow.models.infer_signature(
    input_example,
    mlflow.transformers.generate_signature_output(generation_pipeline, input_example),
    parameters,
)

# Visualize the signature
# signature

# 設定 MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Transformers Introduction")

with mlflow.start_run():
    model_info = mlflow.transformers.log_model(
        transformers_model=generation_pipeline,
        artifact_path="text_generator",
        input_example=input_example,
        signature=signature,
        # Transformer model does not use Pandas Dataframe as input, internal input type conversion should be skipped.
        example_no_conversion=True,
        # Uncomment the following line to save the model in 'reference-only' mode:
        # save_pretrained=False,
    )

# Load our pipeline as a generic python function
sentence_generator = mlflow.pyfunc.load_model(model_info.model_uri)

def format_predictions(predictions):
    """
    Function for formatting the output for readability in a Jupyter Notebook
    """
    formatted_predictions = []

    for prediction in predictions:
        # Split the output into sentences, ensuring we don't split on abbreviations or initials
        sentences = [
            sentence.strip() + ("." if not sentence.endswith(".") else "")
            for sentence in prediction.split(". ")
            if sentence
        ]

        # Join the sentences with a newline character
        formatted_text = "\n".join(sentences)

        # Add the formatted text to the list
        formatted_predictions.append(formatted_text)

    return formatted_predictions

# Validate that our loaded pipeline, as a generic pyfunc, can produce an output that makes sense
predictions = sentence_generator.predict(
    data=[
        "I can't decide whether to go hiking or kayaking this weekend. Can you help me decide?",
        "Please tell me a joke about hiking.",
    ],
    params={"temperature": 0.7},
)

# Format each prediction for notebook readability
formatted_predictions = format_predictions(predictions)

for i, formatted_text in enumerate(formatted_predictions):
    print(f"Response to prompt {i+1}:\n{formatted_text}\n")