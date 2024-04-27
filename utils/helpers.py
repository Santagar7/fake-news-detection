def format_prediction(predicted_class, confidence):
    labels = {0: "Negative", 1: "Positive"}
    return f"The prediction is '{labels[predicted_class]}' with a confidence of {confidence:.2f}."
