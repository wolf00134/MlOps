from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from mlflow.models import infer_signature
import matplotlib.pyplot as plt
import numpy as np
import mlflow
import mlflow.sklearn
import pandas as pd
import os

X, y = make_classification(
    # same as the previous section
    n_samples=1000, n_features=5, n_informative=3, n_classes=2,
    # flip_y - high value to add more noise
    flip_y=0.1,
    # class_sep - low value to reduce space between classes
    class_sep=0.5
)

mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.set_experiment("decision_tree_exp")

# Create DataFrame with features as columns
dataset = pd.DataFrame(X)
# give custom names to the features
dataset.columns = ['X1', 'X2', 'X3', 'X4', 'X5']
# Now add the label as a column
dataset['y'] = y

dataset.info()

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
criterion = ['gini', 'entropy', 'log_loss']
splitter = ['best', 'random']
max_features = ['sqrt', 'log2']

# initialize classifier
# for c in max_features:
classifier = DecisionTreeClassifier(criterion='log_loss', splitter="best", max_features='sqrt')
classifier.fit(x_train, y_train)
predict = classifier.predict(x_test)
accuracy = accuracy_score(y_test, predict)
print(accuracy)

# Run cross validation with 10 folds
scores = cross_validate(
    classifier, X, y, cv=10,
    # measure score for a list of classification metrics
    scoring=['accuracy', 'precision', 'recall', 'f1']
)

scores = pd.DataFrame(scores)
scores.mean().round(4)

with mlflow.start_run():
    # Log the hyperparameters
    mlflow.log_param('n samples', X.shape[0])
    mlflow.log_param('n features', X.shape[1])
    mlflow.log_param('n classes', len(np.unique(y)))

    # mlflow.log_param('n estimators', X.shape[1])
    
    # Log the loss metric
    mlflow.log_metric("accuracy", accuracy)

    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", "Basic LR model for iris data")

    # Infer the model signature
    # signature = infer_signature(X_train, lr.predict(X_train))

    # feature_importances = classifier.feature_importances_
    # for i, importance in enumerate(feature_importances):
    #     mlflow.log_metric(f"feature_{i}_importance", importance)

    # Save and Log Confusion Matrix
    cm = confusion_matrix(y_test, predict)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm)
    cm_display.plot(cmap=plt.cm.Blues)
    os.makedirs("artifacts", exist_ok=True)
    cm_path = "artifacts/confusion_matrix.png"
    plt.savefig(cm_path)
    mlflow.log_artifact(cm_path)

    # Infer the model signature
    signature = infer_signature(x_train, classifier.predict(x_train))

    # Log the Model
    # mlflow.sklearn.log_model(classifier, "model")

    print("Run logged successfully.")

    # Log the model
    # model_info = mlflow.sklearn.log_model(classifier, "model")
    model_info = mlflow.sklearn.log_model(
        sk_model=classifier,
        artifact_path="iris_model",
        signature=signature,
        input_example=x_train,
        registered_model_name="decision_tree_model",
    )

# Load the model back for predictions as a generic Python Function model
loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

predictions = loaded_model.predict(x_test)

# iris_feature_names = datasets.load_iris().feature_names

result = pd.DataFrame(x_test)
result["actual_class"] = y_test
result["predicted_class"] = predictions

result[:4]
