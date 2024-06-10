import mlflow
import pickle

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

@data_exporter
def export_data(data, *args, **kwargs):
    print(data)
    model, vectorizer = data
    
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("mage-homework")

    with mlflow.start_run():
        mlflow.sklearn.log_model(model, "linear-regression")

        with open('vectorizer.pk', 'wb') as f:
            pickle.dump(vectorizer, f)
        
        mlflow.log_artifacts('./vectorizer.pk')
