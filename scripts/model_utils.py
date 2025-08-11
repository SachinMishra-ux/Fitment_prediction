import numpy as np
import joblib

from scipy.sparse import issparse

def load_model(model_path):
    """Load the trained model and return preprocessor & regressor."""
    model = joblib.load(model_path)
    return model.named_steps['preprocessor'], model.named_steps['regressor']

def get_feature_names(column_transformer):
    """Extract feature names after transformation."""
    output_features = []
    for name, trans, cols in column_transformer.transformers_:
        if hasattr(trans, 'get_feature_names_out'):
            names = trans.get_feature_names_out(cols)
        elif trans == 'passthrough':
            names = cols
        else:
            names = [f"{name}_{col}" for col in cols]
        output_features.extend(names)
    return output_features

def transform_input(preprocessor, input_df):
    """Apply preprocessing transformation."""
    X_transformed = preprocessor.transform(input_df)
    if issparse(X_transformed):
        X_transformed = X_transformed.toarray()
    return X_transformed

def predict_with_ci(model, X_transformed):
    """Predict and calculate 95% confidence interval."""
    preds = np.array([tree.predict(X_transformed)[0] for tree in model.estimators_])
    return {
        "mean": np.mean(preds),
        "lower_ci": np.percentile(preds, 2.5),
        "upper_ci": np.percentile(preds, 97.5),
        "all_preds": preds
    }

