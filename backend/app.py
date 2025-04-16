from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import io
import time
import numpy as np

app = Flask(__name__)
CORS(app)

trained_models = {}       
model_mode = "regression" 
label_encoder = None      
prediction_schema = []    

def build_prediction_schema(df, features):
    """
    Build a prediction schema from the originally selected independent features.
    For each feature, if numeric, record type "numeric".
    Otherwise, record type "categorical" along with its unique options and expected dummy column names.
    """
    schema = []
    for col in features:
        try:
            pd.to_numeric(df[col])
            schema.append({"name": col, "type": "numeric"})
        except Exception:
            df[col] = df[col].astype(str)
            options = df[col].unique().tolist()
            dummy_columns = [f"{col}_{opt}" for opt in options]
            schema.append({
                "name": col,
                "type": "categorical",
                "options": options,
                "dummy_columns": dummy_columns
            })
    return schema

def process_features(df, features):
    """
    Process only the selected independent features.
    For numeric features, convert to numeric.
    For categorical features, perform one-hot encoding.
    Return the modified dataframe and the list of processed feature names.
    """
    processed_features = []
    for col in features:
        try:
            df[col] = pd.to_numeric(df[col], errors='raise')
            processed_features.append(col)
        except Exception:
            df[col] = df[col].astype(str)
            dummies = pd.get_dummies(df[col], prefix=col)
            processed_features.extend(dummies.columns.tolist())
            df = pd.concat([df, dummies], axis=1)
            df.drop(columns=[col], inplace=True)
    return df, processed_features

def train_and_evaluate(X_train, X_test, y_train, y_test, feature_names, is_regression=True):
    global trained_models
    results = {}
    
    if is_regression:
        # Regression models
        lr = LinearRegression()
        start_time = time.time()
        lr.fit(X_train, y_train)
        lr_time = time.time() - start_time
        pred_lr = lr.predict(X_test)
        mse_lr = mean_squared_error(y_test, pred_lr)
        r2_lr = r2_score(y_test, pred_lr)
        results["Linear Regression"] = {
            "MSE": round(mse_lr, 4),
            "R2 Score": round(r2_lr, 4),
            "Training Time": round(lr_time, 4)
        }
        
        rf = RandomForestRegressor(n_estimators=100)
        start_time = time.time()
        rf.fit(X_train, y_train)
        rf_time = time.time() - start_time
        pred_rf = rf.predict(X_test)
        mse_rf = mean_squared_error(y_test, pred_rf)
        r2_rf = r2_score(y_test, pred_rf)
        rf_result = {
            "MSE": round(mse_rf, 4),
            "R2 Score": round(r2_rf, 4),
            "Training Time": round(rf_time, 4)
        }
        
        if feature_names is not None:
            importances = rf.feature_importances_
            rf_result["Feature Importance"] = {name: round(imp, 4) for name, imp in zip(feature_names, importances)}
        
        results["Random Forest"] = rf_result
        results["Predictions"] = {
            "Actual": y_test.tolist(),
            "Linear Regression": pred_lr.tolist(),
            "Random Forest": pred_rf.tolist()
        }
        
        trained_models["Model1"] = lr
        trained_models["Model2"] = rf
        
    else:
        lr = LogisticRegression(max_iter=1000)
        start_time = time.time()
        lr.fit(X_train, y_train)
        lr_time = time.time() - start_time
        pred_lr = lr.predict(X_test)
        
        # Calculate metrics
        acc_lr = accuracy_score(y_test, pred_lr)
        prec_lr = precision_score(y_test, pred_lr, average='weighted', zero_division=0)
        rec_lr = recall_score(y_test, pred_lr, average='weighted', zero_division=0)
        f1_lr = f1_score(y_test, pred_lr, average='weighted', zero_division=0)
        
        results["Logistic Regression"] = {
            "Accuracy": round(acc_lr, 4),
            "Precision": round(prec_lr, 4),
            "Recall": round(rec_lr, 4),
            "F1 Score": round(f1_lr, 4),
            "Training Time": round(lr_time, 4)
        }
        
        rf = RandomForestClassifier(n_estimators=100)
        start_time = time.time()
        rf.fit(X_train, y_train)
        rf_time = time.time() - start_time
        pred_rf = rf.predict(X_test)
        
        # Calculate metrics
        acc_rf = accuracy_score(y_test, pred_rf)
        prec_rf = precision_score(y_test, pred_rf, average='weighted', zero_division=0)
        rec_rf = recall_score(y_test, pred_rf, average='weighted', zero_division=0)
        f1_rf = f1_score(y_test, pred_rf, average='weighted', zero_division=0)
        
        rf_result = {
            "Accuracy": round(acc_rf, 4),
            "Precision": round(prec_rf, 4),
            "Recall": round(rec_rf, 4),
            "F1 Score": round(f1_rf, 4),
            "Training Time": round(rf_time, 4)
        }
        
       
        results["Random Forest"] = rf_result
        
        # Add predictions
        results["Predictions"] = {
            "Actual": y_test.tolist(),
            "Logistic Regression": pred_lr.tolist(),
            "Random Forest": pred_rf.tolist()
        }
        
        # Add confusion matrices
        cm_lr = confusion_matrix(y_test, pred_lr)
        cm_rf = confusion_matrix(y_test, pred_rf)
        
        # Get class labels
        if label_encoder:
            class_labels = label_encoder.classes_.tolist()
        else:
            class_labels = sorted(list(set(y_test.tolist())))
            
        # Convert confusion matrices to dictionaries
        cm_lr_dict = {}
        cm_rf_dict = {}
        
        for i, actual_class in enumerate(class_labels):
            cm_lr_dict[actual_class] = {}
            cm_rf_dict[actual_class] = {}
            for j, pred_class in enumerate(class_labels):
                cm_lr_dict[actual_class][pred_class] = int(cm_lr[i, j])
                cm_rf_dict[actual_class][pred_class] = int(cm_rf[i, j])
        
        results["Confusion Matrix"] = {
            "Logistic Regression": cm_lr_dict,
            "Random Forest": cm_rf_dict
        }
        
        trained_models["Model1"] = lr
        trained_models["Model2"] = rf
        
    return results

@app.route('/compare', methods=['GET'])
def compare_models():
    """
    Iris dataset comparison endpoint.
    Accepts query parameters:
      - target: name of the target column.
      - features: list of independent features (comma-separated).
    If target equals "species", a species column is created from iris.target.
    Defaults: if not provided, target = first measurement, features = remaining measurements.
    """
    global trained_models, prediction_schema, model_mode, label_encoder
    try:
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        
        # Get query parameters
        target = request.args.get('target')
        features_param = request.args.get('features')
        
        # Parse features from comma-separated string if provided
        if features_param:
            features = features_param.split(',')
        else:
            features = []
        
        # If target equals "species", create species column and use classification mode
        if target == "species":
            df["species"] = [iris.target_names[t] for t in iris.target]
            target = "species"
            if not features:
                features = iris.feature_names  # use all measurements as features
            is_regression = False
            label_encoder = LabelEncoder()
            df[target] = label_encoder.fit_transform(df[target])
        else:
            if not target or target not in df.columns:
                target = iris.feature_names[0]
            if not features:
                features = [col for col in df.columns if col != target]
            # Determine regression vs classification
            is_regression = True
            try:
                df[target] = pd.to_numeric(df[target], errors='raise')
            except Exception:
                is_regression = False
                label_encoder = LabelEncoder()
                df[target] = label_encoder.fit_transform(df[target])
        
        model_mode = "regression" if is_regression else "classification"
        
        # Build prediction schema from selected independent features
        prediction_schema = build_prediction_schema(df, features)
        
        # Handle NaN values by filling with mean
        for col in features:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mean())
        
        X = df[features].values
        y = df[target].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        results = train_and_evaluate(X_train, X_test, y_train, y_test, features, is_regression)
        
        preview = df.head(5).to_dict(orient='records')
        return jsonify({
            "success": True,
            "results": results,
            "data_preview": preview,
            "columns": df.columns.tolist(),
            "features_used": features,
            "prediction_schema": prediction_schema
        })
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/upload', methods=['POST'])
def handle_upload():
    """
    Custom dataset upload endpoint.
    Expects:
      - A CSV file.
      - A 'target' field for the target column.
      - One or more 'features' fields for independent features.
    Defaults to the last column as target if not provided.
    If no independent features are provided, returns an error.
    Builds the model only on the selected independent features.
    If the target is non-numeric, applies label encoding and uses classification mode.
    Categorical independent features are one-hot encoded.
    Returns a prediction_schema built solely from the selected independent features.
    """
    global trained_models, prediction_schema, model_mode, label_encoder
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file uploaded"}), 400
        file = request.files['file']
        if not file.filename.endswith('.csv'):
            return jsonify({"success": False, "error": "Only CSV files accepted"}), 400
        
        file_content = file.stream.read().decode('utf-8')
        df = pd.read_csv(io.StringIO(file_content))
        
        if df.shape[1] < 2:
            return jsonify({"success": False, "error": "CSV must have at least 2 columns"}), 400

        target = request.form.get("target")
        features = request.form.getlist("features")
        
        if target is None or target not in df.columns:
            target = df.columns[-1]
            features = list(df.columns[:-1])
        else:
            if not features:
                features = [col for col in df.columns if col != target]

        # Determine if regression or classification
        is_regression = True
        try:
            df[target] = pd.to_numeric(df[target], errors='raise')
        except Exception:
            is_regression = False
            label_encoder = LabelEncoder()
            df[target] = label_encoder.fit_transform(df[target])
        
        model_mode = "regression" if is_regression else "classification"

        # Build prediction schema from selected independent features
        prediction_schema = build_prediction_schema(df, features)
        
        # Handle NaN values by filling with mean for numeric columns
        for col in features:
            if pd.api.types.is_numeric_dtype(df[col]) and df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mean())
        
        # Process features (handle categorical variables)
        df, processed_features = process_features(df, features)
        
        X = df[processed_features].values
        y = df[target].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        results = train_and_evaluate(X_train, X_test, y_train, y_test, processed_features, is_regression)
        
        preview = df.head(5).to_dict(orient='records')
        return jsonify({
            "success": True,
            "results": results,
            "data_preview": preview,
            "columns": df.columns.tolist(),
            "features_used": processed_features,
            "prediction_schema": prediction_schema
        })
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint.
    Expects a JSON payload with a "features" object mapping the originally selected independent feature names
    to user-provided values.
    For numeric features, the value is used directly.
    For categorical features, one-hot encoding is applied using the dummy columns in the prediction_schema.
    """
    global trained_models, prediction_schema, model_mode, label_encoder
    try:
        data = request.json
        if not data or "features" not in data:
            return jsonify({"success": False, "error": "Please provide feature values"}), 400
        
        input_features = data["features"]
        feature_vector = []
        
        for item in prediction_schema:
            name = item["name"]
            if item["type"] == "numeric":
                try:
                    value = float(input_features.get(name, 0))
                except Exception:
                    return jsonify({"success": False, "error": f"Invalid numeric value for {name}"}), 400
                feature_vector.append(value)
            elif item["type"] == "categorical":
                selected = str(input_features.get(name, ""))
                for dummy in item["dummy_columns"]:
                    feature_vector.append(1 if dummy == f"{name}_{selected}" else 0)
            else:
                return jsonify({"success": False, "error": f"Unknown feature type for {name}"}), 400
        
        feature_vector = np.array(feature_vector).reshape(1, -1)
        
        model1 = trained_models.get("Model1")
        model2 = trained_models.get("Model2")
        
        if model1 is None or model2 is None:
            return jsonify({"success": False, "error": "Models not trained yet. Please run a model comparison first."}), 400
        
        # Make predictions
        if model_mode == "regression":
            pred1 = model1.predict(feature_vector)
            pred2 = model2.predict(feature_vector)
            
            return jsonify({
                "success": True,
                "predictions": {
                    "Linear Regression": float(pred1[0]),
                    "Random Forest": float(pred2[0])
                }
            })
        else:  # classification
            pred1 = model1.predict(feature_vector)
            pred2 = model2.predict(feature_vector)
            
            # Get probabilities if available
            try:
                prob1 = model1.predict_proba(feature_vector)
                prob2 = model2.predict_proba(feature_vector)
                
                # Get the probability of the predicted class
                pred1_prob = prob1[0][pred1[0]]
                pred2_prob = prob2[0][pred2[0]]
            except:
                pred1_prob = None
                pred2_prob = None
            
            # Convert predictions to original class labels if label encoder exists
            if label_encoder is not None:
                pred1 = label_encoder.inverse_transform(pred1.astype(int))
                pred2 = label_encoder.inverse_transform(pred2.astype(int))
            
            result = {
                "success": True,
                "predictions": {
                    "Logistic Regression": str(pred1[0]),
                    "Random Forest": str(pred2[0])
                }
            }
            
            # Add probabilities if available
            if pred1_prob is not None:
                result["predictions"]["Logistic Regression Probability"] = float(pred1_prob)
            if pred2_prob is not None:
                result["predictions"]["Random Forest Probability"] = float(pred2_prob)
                
            return jsonify(result)
            
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)