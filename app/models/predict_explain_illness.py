import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import shap


class DiseasePredictor:
    """
    A class for predicting diseases based on symptoms using CatBoost with explainability.
    """

    def __init__(self, iterations=500, learning_rate=0.05, depth=6, random_seed=42):
        """
        Initialize the DiseasePredictor with model parameters.

        Args:
            iterations: Number of boosting iterations
            learning_rate: Learning rate for gradient descent
            depth: Depth of the trees
            random_seed: Random seed for reproducibility
        """
        self.model = CatBoostClassifier(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            loss_function='MultiClass',
            eval_metric='Accuracy',
            random_seed=random_seed
            # verbose=100
        )
        self.feature_names = None
        self.trained = False

    def train(self, X, y, test_size=0.2, random_state=42):
        """
        Train the model on the provided public.

        Args:
            X: Features dataframe with symptoms
            y: Target series with disease labels
            test_size: Proportion of public to use for validation
            random_state: Random state for train-test split

        Returns:
            Dictionary with training metrics
        """
        # Store feature names
        self.feature_names = X.columns.tolist()

        # Split the public
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Train the model
        self.model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50
        )

        # Get predictions
        val_preds = self.model.predict(X_val)

        # Evaluate
        accuracy = accuracy_score(y_val, val_preds)
        report = classification_report(y_val, val_preds, output_dict=True)

        # Store validation public for later use
        self.X_val = X_val
        self.y_val = y_val
        self.trained = True

        return {
            "accuracy": accuracy,
            "report": report,
            "validation_data": {
                "X_val": X_val,
                "y_val": y_val,
                "predictions": val_preds
            }
        }

    def plot_feature_importance(self, top_n=20, figsize=(12, 8), save_path="../public/feature_importance.png"):
        """
        Plot the top N most important features.

        Args:
            top_n: Number of top features to show
            figsize: Figure size as (width, height)
            save_path: Path to save the figure (optional)

        Returns:
            DataFrame with feature importances
        """
        if not self.trained:
            raise ValueError("Model must be trained before plotting feature importance")

        # Get feature importances
        feature_importances = self.model.get_feature_importance()
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': feature_importances
        })
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        # Plot
        plt.figure(figsize=figsize)
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(top_n))
        plt.title(f'Top {top_n} Most Important Symptoms for Disease Prediction')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)

        plt.show()

        return importance_df

    def explain_prediction(self, patient_features):
        """
        Generate explanation for a patient's prediction.

        Args:
            patient_features: DataFrame with a single row containing the patient's symptoms

        Returns:
            A dictionary with prediction details and top contributing symptoms
        """
        if not self.trained:
            raise ValueError("Model must be trained before explaining predictions")

        # Make prediction
        pred_class = self.model.predict(patient_features)[0]
        pred_probs = self.model.predict_proba(patient_features)[0]
        disease_name = pred_class

        # Get prediction confidence
        class_names = self.model.classes_
        class_index = np.where(class_names == pred_class)[0][0]
        confidence = pred_probs[class_index] * 100

        # Get SHAP values for this prediction
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(patient_features)

        # Handle case where shap_values has shape (1, n_features, n_classes)
        if len(shap_values.shape) == 3:
            # Reshape: (1, n_features, n_classes) -> (n_features,)
            class_shap_values = shap_values[0, :, class_index]
        else:
            # For traditional shap output
            class_shap_values = shap_values[class_index][0]

        # Combine feature names and SHAP values
        feature_shap = list(zip(self.feature_names, class_shap_values))

        # Sort by absolute SHAP value (importance)
        feature_shap.sort(key=lambda x: abs(x[1]), reverse=True)

        # Get top contributing symptoms (both positive and negative)
        top_positive = [(name, value) for name, value in feature_shap if value > 0][:5]
        top_negative = [(name, value) for name, value in feature_shap if value < 0][:5]

        return {
            "predicted_disease": disease_name,
            "confidence": f"{confidence:.2f}%",
            "supporting_symptoms": top_positive,
            "contradicting_symptoms": top_negative,
            "all_probabilities": dict(zip(class_names, pred_probs))
        }

    def create_shap_summary(self, n_samples=100, figsize=(12, 8), save_path="../public/shap_summary.png"):
        """
        Generate SHAP summary plot for global model interpretability.

        Args:
            n_samples: Number of samples to use for SHAP analysis
            figsize: Figure size as (width, height)
            save_path: Path to save the figure (optional)
        """
        if not self.trained:
            raise ValueError("Model must be trained before creating SHAP summary")

        # Take a subset of validation public for SHAP analysis
        shap_data = self.X_val.iloc[:n_samples]

        # Create explainer
        explainer = shap.TreeExplainer(self.model)

        # Calculate SHAP values
        shap_values = explainer.shap_values(shap_data)

        # Handle different SHAP output formats
        if len(np.array(shap_values).shape) == 3:
            # For shape (1, n_features, n_classes)
            # We'll use the first class for the summary plot
            print("Using first class for summary plot due to 3D SHAP values")
            plt.figure(figsize=figsize)

            # Extract values for the first class
            class_values = shap_values[0, :, 0]
            # Reshape to match shap.summary_plot expectations
            reshaped_values = np.array([class_values for _ in range(len(shap_data))])

            shap.summary_plot(reshaped_values, shap_data, feature_names=self.feature_names)
        else:
            # Standard format
            plt.figure(figsize=figsize)
            shap.summary_plot(shap_values[0], shap_data, feature_names=self.feature_names)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)

        plt.show()

    def save_model(self, path='../public/catboost_disease_model.cbm'):
        """
        Save the trained model to disk.

        Args:
            path: Path to save the model
        """
        if not self.trained:
            raise ValueError("Model must be trained before saving")

        self.model.save_model(path)
        print(f"Model saved to {path}")

    def load_model(self, path='../public/catboost_disease_model.cbm'):
        """
        Load a trained model from disk.

        Args:
            path: Path to the saved model
        """
        self.model = CatBoostClassifier()
        self.model.load_model(path)
        self.trained = True
        print(f"Model loaded from {path}")

    def predict_batch(self, patients_data):
        """
        Predict diseases for multiple patients and explain the predictions.

        Args:
            patients_data: DataFrame with multiple patients' symptoms

        Returns:
            List of predictions and explanations
        """
        if not self.trained:
            raise ValueError("Model must be trained before predicting")

        # Make predictions
        predictions = self.model.predict(patients_data)

        # Get explanations for each patient
        explanations = []
        for i in range(len(patients_data)):
            single_patient = patients_data.iloc[[i]]
            explanation = self.explain_prediction(single_patient)
            explanations.append(explanation)

        return predictions, explanations

    def print_explanation(self, explanation):
        """
        Print a nicely formatted explanation.

        Args:
            explanation: The explanation dictionary from explain_prediction
        """
        print(f"\nPrediction Explanation:")
        print(f"Predicted Disease: {explanation['predicted_disease']}")
        print(f"Confidence: {explanation['confidence']}")

        print(f"\nSupporting Symptoms (increasing probability of this disease):")
        for symptom, value in explanation['supporting_symptoms']:
            print(f"- {symptom}: {value:.4f}")

        print(f"\nContradicting Symptoms (decreasing probability of this disease):")
        for symptom, value in explanation['contradicting_symptoms']:
            print(f"- {symptom}: {value:.4f}")

    def get_confusion_matrix(self, figsize=(12, 10), save_path="../public/confusion_matrix.png"):
        """
        Generate and plot a confusion matrix of the validation results.

        Args:
            figsize: Figure size as (width, height)
            save_path: Path to save the figure (optional)

        Returns:
            Confusion matrix as a DataFrame
        """
        if not self.trained:
            raise ValueError("Model must be trained before generating confusion matrix")

        # Get predictions
        y_pred = self.model.predict(self.X_val)

        # Create confusion matrix
        cm = confusion_matrix(self.y_val, y_pred)

        # Convert to DataFrame for better visualization
        cm_df = pd.DataFrame(
            cm,
            index=self.model.classes_,
            columns=self.model.classes_
        )

        # Plot
        plt.figure(figsize=figsize)
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual Disease')
        plt.xlabel('Predicted Disease')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)

        plt.show()

        return cm_df

    def create_model_input(self, symptoms_list, model_feature_names=None):
        """
        Create a DataFrame with the correct feature names for the trained model.

        Args:
            symptoms_list: List of symptoms present in the patient
            model_feature_names: List of feature names expected by the model
                                (can be obtained from predictor.feature_names)

        Returns:
            DataFrame compatible with the trained model
        """
        # Start with your base symptoms dictionary
        symptoms_dict = {'itching': 0, 'skin_rash': 0, 'nodal_skin_eruptions': 0, 'continuous_sneezing': 0,
                         'shivering': 0, 'chills': 0, 'joint_pain': 0, 'stomach_pain': 0, 'acidity': 0,
                         'ulcers_on_tongue': 0,
                         'muscle_wasting': 0, 'vomiting': 0, 'burning_micturition': 0, 'spotting_ urination': 0,
                         'fatigue': 0,
                         'weight_gain': 0, 'anxiety': 0, 'cold_hands_and_feets': 0, 'mood_swings': 0, 'weight_loss': 0,
                         'restlessness': 0, 'lethargy': 0, 'patches_in_throat': 0, 'irregular_sugar_level': 0, 'cough': 0,
                         'high_fever': 0, 'sunken_eyes': 0, 'breathlessness': 0, 'sweating': 0, 'dehydration': 0,
                         'indigestion': 0, 'headache': 0, 'yellowish_skin': 0, 'dark_urine': 0, 'nausea': 0,
                         'loss_of_appetite': 0,
                         'pain_behind_the_eyes': 0, 'back_pain': 0, 'constipation': 0, 'abdominal_pain': 0, 'diarrhoea': 0,
                         'mild_fever': 0,
                         'yellow_urine': 0, 'yellowing_of_eyes': 0, 'acute_liver_failure': 0, 'fluid_overload': 0,
                         'swelling_of_stomach': 0,
                         'swelled_lymph_nodes': 0, 'malaise': 0, 'blurred_and_distorted_vision': 0, 'phlegm': 0,
                         'throat_irritation': 0,
                         'redness_of_eyes': 0, 'sinus_pressure': 0, 'runny_nose': 0, 'congestion': 0, 'chest_pain': 0,
                         'weakness_in_limbs': 0,
                         'fast_heart_rate': 0, 'pain_during_bowel_movements': 0, 'pain_in_anal_region': 0,
                         'bloody_stool': 0,
                         'irritation_in_anus': 0, 'neck_pain': 0, 'dizziness': 0, 'cramps': 0, 'bruising': 0, 'obesity': 0,
                         'swollen_legs': 0,
                         'swollen_blood_vessels': 0, 'puffy_face_and_eyes': 0, 'enlarged_thyroid': 0, 'brittle_nails': 0,
                         'swollen_extremeties': 0,
                         'excessive_hunger': 0, 'extra_marital_contacts': 0, 'drying_and_tingling_lips': 0,
                         'slurred_speech': 0,
                         'knee_pain': 0, 'hip_joint_pain': 0, 'muscle_weakness': 0, 'stiff_neck': 0, 'swelling_joints': 0,
                         'movement_stiffness': 0,
                         'spinning_movements': 0, 'loss_of_balance': 0, 'unsteadiness': 0, 'weakness_of_one_body_side': 0,
                         'loss_of_smell': 0,
                         'bladder_discomfort': 0, 'foul_smell_of urine': 0, 'continuous_feel_of_urine': 0,
                         'passage_of_gases': 0, 'internal_itching': 0,
                         'toxic_look_(typhos)': 0, 'depression': 0, 'irritability': 0, 'muscle_pain': 0,
                         'altered_sensorium': 0,
                         'red_spots_over_body': 0, 'belly_pain': 0, 'abnormal_menstruation': 0, 'dischromic _patches': 0,
                         'watering_from_eyes': 0,
                         'increased_appetite': 0, 'polyuria': 0, 'family_history': 0, 'mucoid_sputum': 0, 'rusty_sputum': 0,
                         'lack_of_concentration': 0,
                         'visual_disturbances': 0, 'receiving_blood_transfusion': 0, 'receiving_unsterile_injections': 0,
                         'coma': 0,
                         'stomach_bleeding': 0, 'distention_of_abdomen': 0, 'history_of_alcohol_consumption': 0,
                         'fluid_overload.0': 0,
                         'blood_in_sputum': 0, 'prominent_veins_on_calf': 0, 'palpitations': 0, 'painful_walking': 0,
                         'pus_filled_pimples': 0,
                         'blackheads': 0, 'scurring': 0, 'skin_peeling': 0, 'silver_like_dusting': 0,
                         'small_dents_in_nails': 0, 'inflammatory_nails': 0,
                         'blister': 0, 'red_sore_around_nose': 0, 'yellow_crust_ooze': 0}

        # Set values for symptoms in the list
        for symptom in symptoms_list:
            if symptom in symptoms_dict:
                symptoms_dict[symptom] = 1
                print(f"Symptom '{symptom}' set to 1")

        # Create an initial DataFrame
        input_df = pd.DataFrame(columns=list(symptoms_dict.keys()))
        input_df.loc[0] = np.array(list(symptoms_dict.values()))

        # If model feature names are provided, ensure compatibility
        if model_feature_names is not None:
            # Option 1: Create a new DataFrame with the exact expected columns
            corrected_df = pd.DataFrame(columns=model_feature_names)
            corrected_df.loc[0] = 0  # Initialize with zeros

            # Transfer values from input_df to corrected_df where column names match
            for col in model_feature_names:
                # Check if the column exists in our input data
                if col in input_df.columns:
                    corrected_df[col] = input_df[col]
                else:
                    print(f"Warning: Model expects feature '{col}' which wasn't in input data. Setting to 0.")

            return corrected_df

        return input_df



# Example usage:

# Create the predictor
predictor = DiseasePredictor()

## Train the model (assuming train_df is already loaded)
train_df = pd.read_csv('../public/training_data.csv')
train_x = train_df.iloc[:, 0:132]  # Symptoms
train_y = train_df.iloc[:, 132]    # Disease labels
#
# metrics = predictor.train(train_x, train_y)
# print(f"Validation Accuracy: {metrics['accuracy']:.4f}")
#
# # Plot feature importance
# importance_df = predictor.plot_feature_importance()
#
# # Explain a sample prediction
# sample_index = 0
# sample_patient = predictor.X_val.iloc[[sample_index]].copy()
# actual_disease = predictor.y_val.iloc[sample_index]
#
# explanation = predictor.explain_prediction(sample_patient)
# predictor.print_explanation(explanation)
# print(f"Actual Disease: {actual_disease}")
#
# # Create SHAP summary
# predictor.create_shap_summary()
# predictor.get_confusion_matrix()
#
# # Save the model
# predictor.save_model()

# Load the model (in a new session)
new_predictor = DiseasePredictor()
new_predictor.load_model()

new_predictor.feature_names = train_x.columns.tolist()
model_feature_names = new_predictor.feature_names


# with open("features_names.txt", "w") as file:
#     file.write(", ".join(map(str, new_predictor.feature_names)))

# Get the second row from your training data (index 1)
second_row = train_x.iloc[[5]]  # Double brackets to keep it as a DataFrame

# print(second_row)

user_input_df = new_predictor.create_model_input(['chest_pain', 'breathlessness'], model_feature_names)

# print(user_input_df)
# # Make prediction and get explanation
explanation = new_predictor.explain_prediction(user_input_df)
#
# # Print the explanation
#
new_predictor.print_explanation(explanation)
#
# # If you want to see the actual disease for this row (for comparison)
# actual_disease = train_y.iloc[1]
# print(f"\nActual Disease: {actual_disease}")
#
# # If you want to get just the raw prediction without explanation
prediction = new_predictor.model.predict(user_input_df)[0]
print(f"Raw Prediction: {prediction}")
