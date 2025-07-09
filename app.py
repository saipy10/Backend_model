from flask import Flask, request, jsonify
import pandas as pd
import xgboost as xgb

# Manual label encoders (must match training-time encoding)
gender_map = {'female': 0, 'male': 1}
education_map = {'High School': 0, 'Bachelor': 1, 'Master': 2}
home_map = {'MORTGAGE': 0, 'OWN': 1, 'RENT': 2}
intent_map = {'EDUCATION': 0, 'MEDICAL': 1, 'PERSONAL': 2}
default_map = {'No': 0, 'Yes': 1}

# Load both models from .json
credit_score_model = xgb.XGBRegressor()
credit_score_model.load_model("models/xgb_credit_model.json")

loan_approval_model = xgb.XGBClassifier()
loan_approval_model.load_model("models/xgb_loan_approval_model.json")

# Create Flask app
app = Flask(__name__)

def manual_encode(input_json):
    """Apply manual label encoding for incoming user JSON"""
    return {
        "person_age": input_json["person_age"],
        "person_gender": "female",
        "person_education": education_map[input_json["person_education"]],
        "person_income": input_json["person_income"],
        "person_emp_exp": input_json["person_emp_exp"],
        "person_home_ownership": home_map[input_json["person_home_ownership"]],
        "loan_amnt": input_json["loan_amnt"],
        "loan_intent": intent_map[input_json["loan_intent"]],
        "loan_int_rate": input_json["loan_int_rate"],
        "loan_percent_income": input_json["loan_percent_income"],
        "cb_person_cred_hist_length": input_json["cb_person_cred_hist_length"],
        "previous_loan_defaults_on_file": default_map[input_json["previous_loan_defaults_on_file"]]
    }

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get and preprocess input
        user_input = request.get_json()
        encoded_input = manual_encode(user_input)
        input_df = pd.DataFrame([encoded_input])

        # Predict credit score
        credit_score = credit_score_model.predict(input_df)[0]

        # Add predicted credit score as a feature
        input_df["predicted_credit_score"] = credit_score

        # Predict loan approval (binary classification)
        loan_status = loan_approval_model.predict(input_df)[0]

        return jsonify({
            "predicted_credit_score": round(float(credit_score), 2),
            "loan_approved": bool(loan_status)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/')
def home():
    return "Welcome to the Loan Approval & Credit Score Prediction API"

if __name__ == '__main__':
    app.run(debug=True)
