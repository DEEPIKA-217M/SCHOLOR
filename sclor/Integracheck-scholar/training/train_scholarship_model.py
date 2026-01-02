# train_scholarship_model.py
# Trains a model to predict scholarship eligibility based on student data

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("SCHOLARSHIP ELIGIBILITY MODEL TRAINING")
print("="*60)

# Load dataset
DATASET_PATH = r'c:\Users\rithi\Downloads\scholarship_application_mock_100_updated.csv'
df = pd.read_csv(DATASET_PATH)

print(f"\n📊 Dataset loaded: {len(df)} records")
print(f"📋 Columns: {len(df.columns)}")

# Display column info
print("\n📌 Dataset Columns:")
for col in df.columns:
    print(f"   - {col}")

# Create eligibility criteria based on multiple factors
# Scholarship eligibility rules:
# 1. Lower income families get priority
# 2. Higher marks get priority
# 3. Reserved categories (SC/ST/OBC) get consideration
# 4. Orphan/Disabled applicants get priority

def calculate_eligibility(row):
    score = 0
    
    # Income factor (lower income = more eligible)
    if row['Total_Annual_Income'] < 100000:
        score += 30
    elif row['Total_Annual_Income'] < 150000:
        score += 25
    elif row['Total_Annual_Income'] < 200000:
        score += 20
    elif row['Total_Annual_Income'] < 250000:
        score += 15
    else:
        score += 5
    
    # Academic performance (HSC marks)
    if row['Last_Exam_Marks'] >= 90:
        score += 30
    elif row['Last_Exam_Marks'] >= 80:
        score += 25
    elif row['Last_Exam_Marks'] >= 70:
        score += 20
    elif row['Last_Exam_Marks'] >= 60:
        score += 15
    else:
        score += 10
    
    # SSC marks
    if row['SSC_Marks_Obtained'] >= 450:
        score += 15
    elif row['SSC_Marks_Obtained'] >= 400:
        score += 10
    else:
        score += 5
    
    # Category reservation
    if row['Caste_Category'] in ['SC', 'ST']:
        score += 15
    elif row['Caste_Category'] == 'OBC':
        score += 10
    
    # Special categories
    if row['Orphan'] == 'YES':
        score += 10
    if row['Disabled'] == 'YES':
        score += 10
    
    # Eligibility: score >= 60 is Eligible
    return 'Eligible' if score >= 60 else 'Not Eligible'

# Create target variable
df['Eligibility'] = df.apply(calculate_eligibility, axis=1)

print(f"\n📈 Eligibility Distribution:")
print(df['Eligibility'].value_counts())

# Feature Engineering
print("\n⚙️ Preparing features...")

# Select features for training
feature_columns = [
    'SSC_Marks_Obtained',
    'Last_Exam_Marks',
    'Total_Annual_Income',
    'Applicant_Annual_Income',
    'Caste_Category',
    'Gender',
    'Orphan',
    'Disabled',
    'Resident_of_Maharashtra'
]

# Create a copy for processing
df_model = df[feature_columns + ['Eligibility']].copy()

# Encode categorical variables
label_encoders = {}
categorical_cols = ['Caste_Category', 'Gender', 'Orphan', 'Disabled', 'Resident_of_Maharashtra']

for col in categorical_cols:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col].astype(str))
    label_encoders[col] = le

# Features and target
X = df_model.drop('Eligibility', axis=1)
y = df_model['Eligibility']

# Encode target
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)

print(f"\n📊 Feature Matrix Shape: {X.shape}")
print(f"📊 Features used: {list(X.columns)}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\n📊 Training set: {len(X_train)} samples")
print(f"📊 Test set: {len(X_test)} samples")

# Scale numerical features
scaler = StandardScaler()
numerical_cols = ['SSC_Marks_Obtained', 'Last_Exam_Marks', 'Total_Annual_Income', 'Applicant_Annual_Income']
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Train multiple models
print("\n🤖 Training Models...")
print("-"*40)

models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
}

best_model = None
best_accuracy = 0
best_model_name = ""

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n📌 {name}:")
    print(f"   Accuracy: {accuracy*100:.2f}%")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = name

print(f"\n🏆 Best Model: {best_model_name} with {best_accuracy*100:.2f}% accuracy")

# Detailed evaluation of best model
print(f"\n📊 Detailed Evaluation - {best_model_name}")
print("-"*40)
y_pred_best = best_model.predict(X_test_scaled)
print("\nClassification Report:")
print(classification_report(y_test, y_pred_best, target_names=target_encoder.classes_))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred_best)
print(cm)

# Feature Importance (for tree-based models)
if hasattr(best_model, 'feature_importances_'):
    print("\n📊 Feature Importance:")
    importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    for _, row in importance.iterrows():
        bar = '█' * int(row['Importance'] * 50)
        print(f"   {row['Feature']:<25} {bar} {row['Importance']:.3f}")

# Save models and encoders
print("\n💾 Saving models...")
joblib.dump(best_model, 'scholarship_model.joblib')
joblib.dump(scaler, 'scholarship_scaler.joblib')
joblib.dump(label_encoders, 'scholarship_label_encoders.joblib')
joblib.dump(target_encoder, 'scholarship_target_encoder.joblib')

print("   ✅ scholarship_model.joblib")
print("   ✅ scholarship_scaler.joblib")
print("   ✅ scholarship_label_encoders.joblib")
print("   ✅ scholarship_target_encoder.joblib")

# Test prediction function
print("\n🧪 Testing Prediction Function...")
print("-"*40)

def predict_eligibility(applicant_data):
    """
    Predict scholarship eligibility for an applicant
    
    Parameters:
    - SSC_Marks_Obtained: SSC marks (out of 500)
    - Last_Exam_Marks: HSC/Last exam marks (percentage)
    - Total_Annual_Income: Family annual income
    - Applicant_Annual_Income: Applicant's own income
    - Caste_Category: SC/ST/OBC/General
    - Gender: Male/Female
    - Orphan: YES/NO
    - Disabled: YES/NO
    - Resident_of_Maharashtra: YES/NO
    """
    # Create DataFrame
    df_input = pd.DataFrame([applicant_data])
    
    # Encode categorical
    for col in categorical_cols:
        if col in label_encoders:
            try:
                df_input[col] = label_encoders[col].transform(df_input[col].astype(str))
            except ValueError:
                # Handle unseen labels
                df_input[col] = 0
    
    # Scale numerical
    df_input[numerical_cols] = scaler.transform(df_input[numerical_cols])
    
    # Predict
    prediction = best_model.predict(df_input)[0]
    probability = best_model.predict_proba(df_input)[0]
    
    result = target_encoder.inverse_transform([prediction])[0]
    
    return {
        'eligibility': result,
        'confidence': max(probability) * 100,
        'eligible_probability': probability[1] * 100 if len(probability) > 1 else probability[0] * 100
    }

# Test cases
test_cases = [
    {
        'name': 'High Merit, Low Income Student',
        'data': {
            'SSC_Marks_Obtained': 480,
            'Last_Exam_Marks': 95,
            'Total_Annual_Income': 80000,
            'Applicant_Annual_Income': 0,
            'Caste_Category': 'SC',
            'Gender': 'Female',
            'Orphan': 'YES',
            'Disabled': 'NO',
            'Resident_of_Maharashtra': 'YES'
        }
    },
    {
        'name': 'Average Merit, High Income Student',
        'data': {
            'SSC_Marks_Obtained': 400,
            'Last_Exam_Marks': 70,
            'Total_Annual_Income': 300000,
            'Applicant_Annual_Income': 0,
            'Caste_Category': 'General',
            'Gender': 'Male',
            'Orphan': 'NO',
            'Disabled': 'NO',
            'Resident_of_Maharashtra': 'NO'
        }
    },
    {
        'name': 'Good Merit, Medium Income, OBC Student',
        'data': {
            'SSC_Marks_Obtained': 450,
            'Last_Exam_Marks': 82,
            'Total_Annual_Income': 150000,
            'Applicant_Annual_Income': 0,
            'Caste_Category': 'OBC',
            'Gender': 'Female',
            'Orphan': 'NO',
            'Disabled': 'NO',
            'Resident_of_Maharashtra': 'YES'
        }
    }
]

for test in test_cases:
    result = predict_eligibility(test['data'])
    print(f"\n👤 {test['name']}:")
    print(f"   📌 Eligibility: {result['eligibility']}")
    print(f"   📊 Confidence: {result['confidence']:.1f}%")

print("\n" + "="*60)
print("✅ MODEL TRAINING COMPLETE!")
print("="*60)
print("\nThe model can now predict scholarship eligibility based on:")
print("  • Academic performance (SSC & HSC marks)")
print("  • Financial background (Annual income)")
print("  • Social category (SC/ST/OBC/General)")
print("  • Special circumstances (Orphan/Disabled)")
print("  • Residency status")
print("\n")
