"""
IntegaCheck Scholar - Backend API
Comprehensive text analysis for AI detection, plagiarism check, and scholarly quality
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
import sys

# Add utils to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.text_features import (
    extract_text_features, 
    calculate_readability_score, 
    get_readability_level,
    detect_repetitive_patterns
)
from utils.citation_check import generate_citation_report

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Load trained model and vectorizer
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'training', 'scholarly_model.joblib')
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'training', 'vectorizer.joblib')

model = None
vectorizer = None

def load_model():
    """Load the ML model and vectorizer"""
    global model, vectorizer
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print(f"Warning: Could not load model - {e}")
        return False

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Main analysis endpoint
    Returns comprehensive analysis including:
    - AI/Human detection with probability
    - Plagiarism check
    - Citation analysis
    - Text quality metrics
    - Readability scores
    - Recommendations
    """
    data = request.json
    text = data.get('text', '')
    
    if not text or len(text.strip()) < 10:
        return jsonify({'error': 'Please provide text with at least 10 characters'}), 400
    
    # 1. AI/Human Detection
    ai_result = detect_ai_content(text)
    
    # 2. Text Feature Analysis
    text_features = extract_text_features(text)
    
    # 3. Readability Analysis
    readability_score = calculate_readability_score(text)
    readability_level = get_readability_level(readability_score)
    
    # 4. Repetitive Pattern Detection
    patterns = detect_repetitive_patterns(text)
    
    # 5. Citation & Plagiarism Check
    citation_report = generate_citation_report(text)
    
    # 6. Overall Quality Score
    quality_score = calculate_overall_quality(ai_result, citation_report, text_features)
    
    # 7. Generate Summary and Recommendations
    summary = generate_summary(ai_result, citation_report, text_features, readability_level)
    
    return jsonify({
        'success': True,
        'analysis': {
            # AI Detection Results
            'ai_detection': {
                'prediction': ai_result['prediction'],
                'ai_probability': ai_result['ai_probability'],
                'human_probability': ai_result['human_probability'],
                'confidence': ai_result['confidence'],
                'verdict': ai_result['verdict']
            },
            
            # Plagiarism & Citation Results
            'plagiarism_check': {
                'originality_score': citation_report['plagiarism_analysis']['originality_score'],
                'plagiarism_risk': citation_report['plagiarism_analysis']['plagiarism_risk'],
                'common_phrases_detected': len(citation_report['plagiarism_analysis']['common_phrases_found'])
            },
            
            # Citation Analysis
            'citation_analysis': {
                'total_citations': citation_report['citation_details']['total_citations'],
                'has_citations': citation_report['citation_details']['has_citations'],
                'citation_issues': citation_report['citation_details']['citation_issues'],
                'quotes_without_citation': len(citation_report['citation_details']['quotes_without_citation'])
            },
            
            # Text Quality Metrics
            'text_quality': {
                'word_count': text_features['word_count'],
                'sentence_count': text_features['sentence_count'],
                'avg_word_length': text_features['avg_word_length'],
                'avg_sentence_length': text_features['avg_sentence_length'],
                'vocabulary_richness': text_features['vocabulary_richness'],
                'readability_score': readability_score,
                'readability_level': readability_level
            },
            
            # Writing Style Indicators
            'style_indicators': {
                'formal_word_count': text_features['formal_word_count'],
                'personal_pronoun_count': text_features['personal_pronoun_count'],
                'contraction_count': text_features['contraction_count'],
                'exclamation_count': text_features['exclamation_count'],
                'repeated_patterns': patterns['repeated_phrase_count']
            },
            
            # Overall Scores
            'scores': {
                'overall_quality': quality_score,
                'ai_score': ai_result['ai_probability'],
                'originality_score': citation_report['plagiarism_analysis']['originality_score'],
                'citation_quality': citation_report['overall_quality_score']
            },
            
            # Summary and Recommendations
            'summary': summary,
            'recommendations': citation_report['recommendations']
        }
    })

def detect_ai_content(text):
    """Detect if content is AI-generated or human-written"""
    global model, vectorizer
    
    if model is None or vectorizer is None:
        # Fallback heuristic-based detection if model not loaded
        return heuristic_ai_detection(text)
    
    try:
        # Use trained model
        text_vectorized = vectorizer.transform([text])
        prediction = model.predict(text_vectorized)[0]
        probabilities = model.predict_proba(text_vectorized)[0]
        
        # Get class indices
        classes = model.classes_
        ai_idx = list(classes).index('ai') if 'ai' in classes else 0
        human_idx = list(classes).index('human') if 'human' in classes else 1
        
        ai_prob = round(probabilities[ai_idx] * 100, 1)
        human_prob = round(probabilities[human_idx] * 100, 1)
        
        confidence = 'High' if max(ai_prob, human_prob) > 80 else 'Medium' if max(ai_prob, human_prob) > 60 else 'Low'
        
        if prediction == 'ai':
            verdict = f"This text appears to be AI-generated ({ai_prob}% confidence)"
        else:
            verdict = f"This text appears to be human-written ({human_prob}% confidence)"
        
        return {
            'prediction': prediction,
            'ai_probability': ai_prob,
            'human_probability': human_prob,
            'confidence': confidence,
            'verdict': verdict
        }
    except Exception as e:
        print(f"Model prediction error: {e}")
        return heuristic_ai_detection(text)

def heuristic_ai_detection(text):
    """Fallback heuristic-based AI detection"""
    features = extract_text_features(text)
    
    ai_score = 0
    
    # AI indicators
    if features['formal_word_count'] > 3:
        ai_score += 20
    if features['personal_pronoun_count'] == 0:
        ai_score += 15
    if features['contraction_count'] == 0:
        ai_score += 15
    if features['exclamation_count'] == 0:
        ai_score += 10
    if features['avg_sentence_length'] > 20:
        ai_score += 15
    if features['vocabulary_richness'] < 0.5:
        ai_score += 10
    
    # Human indicators
    if features['personal_pronoun_count'] > 2:
        ai_score -= 20
    if features['contraction_count'] > 1:
        ai_score -= 15
    if features['exclamation_count'] > 0:
        ai_score -= 10
    
    ai_prob = max(0, min(100, 50 + ai_score))
    human_prob = 100 - ai_prob
    
    prediction = 'ai' if ai_prob > 50 else 'human'
    confidence = 'High' if abs(ai_prob - 50) > 30 else 'Medium' if abs(ai_prob - 50) > 15 else 'Low'
    
    if prediction == 'ai':
        verdict = f"This text shows AI-like characteristics ({ai_prob}% probability)"
    else:
        verdict = f"This text shows human-like characteristics ({human_prob}% probability)"
    
    return {
        'prediction': prediction,
        'ai_probability': round(ai_prob, 1),
        'human_probability': round(human_prob, 1),
        'confidence': confidence,
        'verdict': verdict
    }

def calculate_overall_quality(ai_result, citation_report, text_features):
    """Calculate overall text quality score"""
    score = 100
    
    # Penalize potential AI content
    if ai_result['prediction'] == 'ai':
        score -= (ai_result['ai_probability'] - 50) * 0.3
    
    # Consider originality
    originality = citation_report['plagiarism_analysis']['originality_score']
    score = (score + originality) / 2
    
    # Consider citation quality
    citation_quality = citation_report['overall_quality_score']
    score = (score * 0.7) + (citation_quality * 0.3)
    
    return round(max(0, min(100, score)), 1)

def generate_summary(ai_result, citation_report, text_features, readability):
    """Generate a human-readable summary"""
    summary = []
    
    # AI Detection Summary
    if ai_result['prediction'] == 'ai':
        summary.append(f"⚠️ AI Detection: Text appears to be AI-generated with {ai_result['ai_probability']}% probability.")
    else:
        summary.append(f"✅ AI Detection: Text appears to be human-written with {ai_result['human_probability']}% probability.")
    
    # Plagiarism Summary
    risk = citation_report['plagiarism_analysis']['plagiarism_risk']
    originality = citation_report['plagiarism_analysis']['originality_score']
    if risk == 'High':
        summary.append(f"🔴 Originality: {originality}% - High plagiarism risk detected.")
    elif risk == 'Medium':
        summary.append(f"🟡 Originality: {originality}% - Some common phrases detected.")
    else:
        summary.append(f"🟢 Originality: {originality}% - Content appears original.")
    
    # Citation Summary
    if citation_report['citation_details']['has_citations']:
        summary.append(f"📚 Citations: {citation_report['citation_details']['total_citations']} citation(s) found.")
    else:
        summary.append("📚 Citations: No citations detected. Consider adding references.")
    
    # Readability Summary
    summary.append(f"📖 Readability: {readability}")
    
    # Word Count
    summary.append(f"📝 Length: {text_features['word_count']} words, {text_features['sentence_count']} sentences")
    
    return summary

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/quick-check', methods=['POST'])
def quick_check():
    """Quick AI detection endpoint (lighter weight)"""
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    result = detect_ai_content(text)
    return jsonify(result)

if __name__ == "__main__":
    print("Loading IntegaCheck Scholar Backend...")
    load_model()
    print("Starting server on http://localhost:5000")
    app.run(debug=False, host='127.0.0.1', port=5000)
