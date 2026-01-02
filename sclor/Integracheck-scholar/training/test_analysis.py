# test_analysis.py
# Test script to demonstrate the ML model capabilities

import sys
import os

# Add the backend folder to path
backend_path = os.path.join(os.path.dirname(__file__), '..', 'css', 'js', 'assets', 'backend')
sys.path.insert(0, backend_path)

import joblib
import json
from utils.text_features import (
    extract_text_features, 
    calculate_readability_score, 
    get_readability_level,
    detect_repetitive_patterns
)
from utils.citation_check import generate_citation_report

# Load model
model_path = os.path.join(os.path.dirname(__file__), 'scholarly_model.joblib')
vectorizer_path = os.path.join(os.path.dirname(__file__), 'vectorizer.joblib')

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

def get_progress_bar(value, max_val=100, length=20):
    """Create a visual progress bar"""
    filled = int((value / max_val) * length)
    empty = length - filled
    if value >= 70:
        return f"[{'█' * filled}{'░' * empty}] {value}% ✅"
    elif value >= 40:
        return f"[{'█' * filled}{'░' * empty}] {value}% ⚠️"
    else:
        return f"[{'█' * filled}{'░' * empty}] {value}% ❌"

def get_risk_indicator(level):
    """Get visual risk indicator"""
    if level == 'Low':
        return "🟢 LOW RISK"
    elif level == 'Medium':
        return "🟡 MEDIUM RISK"
    else:
        return "🔴 HIGH RISK"

def analyze_text(text):
    """Complete analysis of the text with easy-to-understand report"""
    
    # Get all analysis data
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)[0]
    probabilities = model.predict_proba(text_vectorized)[0]
    
    classes = model.classes_
    ai_idx = list(classes).index('ai') if 'ai' in classes else 0
    human_idx = list(classes).index('human') if 'human' in classes else 1
    
    ai_prob = round(probabilities[ai_idx] * 100, 1)
    human_prob = round(probabilities[human_idx] * 100, 1)
    
    features = extract_text_features(text)
    readability = calculate_readability_score(text)
    level = get_readability_level(readability)
    citation_report = generate_citation_report(text)
    patterns = detect_repetitive_patterns(text)
    
    # Calculate overall score
    originality = citation_report['plagiarism_analysis']['originality_score']
    overall_score = round((human_prob + originality) / 2, 1)
    
    # Print Report
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 15 + "📋 INTEGRACHECK SCHOLAR" + " " * 20 + "║")
    print("║" + " " * 15 + "   TEXT ANALYSIS REPORT" + " " * 20 + "║")
    print("╚" + "═" * 58 + "╝")
    
    # Quick Summary Box
    print("\n┌─────────────────────────────────────────────────────────┐")
    print("│                    📊 QUICK SUMMARY                     │")
    print("├─────────────────────────────────────────────────────────┤")
    
    if prediction == 'ai':
        print(f"│  🤖 THIS TEXT APPEARS TO BE: AI-GENERATED              │")
    else:
        print(f"│  👤 THIS TEXT APPEARS TO BE: HUMAN-WRITTEN             │")
    
    print(f"│                                                         │")
    print(f"│  Overall Score: {get_progress_bar(overall_score, 100, 15):40} │")
    print("└─────────────────────────────────────────────────────────┘")
    
    # Main Scores
    print("\n┌─────────────────────────────────────────────────────────┐")
    print("│                    📈 MAIN SCORES                       │")
    print("├─────────────────────────────────────────────────────────┤")
    print(f"│                                                         │")
    print(f"│  Human Written Score                                    │")
    print(f"│  {get_progress_bar(human_prob, 100, 25):55} │")
    print(f"│                                                         │")
    print(f"│  AI Generated Score                                     │")
    print(f"│  {get_progress_bar(ai_prob, 100, 25):55} │")
    print(f"│                                                         │")
    print(f"│  Originality Score                                      │")
    print(f"│  {get_progress_bar(originality, 100, 25):55} │")
    print("└─────────────────────────────────────────────────────────┘")
    
    # What This Means
    print("\n┌─────────────────────────────────────────────────────────┐")
    print("│                 💡 WHAT THIS MEANS                      │")
    print("├─────────────────────────────────────────────────────────┤")
    
    if prediction == 'ai':
        if ai_prob > 80:
            print("│  ⚠️  High chance this text was written by AI.          │")
            print("│      Consider rewriting in your own words.             │")
        elif ai_prob > 60:
            print("│  ⚠️  This text shows some AI-like patterns.            │")
            print("│      Add more personal touch to make it authentic.     │")
        else:
            print("│  ℹ️  Text shows slight AI patterns but mostly okay.    │")
    else:
        if human_prob > 80:
            print("│  ✅  Great! This text appears genuinely human-written. │")
            print("│      Your writing style is natural and authentic.      │")
        elif human_prob > 60:
            print("│  ✅  Good! Text appears mostly human-written.          │")
        else:
            print("│  ℹ️  Text is acceptable but could be more natural.     │")
    
    print("└─────────────────────────────────────────────────────────┘")
    
    # Plagiarism Check
    plag_risk = citation_report['plagiarism_analysis']['plagiarism_risk']
    print("\n┌─────────────────────────────────────────────────────────┐")
    print("│                 🔍 PLAGIARISM CHECK                     │")
    print("├─────────────────────────────────────────────────────────┤")
    print(f"│                                                         │")
    print(f"│  Risk Level: {get_risk_indicator(plag_risk):43} │")
    print(f"│  Originality: {originality}%                                       │")
    
    common_phrases = len(citation_report['plagiarism_analysis']['common_phrases_found'])
    if common_phrases > 0:
        print(f"│  ⚠️  Found {common_phrases} common/overused phrase(s)                 │")
    else:
        print(f"│  ✅  No common overused phrases detected                │")
    print("└─────────────────────────────────────────────────────────┘")
    
    # Citation Analysis
    cite = citation_report['citation_details']
    print("\n┌─────────────────────────────────────────────────────────┐")
    print("│                 📚 CITATION ANALYSIS                    │")
    print("├─────────────────────────────────────────────────────────┤")
    
    if cite['total_citations'] > 0:
        print(f"│  ✅  Found {cite['total_citations']} citation(s) in your text                  │")
        if cite['apa_citations'] > 0:
            print(f"│      • APA Style: {cite['apa_citations']} citation(s)                         │")
        if cite['mla_citations'] > 0:
            print(f"│      • MLA Style: {cite['mla_citations']} citation(s)                         │")
        if cite['ieee_citations'] > 0:
            print(f"│      • IEEE Style: {cite['ieee_citations']} citation(s)                        │")
    else:
        print(f"│  ⚠️  No citations found in your text                   │")
        print(f"│      Consider adding references to support your claims │")
    
    if cite['citation_issues']:
        print(f"│  ❌  Some quotes may be missing proper citations        │")
    
    print("└─────────────────────────────────────────────────────────┘")
    
    # Text Statistics (Simple)
    print("\n┌─────────────────────────────────────────────────────────┐")
    print("│                 📝 TEXT STATISTICS                      │")
    print("├─────────────────────────────────────────────────────────┤")
    print(f"│                                                         │")
    print(f"│  📄 Word Count:        {features['word_count']:<10}                      │")
    print(f"│  📃 Sentence Count:    {features['sentence_count']:<10}                      │")
    print(f"│  📖 Reading Level:     {level[:25]:<25}   │")
    print(f"│                                                         │")
    print("└─────────────────────────────────────────────────────────┘")
    
    # Writing Style (Simple indicators)
    print("\n┌─────────────────────────────────────────────────────────┐")
    print("│                 ✍️  WRITING STYLE                        │")
    print("├─────────────────────────────────────────────────────────┤")
    
    style_score = 0
    style_notes = []
    
    if features['personal_pronoun_count'] > 0:
        style_notes.append("│  ✅  Uses personal language (I, we, you)               │")
        style_score += 25
    else:
        style_notes.append("│  ⚠️  No personal pronouns - sounds formal/robotic     │")
    
    if features['contraction_count'] > 0:
        style_notes.append("│  ✅  Uses contractions (natural speech)                │")
        style_score += 25
    else:
        style_notes.append("│  ⚠️  No contractions - sounds very formal             │")
    
    if features['formal_word_count'] > 2:
        style_notes.append("│  ⚠️  Many formal/academic words detected              │")
    else:
        style_notes.append("│  ✅  Natural vocabulary, not overly academic           │")
        style_score += 25
    
    if features['exclamation_count'] > 0:
        style_notes.append("│  ✅  Shows emotion/enthusiasm                          │")
        style_score += 25
    
    for note in style_notes:
        print(note)
    
    print("└─────────────────────────────────────────────────────────┘")
    
    # Recommendations
    print("\n┌─────────────────────────────────────────────────────────┐")
    print("│                 💡 RECOMMENDATIONS                      │")
    print("├─────────────────────────────────────────────────────────┤")
    
    recs = []
    if prediction == 'ai' and ai_prob > 60:
        recs.append("│  1️⃣  Add more personal experiences and opinions       │")
        recs.append("│  2️⃣  Use contractions (don't, can't, won't)           │")
        recs.append("│  3️⃣  Vary your sentence structure                     │")
    
    if originality < 70:
        recs.append("│  📝  Rephrase common phrases in your own words        │")
    
    if cite['total_citations'] == 0:
        recs.append("│  📚  Add citations to support your claims             │")
    
    if features['personal_pronoun_count'] == 0:
        recs.append("│  👤  Include personal pronouns for authenticity       │")
    
    if not recs:
        recs.append("│  ✅  Great job! Your text looks authentic!            │")
    
    for rec in recs[:4]:  # Show max 4 recommendations
        print(rec)
    
    print("└─────────────────────────────────────────────────────────┘")
    
    # Final Verdict
    print("\n╔═════════════════════════════════════════════════════════╗")
    print("║                    🏆 FINAL VERDICT                     ║")
    print("╠═════════════════════════════════════════════════════════╣")
    
    if overall_score >= 80:
        print("║                                                         ║")
        print("║      ⭐⭐⭐⭐⭐  EXCELLENT - Highly Authentic!          ║")
        print("║                                                         ║")
        print("║   Your text appears genuine and original.               ║")
        print("║   Ready for submission!                                 ║")
    elif overall_score >= 60:
        print("║                                                         ║")
        print("║      ⭐⭐⭐⭐☆  GOOD - Mostly Authentic                 ║")
        print("║                                                         ║")
        print("║   Your text is acceptable with minor concerns.          ║")
        print("║   Consider the recommendations above.                   ║")
    elif overall_score >= 40:
        print("║                                                         ║")
        print("║      ⭐⭐⭐☆☆  FAIR - Needs Improvement                 ║")
        print("║                                                         ║")
        print("║   Some sections may need rewriting.                     ║")
        print("║   Follow the recommendations to improve.                ║")
    else:
        print("║                                                         ║")
        print("║      ⭐⭐☆☆☆  POOR - Significant Concerns              ║")
        print("║                                                         ║")
        print("║   This text shows strong AI patterns.                   ║")
        print("║   Consider rewriting in your own words.                 ║")
    
    print("║                                                         ║")
    print("╚═════════════════════════════════════════════════════════╝")
    print("\n")

# Test with AI-like text
print("\n" + "🔬 TEST 1: AI-LIKE TEXT ".center(60, "─"))
ai_text = """The implementation of machine learning algorithms requires careful consideration of various hyperparameters. 
Furthermore, it is important to note that the systematic application of these methodologies yields significant improvements 
in overall performance metrics. The analysis reveals that the proposed methodology outperforms existing baseline approaches."""
analyze_text(ai_text)

# Test with human-like text
print("\n" + "🔬 TEST 2: HUMAN-LIKE TEXT ".center(60, "─"))
human_text = """I've been working on this project for months, and I'm really excited about the results! 
My team and I struggled at first, but we learned so much along the way. 
The journey wasn't easy - there were many late nights and countless cups of coffee."""
analyze_text(human_text)

# Test with academic text with citations
print("\n" + "🔬 TEST 3: ACADEMIC TEXT WITH CITATIONS ".center(60, "─"))
academic_text = """According to Smith (2023), machine learning has revolutionized data analysis. 
The findings suggest that neural networks outperform traditional methods [1]. 
Previous studies have shown significant improvements in accuracy (Johnson, 2022)."""
analyze_text(academic_text)
