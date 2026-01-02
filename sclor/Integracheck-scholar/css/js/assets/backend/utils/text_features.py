"""
Text Feature Extraction Module
Extracts various linguistic features for AI/Human detection and text quality analysis
"""
import re
from collections import Counter

def extract_text_features(text):
    """Extract comprehensive text features for analysis"""
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Basic statistics
    word_count = len(words)
    sentence_count = len(sentences)
    avg_word_length = sum(len(w) for w in words) / max(word_count, 1)
    avg_sentence_length = word_count / max(sentence_count, 1)
    
    # Vocabulary richness
    unique_words = set(w.lower() for w in words)
    vocabulary_richness = len(unique_words) / max(word_count, 1)
    
    # Punctuation analysis
    punctuation_count = len(re.findall(r'[,.!?;:\-\(\)]', text))
    punctuation_ratio = punctuation_count / max(word_count, 1)
    
    # Formal language indicators (common in AI text)
    formal_words = ['furthermore', 'moreover', 'consequently', 'therefore', 'thus',
                    'hence', 'accordingly', 'subsequently', 'nevertheless', 'nonetheless',
                    'aforementioned', 'methodology', 'implementation', 'utilization']
    formal_count = sum(1 for w in words if w.lower() in formal_words)
    
    # Personal pronouns (common in human text)
    personal_pronouns = ['i', 'me', 'my', 'we', 'us', 'our', 'you', 'your']
    pronoun_count = sum(1 for w in words if w.lower() in personal_pronouns)
    
    # Contractions (common in human text)
    contractions = re.findall(r"\b\w+'\w+\b", text)
    contraction_count = len(contractions)
    
    # Exclamation marks (more common in human text)
    exclamation_count = text.count('!')
    
    return {
        'word_count': word_count,
        'sentence_count': sentence_count,
        'avg_word_length': round(avg_word_length, 2),
        'avg_sentence_length': round(avg_sentence_length, 2),
        'vocabulary_richness': round(vocabulary_richness, 3),
        'punctuation_ratio': round(punctuation_ratio, 3),
        'formal_word_count': formal_count,
        'personal_pronoun_count': pronoun_count,
        'contraction_count': contraction_count,
        'exclamation_count': exclamation_count
    }

def calculate_readability_score(text):
    """Calculate Flesch Reading Ease Score"""
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s for s in sentences if s.strip()]
    
    word_count = len(words)
    sentence_count = max(len(sentences), 1)
    
    # Count syllables (simplified)
    def count_syllables(word):
        word = word.lower()
        vowels = 'aeiou'
        count = 0
        prev_vowel = False
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel
        return max(count, 1)
    
    syllable_count = sum(count_syllables(w) for w in words)
    
    # Flesch Reading Ease formula
    if word_count == 0:
        return 0
    
    score = 206.835 - 1.015 * (word_count / sentence_count) - 84.6 * (syllable_count / word_count)
    return max(0, min(100, round(score, 1)))

def get_readability_level(score):
    """Convert readability score to human-readable level"""
    if score >= 90:
        return "Very Easy (5th grade)"
    elif score >= 80:
        return "Easy (6th grade)"
    elif score >= 70:
        return "Fairly Easy (7th grade)"
    elif score >= 60:
        return "Standard (8th-9th grade)"
    elif score >= 50:
        return "Fairly Difficult (10th-12th grade)"
    elif score >= 30:
        return "Difficult (College)"
    else:
        return "Very Difficult (Professional)"

def detect_repetitive_patterns(text):
    """Detect repetitive phrases or patterns (potential AI indicator)"""
    words = text.lower().split()
    
    # Check for repeated phrases (3-grams)
    trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
    trigram_counts = Counter(trigrams)
    repeated_phrases = {k: v for k, v in trigram_counts.items() if v > 1}
    
    return {
        'repeated_phrase_count': len(repeated_phrases),
        'repeated_phrases': list(repeated_phrases.keys())[:5]  # Top 5
    }
