"""
Citation and Plagiarism Check Module
Checks for proper citations and potential plagiarism indicators
"""
import re
from difflib import SequenceMatcher

# Sample academic phrases database (in production, this would be a larger database)
COMMON_ACADEMIC_PHRASES = [
    "it is important to note that",
    "in conclusion",
    "the results indicate that",
    "this study demonstrates",
    "according to research",
    "previous studies have shown",
    "the findings suggest that",
    "it can be concluded that",
    "the data reveals that",
    "based on the analysis"
]

def check_citations(text):
    """Check for proper citation formats"""
    # Common citation patterns
    apa_pattern = r'\([A-Z][a-z]+,?\s*\d{4}\)'  # (Author, 2024)
    mla_pattern = r'\([A-Z][a-z]+\s+\d+\)'  # (Author 45)
    ieee_pattern = r'\[\d+\]'  # [1]
    
    apa_citations = re.findall(apa_pattern, text)
    mla_citations = re.findall(mla_pattern, text)
    ieee_citations = re.findall(ieee_pattern, text)
    
    total_citations = len(apa_citations) + len(mla_citations) + len(ieee_citations)
    
    # Check for quotation marks without citations
    quotes = re.findall(r'"[^"]{20,}"', text)
    quotes_without_citation = []
    for quote in quotes:
        # Check if citation follows within 50 characters
        idx = text.find(quote)
        following_text = text[idx:idx+len(quote)+50]
        if not re.search(r'\([A-Z][a-z]+.*?\d{4}\)|\[\d+\]', following_text):
            quotes_without_citation.append(quote[:50] + '...')
    
    return {
        'total_citations': total_citations,
        'apa_citations': len(apa_citations),
        'mla_citations': len(mla_citations),
        'ieee_citations': len(ieee_citations),
        'quotes_without_citation': quotes_without_citation,
        'citation_issues': len(quotes_without_citation) > 0,
        'has_citations': total_citations > 0
    }

def check_plagiarism_indicators(text, reference_texts=None):
    """
    Check for plagiarism indicators
    In production, this would check against a database of documents
    """
    text_lower = text.lower()
    
    # Check for common academic phrases (overuse might indicate copying)
    phrase_matches = []
    for phrase in COMMON_ACADEMIC_PHRASES:
        if phrase in text_lower:
            phrase_matches.append(phrase)
    
    # Calculate similarity with reference texts if provided
    similarity_results = []
    if reference_texts:
        for ref_text in reference_texts:
            similarity = SequenceMatcher(None, text_lower, ref_text.lower()).ratio()
            if similarity > 0.3:  # 30% threshold
                similarity_results.append({
                    'similarity': round(similarity * 100, 1),
                    'preview': ref_text[:100] + '...'
                })
    
    # Estimate originality (simplified)
    word_count = len(text.split())
    common_phrase_ratio = len(phrase_matches) / max(word_count / 20, 1)  # Normalize
    
    originality_score = max(0, min(100, 100 - (common_phrase_ratio * 30) - (len(similarity_results) * 20)))
    
    return {
        'originality_score': round(originality_score, 1),
        'common_phrases_found': phrase_matches,
        'similarity_matches': similarity_results,
        'plagiarism_risk': 'High' if originality_score < 50 else 'Medium' if originality_score < 75 else 'Low'
    }

def check_self_citation(text):
    """Check for excessive self-citation patterns"""
    # Pattern for author self-citation (simplified)
    self_cite_patterns = [
        r'our previous (work|study|research)',
        r'we have (previously|earlier) (shown|demonstrated)',
        r'in our (earlier|previous) (paper|work)',
        r'as we (noted|mentioned|discussed) (earlier|before|previously)'
    ]
    
    self_citations = []
    for pattern in self_cite_patterns:
        matches = re.findall(pattern, text.lower())
        self_citations.extend(matches)
    
    return {
        'self_citation_count': len(self_citations),
        'excessive_self_citation': len(self_citations) > 3
    }

def generate_citation_report(text):
    """Generate comprehensive citation report"""
    citation_check = check_citations(text)
    plagiarism_check = check_plagiarism_indicators(text)
    self_citation = check_self_citation(text)
    
    # Overall citation quality score
    quality_score = 100
    if not citation_check['has_citations']:
        quality_score -= 30
    if citation_check['citation_issues']:
        quality_score -= 20
    if plagiarism_check['originality_score'] < 70:
        quality_score -= 25
    if self_citation['excessive_self_citation']:
        quality_score -= 10
    
    return {
        'citation_details': citation_check,
        'plagiarism_analysis': plagiarism_check,
        'self_citation': self_citation,
        'overall_quality_score': max(0, quality_score),
        'recommendations': generate_recommendations(citation_check, plagiarism_check, self_citation)
    }

def generate_recommendations(citation, plagiarism, self_cite):
    """Generate actionable recommendations"""
    recommendations = []
    
    if not citation['has_citations']:
        recommendations.append("Add citations to support your claims and give credit to sources.")
    
    if citation['citation_issues']:
        recommendations.append("Some quotes appear to lack proper citations. Review and add appropriate references.")
    
    if plagiarism['originality_score'] < 70:
        recommendations.append("Consider rephrasing some sections to improve originality.")
    
    if self_cite['excessive_self_citation']:
        recommendations.append("Reduce self-citations and include more diverse sources.")
    
    if len(plagiarism['common_phrases_found']) > 3:
        recommendations.append("Try to use more original phrasing instead of common academic expressions.")
    
    if not recommendations:
        recommendations.append("Your text appears to have good citation practices!")
    
    return recommendations
