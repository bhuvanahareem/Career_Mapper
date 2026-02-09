import os
import fitz  # PyMuPDF: The library that reads PDFs
import spacy
from flask import Flask, render_template, request
from spacy.matcher import PhraseMatcher
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
nlp = spacy.load("en_core_web_sm") # Loading the pre-trained English model

# Configuration
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- THE DATA SCIENCE & NLP LOGIC ---

def get_skills_from_pdf(pdf_path):
    """NLP: Extracting specific words from a messy PDF string."""
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text() # type: ignore
    
    # This is our 'Skill Library'. In a real app, this would have 1000s of words.
    skill_bank = ["Python", "SQL", "Tableau", "Flask", "HTML", "CSS", "Machine Learning", "Statistics", "Data Visualization", "Javascript"]
    
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(text) for text in skill_bank]
    matcher.add("Skills", patterns)
    
    doc = nlp(text)
    matches = matcher(doc)
    # Return unique skills found, formatted nicely
    return list(set([doc[start:end].text.title() for _, start, end in matches]))

def calculate_match(user_skills, target_role):
    """DATA SCIENCE: Using Math to compare two sets of data."""
    job_db = {
        "Data Analytics": ["Python", "SQL", "Tableau", "Statistics", "Data Visualization"],
        "Web Dev": ["HTML", "CSS", "Flask", "SQL", "Javascript"]
    }
    
    target_skills = job_db.get(target_role, [])
    if not user_skills: return 0, target_skills
    
    # Cosine Similarity: Turning words into vectors (numbers) to find the 'angle' between them
    # A score of 1.0 is a perfect match, 0.0 is no match.
    vectorizer = CountVectorizer()
    count_matrix = vectorizer.fit_transform([" ".join(user_skills), " ".join(target_skills)])
    score = cosine_similarity(count_matrix[0:1], count_matrix[1:2])[0][0] # type: ignore
    
    # Gap Analysis: Simple set subtraction
    missing = [s for s in target_skills if s not in user_skills]
    return round(score * 100, 2), missing

# --- THE ROUTES (Web Part) ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['resume']
    target = request.form.get('target_role')
    
    # Save the file temporarily
    file_path = os.path.join(UPLOAD_FOLDER, file.filename) # type: ignore
    file.save(file_path)
    
    # Execute the NLP and Data Science functions
    user_found = get_skills_from_pdf(file_path)
    match_percentage, missing_skills = calculate_match(user_found, target)
    
    return render_template('result.html', 
                           score=match_percentage, 
                           gaps=missing_skills, 
                           role=target, 
                           found=user_found)

if __name__ == '__main__':
    app.run(debug=True)