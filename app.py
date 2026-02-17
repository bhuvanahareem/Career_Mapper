import os
import fitz  # PyMuPDF
import spacy
import json
import difflib
from flask import Flask, render_template, request
from spacy.matcher import PhraseMatcher

app = Flask(__name__)
nlp = spacy.load("en_core_web_sm")

# Load your custom dataset
with open('jobs_db.json', 'r') as f:
    JOBS_DB = json.load(f)

# Initialize Matcher with all skills from our DB
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
all_skills = list(set([skill for sublist in JOBS_DB.values() for skill in sublist]))
patterns = [nlp.make_doc(skill) for skill in all_skills]
matcher.add("SKILL_BANK", patterns)

def extract_skills_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    
    doc_nlp = nlp(text)
    matches = matcher(doc_nlp)
    found = [doc_nlp[start:end].text.strip().lower() for _, start, end in matches]
    return list(set(found))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['resume']
    target_input = request.form.get('target_role', '').strip()
    
    if not file or not target_input:
        return "Please upload a resume and enter a domain.", 400

    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    # 1. Fuzzy Match the Job Title
    role_names = list(JOBS_DB.keys())
    best_matches = difflib.get_close_matches(target_input.lower(), [r.lower() for r in role_names], n=1, cutoff=0.3)

    if not best_matches:
        return f"<h1>Role Not Found</h1><p>'{target_input}' is not in our database.</p><a href='/'>Back</a>"
    
    # Get official name and required skills
    target_role = next(role for role in role_names if role.lower() == best_matches[0])
    required_skills = [s.lower() for s in JOBS_DB[target_role]]

    # 2. Extract skills from resume
    resume_skills_lower = extract_skills_from_pdf(file_path)

    # 3. Filtered Logic: Only skills that belong to the target job
    matched_skills = [s.title() for s in required_skills if s in resume_skills_lower]
    missing_skills = [s.title() for s in required_skills if s not in resume_skills_lower]
    
    score = round((len(matched_skills) / len(required_skills)) * 100, 2)

    # 4. Growth/Pivot Logic
    growth_options = []
    pivot_option = None

    if score >= 85:
        # Growth: Find roles where user has >60% match
        for role, skills in JOBS_DB.items():
            if role == target_role: continue
            r_skills = [s.lower() for s in skills]
            m_count = len([s for s in r_skills if s in resume_skills_lower])
            m_pct = round((m_count / len(r_skills)) * 100, 2)
            if m_pct >= 60:
                growth_options.append({"role": role, "pct": m_pct})
    
    elif score < 30:
        # Pivot: Find any role where user has >30% match
        highest_pct = 0
        for role, skills in JOBS_DB.items():
            if role == target_role: continue
            r_skills = [s.lower() for s in skills]
            m_count = len([s for s in r_skills if s in resume_skills_lower])
            m_pct = round((m_count / len(r_skills)) * 100, 2)
            if m_pct > 30 and m_pct > highest_pct:
                highest_pct = m_pct
                pivot_option = {"role": role, "pct": m_pct}

    return render_template('results.html', 
                           role=target_role, 
                           score=score, 
                           found=matched_skills, 
                           gaps=missing_skills,
                           growth=growth_options,
                           pivot=pivot_option,
                           num_found=len(matched_skills),
                           num_missing=len(missing_skills))

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)