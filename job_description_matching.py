
# !pip install -q sentence-transformers
# !pip install -q pandas sklearn matplotlib

import pandas as pd
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

df = pd.read_csv("UpdatedResumeDataSet.csv")
df = df[['Resume', 'Category']]
df.head()

job_descriptions = {
    "Data Science": "Looking for a data scientist with Python, machine learning, and SQL expertise.",
    "HR": "Seeking an HR manager experienced in talent acquisition and employee engagement.",
    "Advocate": "Hiring a legal associate with knowledge of civil and criminal law.",
    "Arts": "Looking for a creative individual with skills in design and fine arts.",
    "Web Designing": "Looking for a frontend developer with experience in HTML, CSS, and JavaScript."
}

df = df[df['Category'].isin(job_descriptions.keys())]  # Filter to common categories
df['Job_Description'] = df['Category'].map(job_descriptions)
df.head()

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_similarity(row):
    resume_emb = model.encode(row['Resume'], convert_to_tensor=True)
    jd_emb = model.encode(row['Job_Description'], convert_to_tensor=True)
    score = util.cos_sim(resume_emb, jd_emb).item()
    return score

df['Match_Score'] = df.apply(get_similarity, axis=1)
df.head()

df['Fit_Label'] = df['Match_Score'].apply(lambda x: 1 if x > 0.4 else 0)
X = df[['Resume', 'Job_Description']]
y = df['Fit_Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Re-encode for training
train_scores = [util.cos_sim(model.encode(r), model.encode(jd)).item() for r, jd in zip(X_train['Resume'], X_train['Job_Description'])]
test_scores = [util.cos_sim(model.encode(r), model.encode(jd)).item() for r, jd in zip(X_test['Resume'], X_test['Job_Description'])]

clf = LogisticRegression()
clf.fit(np.array(train_scores).reshape(-1,1), y_train)

y_pred = clf.predict(np.array(test_scores).reshape(-1,1))
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

def predict_fit(resume_text, job_text):
    res_emb = model.encode(resume_text)
    jd_emb = model.encode(job_text)
    score = util.cos_sim(res_emb, jd_emb).item()
    label = clf.predict([[score]])
    return score, "Fit" if label[0] == 1 else "Not Fit"

# Example:
resume_input = "Python with NLP knowledge"
job_input = "Looking for a data scientist with Python, deep learning, and NLP knowledge."

score, result = predict_fit(resume_input, job_input)
print(f"Score: {round(score, 2)}, Prediction: {result}")

# pip install streamlit

# pip install gradio

# Gradio Interface
import gradio as gr
import numpy as np

def gradio_interface(resume_text, job_text):
    score, result = predict_fit(resume_text, job_text)
    return round(score, 2), result
iface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.components.Textbox(lines=5, label="Resume Text"),
        gr.components.Textbox(lines=5, label="Job Description Text"),
    ],
    outputs=[
        gr.components.Number(label="Match Score"),
        gr.components.Textbox(label="Prediction"),
    ],
    title="JobFit AI - AI Powered Candidate Fit Assessment",
    description="Enter resume text and job description to get a match score and prediction.",
)

iface.launch()