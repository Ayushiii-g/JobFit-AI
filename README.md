# JobFit AI - JobFit AI – AI-Powered Job-Candidate Fit Assessment!
This project uses Natural Language Processing (NLP) and Machine Learning to evaluate the semantic similarity between resumes and job descriptions. It helps automate the candidate screening process by determining how well a resume fits a job role.

Built using the `sentence-transformers` library and deployed via a simple Gradio interface.
## 📂 Dataset

- **Source:** `UpdatedResumeDataSet.csv`  
- **Format:** Contains resumes labeled by job category

---

## ⚙️ How It Works

### Sentence Embedding  
Uses the `all-MiniLM-L6-v2` model to generate semantic embeddings for both resumes and job descriptions.

### Similarity Score  
Cosine similarity is computed between resume and job description embeddings to generate a match score.

### Classification  
A logistic regression model classifies the pair as:  
- **Fit (1)** if score > 0.4  
- **Not Fit (0)** otherwise

### Gradio Interface  
A user-friendly interface allows users to input resume and job description text and instantly get the match score and classification result.

---


