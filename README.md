# 📊 AI-Powered Skill Gap Analysis Tool

An intelligent desktop web application built with **Streamlit**, **Scikit-Learn (TF-IDF & K-Means)**, and **Pandas** designed to evaluate intern profiles against industry job descriptions. The tool automatically clusters career tracks, compares individual skill sets, identifies key competencies, and recommends targeted training modules to bridge skill gaps.

---

## 🚀 Features

*   **Automated Text Vectorization:** Uses **TF-IDF (Term Frequency-Inverse Document Frequency)** with unigrams and bigrams to convert raw resume text and job descriptions into meaningful numerical vectors.
*   **Industry Clustering:** Leverages **K-Means Clustering** to group job openings into distinct career tracks (e.g., Data Science, Full-Stack Development).
*   **Smart Gap Analysis:** Computes vector differences between an intern's current skill profile and the assigned industry cluster centroid to pinpoint missing skills.
*   **Interactive Dashboard:** Built with **Streamlit** for real-time selection, adjustable cluster parameters ($K$), and a clean, responsive layout.

---
🛠️ Installation & Setup Instructions
1. Clone or Open the Project Directory
Open your terminal and navigate to your project root folder:

Bash
cd skill_gap_analyzer/
2. Install Dependencies
Make sure you have Python installed, then run:

Bash
pip install -r requirements.txt
3. Run the Application
Launch the interactive Streamlit web app by running:

Bash
streamlit run app.py
📊 Dataset Format
data/interns.csv

Code snippet
intern_id,name,skills_text
101,Ayesha Khan,"Python, SQL, Pandas, basic HTML, data cleaning"
data/job_descriptions.csv

Code snippet
job_id,role_title,job_description
1,Data Scientist / AI Intern,"Looking for an AI Intern proficient in Python, Machine Learning, Deep Learning, TensorFlow"
💡 Tech Stack
Frontend & UI: Streamlit

Machine Learning / NLP: Scikit-Learn (TfidfVectorizer, KMeans), NumPy

Data Manipulation: Pandas
