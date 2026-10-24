import pandas as pd
import re

def load_data():
    """Load intern and job description datasets."""
    interns_df = pd.read_csv("data/interns.csv")
    jobs_df = pd.read_csv("data/job_descriptions.csv")
    return interns_df, jobs_df

def clean_text(text):
    """Basic text cleaning: lowercase, remove special characters."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text