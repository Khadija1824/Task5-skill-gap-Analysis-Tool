import streamlit as st
import pandas as pd
from utils.preprocessing import load_data
from utils.model import train_clustering_model, identify_gaps

st.set_page_config(page_title="Skill Gap Analysis Tool", page_icon="📊", layout="wide")

st.title("📊 AI-Powered Skill Gap Analysis Tool")
st.markdown("Analyze intern profiles against industry job descriptions using **TF-IDF & K-Means Clustering**.")

# Load Data
try:
    interns_df, jobs_df = load_data()
except Exception as e:
    st.error(f"Error loading data files: {e}")
    st.stop()

# Sidebar configuration
st.sidebar.header("Configuration")
n_clusters = st.sidebar.slider("Number of Job Clusters (K)", min_value=2, max_value=3, value=2)

# Train model & cluster jobs
vectorizer, kmeans, intern_vectors, job_vectors, job_clusters = train_clustering_model(interns_df, jobs_df, n_clusters)
centroids = kmeans.cluster_centers_

# Add cluster assignments to jobs dataframe
jobs_df['Assigned_Cluster'] = job_clusters

# Main Interface: Select Intern
st.subheader("🔍 Select an Intern to Analyze")
selected_intern_name = st.selectbox("Choose Intern", interns_df['name'].tolist())

# Get selected intern data
intern_row = interns_df[interns_df['name'] == selected_intern_name].iloc[0]
intern_idx = interns_df[interns_df['name'] == selected_intern_name].index[0]
intern_vec = intern_vectors[intern_idx]

# Simple heuristic to assign intern to closest job cluster centroid using cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
similarities = cosine_similarity(intern_vec, centroids)
assigned_cluster_id = int(similarities.argmax())

col1, col2 = st.columns(2)

with col1:
    st.info(f"**Selected Intern:** {intern_row['name']}")
    st.write(f"**Current Skills:** {intern_row['skills_text']}")
    st.success(f"**Matched Career Track (Cluster ID):** {assigned_cluster_id}")

with col2:
    st.markdown("### 🎯 Recommended Training & Skill Gaps")
    missing_skills = identify_gaps(intern_vec, assigned_cluster_id, centroids, vectorizer, top_n=4)
    
    if missing_skills:
        st.warning("To bridge the gap for this career track, consider learning:")
        for skill in missing_skills:
            st.markdown(f"- 📚 **{skill}** (Recommended module available)")
    else:
        st.success("🎉 Great job! No major skill gaps detected for this cluster track.")

# Display datasets preview in expandable sections
with st.expander("📁 View Raw Datasets"):
    tab1, tab2 = st.tabs(["Interns Database", "Job Descriptions"])
    with tab1:
        st.dataframe(interns_df)
    with tab2:
        st.dataframe(jobs_df[['job_id', 'role_title', 'job_description']])