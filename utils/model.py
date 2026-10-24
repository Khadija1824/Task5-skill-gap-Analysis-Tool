import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def train_clustering_model(interns_df, jobs_df, n_clusters=2):
    """Vectorize text and cluster job descriptions."""
    # Clean text columns
    all_texts = interns_df['skills_text'].tolist() + jobs_df['job_description'].tolist()
    
    # Vectorize using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english', max_features=500, ngram_range=(1, 2))
    X = vectorizer.fit_transform(all_texts)
    
    intern_vectors = X[:len(interns_df)]
    job_vectors = X[len(interns_df):]
    
    # Fit K-Means on job descriptions
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    job_clusters = kmeans.fit_predict(job_vectors)
    
    return vectorizer, kmeans, intern_vectors, job_vectors, job_clusters

def identify_gaps(intern_vec, assigned_cluster_id, centroids, vectorizer, top_n=4):
    """Compare intern vector to cluster centroid to find missing skills."""
    centroid = centroids[assigned_cluster_id]
    intern_array = intern_vec.toarray()[0]
    
    # Calculate deficit score (centroid weight - intern weight)
    gap_scores = centroid - intern_array
    top_gap_indices = np.argsort(gap_scores)[::-1][:top_n]
    
    feature_names = np.array(vectorizer.get_feature_names_out())
    
    # Correctly loop through top_gap_indices where idx is defined
    missing_skills = []
    for idx in top_gap_indices:
        if gap_scores[idx] > 0:
            missing_skills.append(feature_names[idx].title())
            
    return missing_skills