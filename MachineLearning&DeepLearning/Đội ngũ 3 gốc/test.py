




import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

# Tải danh sách stopwords tiếng Việt
stopwords = set(stopwords.words('vietnamese'))

# Load data từ dataframe 
df = pd.read_excel('kickoff_data.xlsx')
df['original_text']= df['Vấn đề mà doanh nghiệp, tổ chức của bạn đang gặp phải là gì? Hãy chia sẻ thật cụ thể']

# Xóa các stopword và chuẩn hóa văn bản
def preprocess(text):
    words = nltk.word_tokenize(text.lower())
    filtered_words = [word for word in words if word not in stopwords]
    return ' '.join(filtered_words)

df['normalized_text'] = df['original_text'].fillna('').astype(str).apply(preprocess)

# Tạo ma trận TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['normalized_text'])

from sklearn.cluster import KMeans
import numpy as np

# Tạo mô hình KMeans với số cụm mong muốn
num_clusters = 12
km = KMeans(n_clusters=num_clusters)

# Huấn luyện mô hình trên ma trận TF-IDF
km.fit(tfidf_matrix)

# Lấy nhãn của từng văn bản
clusters = km.labels_.tolist()


# Lấy các điểm trung tâm của từng cụm
centroids = km.cluster_centers_
def get_top_keywords(matrix, n=5):
    """
    Trả về danh sách n từ khóa quan trọng nhất trong ma trận TF-IDF.
    
    Tham số:
    - matrix: ma trận TF-IDF của các văn bản trong một cụm
    - n: số lượng từ khóa cần trả về (mặc định là 5)
    """
    # Tính toán tổng điểm TF-IDF cho mỗi từ
    scores = matrix.sum(axis=0)

    # Lấy danh sách các từ theo thứ tự giảm dần của điểm số
    keywords = [(word, scores[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    keywords = sorted(keywords, key=lambda x: -x[1])

    # Trả về n từ khóa đầu tiên
    top_keywords = [keyword[0] for keyword in keywords[:n]]
    
    return top_keywords

# Đặt tên cho các cụm dựa trên nội dung
for i in range(num_clusters):
    cluster_docs = [j for j, x in enumerate(clusters) if x == i]
    cluster_tfidf = tfidf_matrix[cluster_docs]
    cluster_keywords = get_top_keywords(cluster_tfidf, n=5)
    cluster_label = "Cụm " + str(i+1) + ": " + ", ".join(cluster_keywords)
    
    cluster_mask = np.array(clusters) == i
    df.loc[cluster_mask, 'cluster'] = cluster_label
    

# In ra kết quả phân cụm
print(df.groupby('cluster').size())


from sklearn.metrics import silhouette_score

# Tạo danh sách số lượng cụm cần thử
num_clusters_list = range(2, 13)
cluster_near_1_most = []
for num_clusters in num_clusters_list:
    km = KMeans(n_clusters=num_clusters)
    km.fit(tfidf_matrix)
    cluster_labels = km.labels_
    silhouette_avg = silhouette_score(tfidf_matrix, cluster_labels)
    print(f'K={num_clusters} - silhouette score: {silhouette_avg:.4f}')
    cluster_near_1_most.append(abs(1 - silhouette_avg))
