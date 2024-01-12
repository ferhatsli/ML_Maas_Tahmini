import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# Veri setini yüklenmesi
df = pd.read_csv('/content/jobs_in_data.csv')  

# İlk beş satırı gösterilmesi
print(df.head())

# Eksik verileri kontrol edilmesi
print(df.isnull().sum())

# Eksik verileri doldur veya silmesi
df.fillna(0)  # ya da df.dropna()

# Temel istatistikler
print(df.describe())


# Maaş dağılımını görselleştirilmesi
df['salary'].hist()
plt.show()

"""# 2.ADIM"""

# 'salary' ve 'experience_level' sütunlarını seçilmesi
selected_columns = df[['salary', 'experience_level']]

# 'experience_level' için etiket kodlaması yapılması
label_encoder = LabelEncoder()
selected_columns['experience_level_encoded'] = label_encoder.fit_transform(selected_columns['experience_level'])

# Kodlanmış veriyi kontrol edilmesi
print(selected_columns.head())



# K-Means modelini oluşturulması (örneğin 3 kümeye ayırmak için)
kmeans = KMeans(n_clusters=3)

# Modeli eğit
selected_columns['cluster'] = kmeans.fit(selected_columns[['salary', 'experience_level_encoded']])

selected_columns['cluster'] = kmeans.labels_

plt.figure(figsize=(10, 6))
plt.scatter(selected_columns['salary'], selected_columns['experience_level_encoded'], c=selected_columns['cluster'], cmap='viridis')
plt.title('K-Means Kümeleme Sonuçları')
plt.xlabel('Maaş')
plt.ylabel('Deneyim Seviyesi (Kodlanmış)')
plt.colorbar(label='Küme')
plt.show()

selected_columns_gmm = df[['salary', 'experience_level']]

# 'experience_level' için etiket kodlaması yapılması
label_encoder = LabelEncoder()
selected_columns_gmm['experience_level_encoded'] = label_encoder.fit_transform(selected_columns_gmm['experience_level'])

# GMM modelini oluşturulması (örneğin 3 kümeye ayırmak için)
gmm = GaussianMixture(n_components=3)

# Modeli eğitilmesi
gmm.fit(selected_columns_gmm[['salary', 'experience_level_encoded']])

# Küme etiketlerini tahmin edilmesi
labels = gmm.predict(selected_columns_gmm[['salary', 'experience_level_encoded']])

# Etiketleri gösterilmesi
print("Küme Etiketleri: \n", labels)

import matplotlib.pyplot as plt

# Veri seti ve etiketleri hazırlanılması
selected_columns_gmm['cluster'] = labels  # GMM'den elde edilen küme etiketlerini ekleyin

# Görselleştirilmesi
plt.figure(figsize=(10, 6))
for i in range(gmm.n_components):
    # Her küme için ayrı bir renk kullanarak görselleştirilmesi
    cluster_data = selected_columns_gmm[selected_columns_gmm['cluster'] == i]
    plt.scatter(cluster_data['salary'], cluster_data['experience_level_encoded'], label=f'Küme {i}')

plt.title('GMM Kümeleme Sonuçları')
plt.xlabel('Maaş')
plt.ylabel('Tecrübe Seviyesi (Kodlanmış)')
plt.legend()
plt.show()



# K-Means için siluet skoru
silhouette_kmeans = silhouette_score(selected_columns[['salary', 'experience_level_encoded']], kmeans.labels_)
print(f"K-Means Siluet Skoru: {silhouette_kmeans}")

# GMM için siluet skoru
silhouette_gmm = silhouette_score(selected_columns_gmm[['salary', 'experience_level_encoded']], gmm.predict(selected_columns_gmm[['salary', 'experience_level_encoded']]))
print(f"GMM Siluet Skoru: {silhouette_gmm}")

# K-Means ve GMM etiketlerini yan yana görselleştirmek için subplots kullanılması
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# K-Means görselleştirilmesi
ax1.scatter(selected_columns['salary'], selected_columns['experience_level_encoded'], c=kmeans.labels_)
ax1.set_title('K-Means Kümeleme')
ax1.set_xlabel('Maaş')
ax1.set_ylabel('Tecrübe Seviyesi (Kodlanmış)')

# GMM görselleştirilmesi
ax2.scatter(selected_columns_gmm['salary'], selected_columns_gmm['experience_level_encoded'], c=labels)
ax2.set_title('GMM Kümeleme')
ax2.set_xlabel('Maaş')
ax2.set_ylabel('Tecrübe Seviyesi (Kodlanmış)')

plt.show()