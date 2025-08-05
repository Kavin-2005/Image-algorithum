import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from PIL import Image
from google.colab import files
from io import BytesIO

# Step 1: Upload image
uploaded = files.upload()
for file_name in uploaded.keys():
    image_path = file_name

# Step 2: Read and preprocess image
img = Image.open(image_path).convert('L')
img = img.resize((256, 256))
img_array = np.array(img)

# Step 3: PCA and Reconstruction
pca = PCA(n_components=50)
transformed = pca.fit_transform(img_array)
reconstructed = pca.inverse_transform(transformed)

# Step 4: Plot original and reconstructed image
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(img_array, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Reconstructed Image (50 PCs)")
plt.imshow(reconstructed, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

# Step 5: Optional - Explained Variance Plot
pca_full = PCA().fit(img_array)
plt.figure(figsize=(8, 4))
plt.plot(np.cumsum(pca_full.explained_variance_ratio_), marker='o')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance vs. Components')
plt.grid(True)
plt.show()
