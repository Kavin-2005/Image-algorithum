import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, TruncatedSVD, KernelPCA
from sklearn.manifold import TSNE
from PIL import Image
from google.colab import files

# Step 1: Upload image
uploaded = files.upload()
for file_name in uploaded:
    image_path = file_name

# Step 2: Load image in grayscale
img = Image.open(image_path).convert('L')   # Convert to grayscale
img = img.resize((128, 128))                # Resize to manageable size
X = np.array(img) / 255.0                   # Normalize

# Flatten the image for SVD/PCA
X_flat = X.reshape(-1, 1)  # Each pixel is a feature

# Function to reshape and display image
def show_image(title, data, shape=(128, 128)):
    plt.imshow(data.reshape(shape), cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# 1. PCA
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X)
X_pca_inv = pca.inverse_transform(X_pca)
show_image("PCA Reconstructed", X_pca_inv)

# 2. Kernel PCA
kpca = KernelPCA(n_components=50, kernel='rbf', gamma=0.02, fit_inverse_transform=True)
X_kpca = kpca.fit_transform(X)
X_kpca_inv = kpca.inverse_transform(X_kpca)
show_image("Kernel PCA Reconstructed", X_kpca_inv)

# 3. Truncated SVD
X_flat_for_svd = X.reshape(-1, 128)  # Reshape into (128, 128)
svd = TruncatedSVD(n_components=50)
X_svd = svd.fit_transform(X_flat_for_svd)
X_svd_inv = svd.inverse_transform(X_svd)
show_image("SVD Reconstructed", X_svd_inv)

# 4. t-SNE (only visualization, no inverse transform)
X_tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42).fit_transform(X)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=5, c='gray')
plt.title("t-SNE Visualization of Pixels")
plt.show()
