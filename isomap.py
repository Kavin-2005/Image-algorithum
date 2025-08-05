import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, TruncatedSVD, KernelPCA
from sklearn.manifold import Isomap
from PIL import Image
from google.colab import files

# Step 1: Upload image
uploaded = files.upload()
for file_name in uploaded:
    image_path = file_name

# Step 2: Load image in grayscale
img = Image.open(image_path).convert('L')   # Convert to grayscale
img = img.resize((58, 58))                # Resize to manageable size
X = np.array(img) / 255.0                   # Normalize

# Flatten the image for SVD/PCA
X_flat = X.reshape(-1, 1)  # Each pixel is a feature

# Function to reshape and display image
def show_image(title, data, shape=(58, 58)):
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
X_flat_for_svd = X.reshape(-1, 58)  # Reshape into (128, 128)
svd = TruncatedSVD(n_components=50)
X_svd = svd.fit_transform(X_flat_for_svd)
X_svd_inv = svd.inverse_transform(X_svd)
show_image("SVD Reconstructed", X_svd_inv)

# 4. Isomap (Manifold Learning)
X_reshaped = X.reshape(-1, 1)  # Flattened image (each pixel = 1 feature)
isomap = Isomap(n_neighbors=5, n_components=2)
X_isomap = isomap.fit_transform(X_reshaped)

# Plot Isomap result (2D)
plt.scatter(X_isomap[:, 0], X_isomap[:, 1], s=2, c=X.reshape(-1), cmap='gray')
plt.title("Isomap - Manifold Learning")
plt.axis('off')
plt.show()
