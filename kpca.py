import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
from PIL import Image
from google.colab import files

# Step 1: Upload image
uploaded = files.upload()
for file_name in uploaded.keys():
    image_path = file_name

# Step 2: Load and preprocess image
img = Image.open(image_path).convert('L')  # Grayscale
img = img.resize((128, 128))  # Resize for efficiency
img_array = np.array(img)
X = img_array / 255.0  # Normalize to [0, 1]

# Step 3: Apply Kernel PCA
kpca = KernelPCA(n_components=50, kernel='rbf', gamma=0.02, fit_inverse_transform=True)
X_kpca = kpca.fit_transform(X)
X_inv = kpca.inverse_transform(X_kpca)

# Step 4: Plot Original and KPCA-Reconstructed Image
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(X, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Reconstructed Image (KPCA)")
plt.imshow(X_inv, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

# Step 5: Approximate "Explained Variance" Graph (using reconstruction error)
components_range = list(range(5, 130, 10))
errors = []

for n in components_range:
    kpca_temp = KernelPCA(n_components=n, kernel='rbf', gamma=0.02, fit_inverse_transform=True)
    X_kpca_temp = kpca_temp.fit_transform(X)
    X_reconstructed = kpca_temp.inverse_transform(X_kpca_temp)
    mse = np.mean((X - X_reconstructed) ** 2)
    errors.append(mse)

# Plot approximation of explained variance (lower error = better info retention)
plt.figure(figsize=(8, 4))
plt.plot(components_range, errors, marker='o')
plt.xlabel("Number of Components")
plt.ylabel("Reconstruction Error (MSE)")
plt.title("KPCA: Reconstruction Error vs Components")
plt.grid(True)
plt.tight_layout()
plt.show()
