# Step 1: Upload images
from google.colab import files
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, TruncatedSVD, KernelPCA
from sklearn.manifold import Isomap, TSNE

print("Upload your leaf images (JPG, PNG, etc.):")
uploaded = files.upload()

image_paths = list(uploaded.keys())
print(f"Uploaded {len(image_paths)} files: {image_paths}")

# Step 2: Load and preprocess images
def load_and_preprocess(image_paths, size=(100,100)):
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: could not read {path}")
            continue
        img = cv2.resize(img, size)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        images.append(img_gray.flatten())
    return np.array(images)

X = load_and_preprocess(image_paths)
print("Preprocessed image data shape:", X.shape)

# Step 3: Define dimensionality reduction function
def reduce_dimensionality(X, method='pca'):
    if method == 'pca':
        model = PCA(n_components=2)
    elif method == 'svd':
        model = TruncatedSVD(n_components=2)
    elif method == 'kernel_pca':
        model = KernelPCA(n_components=2, kernel='rbf')
    elif method == 'isomap':
        model = Isomap(n_components=2)
    elif method == 'tsne':
        model = TSNE(n_components=2, random_state=42)
    else:
        raise ValueError(f"Unknown method {method}")
    return model.fit_transform(X)

# Step 4: Visualization
def plot_2d(X_reduced, title='2D Projection'):
    plt.figure(figsize=(8,6))
    plt.scatter(X_reduced[:,0], X_reduced[:,1], c='green')
    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid(True)
    plt.show()

# Step 5: Run all methods and plot
methods = ['pca', 'svd', 'kernel_pca', 'isomap', 'tsne']

for method in methods:
    print(f"Running {method.upper()}...")
    X_reduced = reduce_dimensionality(X, method=method)
    plot_2d(X_reduced, title=f"{method.upper()} Projection")
