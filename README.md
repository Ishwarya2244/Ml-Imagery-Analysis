📌 MNIST Digit Clustering using K-Means

This project demonstrates unsupervised learning using K-Means (MiniBatchKMeans) on the MNIST handwritten digits dataset.
The goal is to cluster handwritten digit images, infer their labels, evaluate performance, and visualize cluster centroids.

📂 Dataset

MNIST Dataset

60,000 training images

10,000 testing images

Image size: 28 × 28 pixels

Digits: 0 to 9

Dataset is loaded directly using keras.datasets.mnist.

🛠️ Technologies Used

Python

NumPy

Matplotlib

Scikit-learn

Keras (for MNIST dataset)

⚙️ Project Workflow

Import Dependencies

Load MNIST Dataset

Visualize Sample Images

Data Preprocessing

Flatten images (28×28 → 784)

Normalize pixel values (0–1)

Apply MiniBatch K-Means Clustering

Infer Cluster Labels

Evaluate Performance

Accuracy

Homogeneity Score

Inertia

Test on Unseen Data

Visualize Cluster Centroids

📊 Model Evaluation

The model was trained with different numbers of clusters to analyze performance.

Clusters	Accuracy
10	55.87%
16	64.69%
36	75.14%
64	81.37%
144	87.20%
256	89.34%

✅ Test Accuracy (256 clusters): 90.43%

🖼️ Cluster Centroid Visualization

Each cluster centroid is reshaped back into a 28×28 image to visualize the learned digit patterns.
Inferred digit labels are displayed above each centroid image.

📌 Key Learnings

1.Understanding Unsupervised Learning

2.Implementing K-Means Clustering

3.Label inference in clustering problems

4.Evaluating clustering performance

5.Visualizing learned patterns

▶️ How to Run
pip install numpy matplotlib scikit-learn keras tensorflow
Run the notebook or Python file step by step in Jupyter Notebook.


📁 Project Structure
MNIST-KMeans-Clustering/
├── mnist_kmeans.ipynb
├── README.md

🌟 Future Improvements

*Try PCA before clustering
*Use Gaussian Mixture Models
*Compare with Supervised Learning models
*Improve visualization and labeling accuracy

🙋‍♀️ Author

Ishwarya R 
Department of Artificial Intelligence & Data Science-3rd year
Adhiyamaan College of Engineering,Hosur
