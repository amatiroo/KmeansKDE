# KmeansKDE
K-means with randomly chosen initial centroids and  K-means with KDE (Kernel Density Estimation) to properly select centroids.


This is a Python code for the K-Means clustering algorithm.

This code reads the iris dataset using pandas, sets the labels to numerical values,
and drops one feature (petal_width) for clustering based on sepal_length and sepal_width. Then, it selects random points as initial centroids,
plots them on the graph in red, and assigns each data point to the nearest centroid, visualized in green, blue, and yellow.
It repeats this process, computes new centroids based on the initial assignment until the centroids are stable, and returns the updated centroids.

KDE code uses the code uses Kernel Density Estimation to identify high-density regions in the data and selects three points
that are separated by a minimum distance threshold. These points will serve as the initial centroids for a K-means clustering algorithm.

