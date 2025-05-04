# Iris Flower Classification using K-Nearest Neighbors (KNN)

This repository Iris Flower Classification, focusing on implementing the K-Nearest Neighbors algorithm for a multi-class classification problem using the classic Iris dataset.

## Objective
To understand and apply the KNN algorithm for classification. This involves data loading, preprocessing (feature normalization and target encoding), training a KNN model, experimenting with different values of 'K' (number of neighbors), evaluating the model's performance, and optionally visualizing decision boundaries.

## Dataset
- **Source:** Iris Dataset
- **File:** `Iris.csv`
- **Description:** A well-known dataset in pattern recognition, it contains 150 samples from three species of Iris flowers (Iris setosa, Iris virginica, and Iris versicolor). Four features were measured from each sample: the length and width of the sepals and petals, in centimeters.
- **Features:** `SepalLengthCm`, `SepalWidthCm`, `PetalLengthCm`, `PetalWidthCm`
- **Target Variable:** `Species` (Categorical: Iris-setosa, Iris-versicolor, Iris-virginica). This was encoded numerically (e.g., 0, 1, 2) for the model.

## Files in this Repository
- `Iris_KNN_Classification.ipynb`: Jupyter Notebook containing the Python code for data analysis, preprocessing, KNN model training, K value tuning, evaluation, and visualization.
- `Iris.csv`: The dataset file used.
- `README.md`: This explanatory file.
- `.gitignore` (Optional): Specifies files intentionally untracked by Git.

## Tools and Libraries Used
- Python 3.x
- Pandas: For data loading and manipulation.
- NumPy: For numerical operations.
- Scikit-learn:
    - `train_test_split`: For splitting the dataset.
    - `StandardScaler`: For feature normalization/standardization.
    - `LabelEncoder`: For encoding the target variable.
    - `KNeighborsClassifier`: The KNN model implementation.
    - `accuracy_score`, `confusion_matrix`, `classification_report`: For evaluating model performance.
- Matplotlib & Seaborn: For data visualization (K vs. Accuracy plot, Confusion Matrix, Decision Boundaries).
- Jupyter Notebook: For interactive code development.

## Methodology / Steps Taken
1.  **Load Data:** Imported the `Iris.csv` dataset using Pandas.
2.  **Inspect & Clean:** Examined data types, checked for missing values (none found), and dropped the unnecessary `Id` column.
3.  **Encode Target:** Converted the categorical `Species` column into numerical labels using `LabelEncoder`.
4.  **Feature/Target Split:** Separated the four measurement columns as features (X) and the encoded species label as the target (y).
5.  **Train/Test Split:** Divided the data into training (70%) and testing (30%) sets, using stratification to maintain class proportions.
6.  **Normalize Features:** Applied `StandardScaler` to the feature columns. This is crucial for KNN as it relies on distance metrics. The scaler was fitted only on the training data and then used to transform both training and test sets.
7.  **Initial KNN Model:** Trained a `KNeighborsClassifier` with an initial K value (e.g., K=5) on the scaled training data and evaluated its accuracy on the scaled test data.
8.  **Experiment with K:** Trained and evaluated KNN models for a range of K values (e.g., 1 to 25). Plotted the test accuracy against K to find the optimal value(s).
9.  **Evaluate with Best K:** Retrained the model using the best K identified and performed a detailed evaluation using accuracy, confusion matrix, and a classification report.
10. **Visualize Decision Boundaries :** Trained a KNN model on two selected (scaled) features (e.g., Petal Length and Petal Width) to visualize the classification boundaries in a 2D plot.

## How to Run
1.  Clone this repository:
2.  Navigate to the cloned directory:
3.  Install required libraries:
    ```bash
    pip install pandas scikit-learn matplotlib seaborn notebook
    ```
4.  Launch Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
5.  Open and run the `Iris_KNN_Classification.ipynb` notebook cells sequentially.

## Questions and Answers

### 1. How does the KNN algorithm work?
KNN (K-Nearest Neighbors) is a non-parametric, instance-based learning algorithm used for both classification and regression.
*   **Training Phase:** It simply stores the entire training dataset (features and corresponding labels). There's no explicit model building.
*   **Prediction Phase (for a new data point):**
    1.  **Calculate Distances:** Compute the distance (e.g., Euclidean distance) between the new data point and *all* points in the stored training dataset.
    2.  **Find Neighbors:** Identify the 'K' training data points that are closest (have the smallest distances) to the new point. K is a user-defined integer.
    3.  **Vote/Average:**
        *   **Classification:** Assign the class label to the new point based on a majority vote among its K nearest neighbors. (The most common class among the K neighbors becomes the prediction).
        *   **Regression:** Predict the value for the new point by averaging the values of its K nearest neighbors.

### 2. How do you choose the right K?
Choosing the optimal value for K is crucial and often data-dependent.
*   **Small K (e.g., K=1):** The model is very sensitive to noise and outliers (high variance, low bias). Decision boundaries can be complex and irregular.
*   **Large K:** The model becomes smoother and less sensitive to noise (low variance, potentially higher bias). It considers more neighbors, potentially blurring the distinction between classes if K is too large.
*   **Methods for Choosing K:**
    *   **Rule of Thumb:** Often start with K = sqrt(N), where N is the number of samples in the training set.
    *   **Experimentation (Elbow Method):** Train and evaluate the KNN model for a range of K values (e.g., 1 to 20 or more). Plot the accuracy (or error rate) against K. Choose the K where the accuracy starts to plateau or the error rate significantly flattens (forming an "elbow").
    *   **Cross-Validation:** Use techniques like k-fold cross-validation to evaluate performance for different K values on validation sets and select the K that yields the best average performance.
    *   **Odd K for Binary Classification:** Often preferred to avoid ties in majority voting.

### 3. Why is normalization important in KNN?
Normalization (or Standardization) is **critically important** for KNN because the algorithm relies heavily on **distance calculations** between data points.
*   Features with larger ranges or magnitudes would dominate the distance calculation if not scaled. For example, a difference of 10 in a feature ranging from 0-1000 would contribute much more to the Euclidean distance than a difference of 1 in a feature ranging from 0-10.
*   Normalization ensures that all features contribute proportionally to the distance metric, preventing features with larger values from unfairly influencing which neighbors are considered "closest".
*   Common methods include Min-Max Scaling (scales to [0, 1]) and Standardization (Z-score scaling, mean=0, stddev=1). Standardization is often preferred as it's less sensitive to outliers than Min-Max scaling.

### 4. What is the time complexity of KNN?
KNN has unusual complexity characteristics:
*   **Training Time:** **O(1)** (or very fast, O(d*N) if you count storing data). It simply stores the dataset.
*   **Prediction Time (Naive):** **O(N*d)** for *each* test point, where N is the number of training samples and d is the number of features. This is because it needs to compute the distance from the new point to *every* training point. This can be very slow for large datasets.
*   **Prediction Time (Optimized):** Using data structures like K-D Trees or Ball Trees can significantly speed up the neighbor search, making prediction time closer to **O(log N * d)** or similar on average, although worst-case can still be high.
*   **Space Complexity:** **O(N*d)**, as it needs to store the entire training dataset.

### 5. What are pros and cons of KNN?
*   **Pros:**
    *   **Simple and Intuitive:** Easy to understand and implement.
    *   **No Training Phase:** Training is fast (just data storage).
    *   **Naturally Handles Multi-Class:** Easily extends to multi-class problems.
    *   **Non-Parametric:** Makes no assumptions about the underlying data distribution.
    *   **Effective for Complex Data:** Can learn complex, non-linear decision boundaries.
*   **Cons:**
    *   **Computationally Expensive Prediction:** Can be very slow during testing/prediction for large datasets.
    *   **Requires Feature Scaling:** Performance is highly sensitive to the scale of features.
    *   **Sensitive to Irrelevant Features:** Irrelevant features can distort distance calculations ("Curse of Dimensionality").
    *   **High Memory Requirement:** Needs to store the entire training dataset.
    *   **Choosing Optimal K:** Finding the best K value requires experimentation.
    *   **Sensitive to Imbalanced Data:** Majority class can dominate predictions if K is large or data is skewed (though distance weighting can help).

### 6. Is KNN sensitive to noise?
*   **Yes, particularly for small values of K.**
*   If K=1, a single noisy data point or outlier in the training set can directly misclassify a nearby test point.
*   As K increases, the influence of single noisy points is averaged out by considering more neighbors, making the prediction more robust to noise. However, too large a K can oversmooth the decision boundary.

### 7. How does KNN handle multi-class problems?
*   KNN handles multi-class classification naturally without modification.
*   When predicting the class for a new point, it finds the K nearest neighbors in the training data.
*   It then performs a **majority vote** among these K neighbors. The class that appears most frequently among the neighbors is assigned as the predicted class for the new point.
*   Tie-breaking rules might be needed if multiple classes have the same highest count (e.g., choosing randomly, choosing the class of the nearest neighbor among the tied classes, or using distance weighting).

### 8. What’s the role of distance metrics in KNN?
*   The distance metric defines how "closeness" or "similarity" between data points is measured. It's fundamental to finding the nearest neighbors.
*   **Common Metrics:**
    *   **Euclidean Distance (L2 norm):** The most common default. Represents the straight-line distance between two points in multi-dimensional space. Sensitive to feature scales. `sqrt(sum((xi - yi)^2))`
    *   **Manhattan Distance (L1 norm):** Sum of the absolute differences between coordinates. Less sensitive to outliers than Euclidean. `sum(|xi - yi|)`
    *   **Minkowski Distance:** A generalization of Euclidean and Manhattan. `(sum(|xi - yi|^p))^(1/p)`. Euclidean is p=2, Manhattan is p=1.
    *   **Cosine Similarity:** Measures the cosine of the angle between two vectors. Often used for text data or high-dimensional sparse data. Focuses on orientation rather than magnitude.
*   The choice of metric can significantly impact KNN performance and should ideally be chosen based on the nature of the data and the problem domain. Euclidean is a good starting point for many real-valued feature sets after scaling.
