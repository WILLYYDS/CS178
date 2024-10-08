import numpy as np

class NearestCentroidClassifier:
    def __init__(self):
        # A list containing the centroids; to be filled in with the fit method.
        self.centroids = []
                
    def fit(self, X, y):
        """ Fits the nearest centroid classifier with training features X and training labels y.
        
        X: array of training features; shape (m,n), where m is the number of datapoints,
            and n is the number of features.
        y: array training labels; shape (m, ), where m is the number of datapoints.
        
        """
        # First, identify what possible classes exist in the training data set:
        self.classes_ = np.unique(y)
        
        # For each class, compute its centroid (mean vector of all points in that class)
        for c in self.classes_:
            # Extract data points corresponding to class c
            class_data = X[y == c]
            # Compute the mean of the class data (this is the centroid)
            centroid = np.mean(class_data, axis=0)
            # Append the centroid to the list of centroids
            self.centroids.append(centroid)

    def predict(self, X):
        """ Makes predictions with the nearest centroid classifier on the features in X.
        
        X: array of features; shape (m,n), where m is the number of datapoints,
            and n is the number of features.
        
        Returns:
        y_pred: a numpy array of predicted labels; shape (m, ), where m is the number of datapoints.
        """
        # List to store predicted labels
        y_pred = []
        
        # Iterate over each test point in X
        for x in X:
            # Calculate the Euclidean distance from x to each centroid
            distances = [np.linalg.norm(x - centroid) for centroid in self.centroids]
            # Find the index of the nearest centroid
            nearest_centroid_index = np.argmin(distances)
            # Append the corresponding class (self.classes_[nearest_centroid_index]) to y_pred
            y_pred.append(self.classes_[nearest_centroid_index])
        
        # Convert the predictions to a numpy array and return
        return np.array(y_pred)

# 测试代码
if __name__ == "__main__":
    # Example data
    X_train = np.array([[1, 2], [1, 3], [4, 5], [6, 7], [8, 9], [9, 10]])
    y_train = np.array([0, 0, 1, 1, 2, 2])

    # 创建分类器并训练
    classifier = NearestCentroidClassifier()
    classifier.fit(X_train, y_train)

    # 打印质心
    print("Centroids:", classifier.centroids)

    # 测试数据
    X_test = np.array([[1, 2], [5, 5], [9, 9]])
    y_test = np.array([0, 1, 2])

    # 预测
    y_pred = classifier.predict(X_test)

    # 打印预测结果和真实标签
    print("Predicted labels:", y_pred)
    print("True labels:", y_test)
