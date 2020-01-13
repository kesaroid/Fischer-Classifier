import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import pinv, eig, inv
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix

dataset_dir = "cifar-10-batches-py"
nb_classes = 10

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def create_dataset(folder):
    x_data_temp = []; y_data_temp = []
    x_test_temp = []; y_test_temp = []

    temp_data = unpickle(folder + '/data_batch_1')
    x_data_temp.append(temp_data[b'data'])
    y_data_temp.append(temp_data[b'labels'])

    test_data_temp = unpickle(folder + '/test_batch')
    x_test_temp.append(test_data_temp[b'data'])
    y_test_temp.append(test_data_temp[b'labels'])

    x_data_temp = np.array(x_data_temp); y_data_temp = np.array(y_data_temp)
    x_test_temp = np.array(x_test_temp); y_test_temp = np.array(y_test_temp)
    
    x_data_temp = x_data_temp.reshape(x_data_temp.shape[0] * x_data_temp.shape[1], x_data_temp.shape[2])
    y_data_temp = y_data_temp.reshape(y_data_temp.shape[0] * y_data_temp.shape[1])
    x_test_temp = x_test_temp.reshape(x_test_temp.shape[0] * x_test_temp.shape[1], x_test_temp.shape[2])
    y_test_temp = y_test_temp.reshape(y_test_temp.shape[0] * y_test_temp.shape[1])

    X_train = x_data_temp.reshape(x_data_temp.shape[0], 3, 32, 32)
    X_test = x_test_temp.reshape(x_test_temp.shape[0], 3, 32, 32)
    
    return X_train, y_data_temp, X_test, y_test_temp

class DataSet:
  def __init__(self, data, targets, valid_classes=None):
    self.valid_classes = np.unique(targets)
    self.data = self.to_dict(data, targets)

  def to_dict(self, data, targets):
    data_dict = {}
    for x, y in zip(data, targets):
      if y in self.valid_classes:
        if y not in data_dict:
          data_dict[y] = [x.flatten()]
        else:
          data_dict[y].append(x.flatten())

    for i in self.valid_classes:
      data_dict[i] = np.asarray(data_dict[i])

    return data_dict

  def get_data_as_dict(self):
    return self.data

  def get_all_data(self):
    data = []
    labels = []
    for label, class_i_data in self.data.items():
      data.extend(class_i_data)
      labels.extend(class_i_data.shape[0] * [label])
    data = np.asarray(data)
    labels = np.asarray(labels)
    return data, labels

class Fischer:
  def __init__(self):
    self.projection_dim = 2
    self.W = None
    self.g_means, self.g_covariance, self.priors = None, None, None

  def fit(self,X):
    means_k = self._compute_means(X)
    
    class_cov = []
    for i, m in means_k.items():
        subtracted = np.subtract(X[i], m)
        class_cov.append(np.dot(np.transpose(subtracted), subtracted))

    class_cov = np.asarray(class_cov)
    covariance = np.sum(class_cov, axis=0)

    total_images = {}
    data_sum = 0
    for class_id, data in X.items():
        total_images[class_id] = data.shape[0]
        data_sum += np.sum(data, axis=0)

    # Total number of images and it's mean
    self.N = sum(list(total_images.values()))
    m = data_sum / self.N

    # Between class covariance
    B = []
    for class_id, mean_class_i in means_k.items():
        sub_ = mean_class_i - m
        B.append(np.multiply(total_images[class_id], np.outer(sub_, sub_.T)))
    B = np.sum(B, axis=0)

    # Find Inverse covariance
    matrix = np.dot(pinv(covariance), B)
    
    # Find Eigen values and vectors
    eigen_values, eigen_vectors = eig(matrix)
    eiglist = [(eigen_values[i], eigen_vectors[:, i]) for i in range(len(eigen_values))]

    # Sort Eigen values and sort
    eiglist = sorted(eiglist, key=lambda x: x[0], reverse=True)

    # Take the first dimensional eigvectors
    self.W = np.array([eiglist[i][1] for i in range(self.projection_dim)])
    self.W = np.asarray(self.W).T

    # Get the gaussian distribution
    self.g_means, self.g_covariance, self.priors = self.gaussian(X)

  # Returns the parameters of the Gaussian distributions
  def gaussian(self, X):
    means = {}
    covariance = {}
    priors = {}
    for class_id, values in X.items():
      proj = np.dot(values, self.W)
      means[class_id] = np.mean(proj, axis=0)
      covariance[class_id] = np.cov(proj, rowvar=False)
      priors[class_id] = values.shape[0] / self.N
    return means, covariance, priors
  
  # For a multivariate dataset, create gaussian distribution
  def gaussian_distribution(self, x, u, cov):
    scalar = (1. / ((2 * np.pi) ** (x.shape[0] / 2.))) * (1 / np.sqrt(np.linalg.det(cov)))
    x_sub_u = np.subtract(x, u)
    return scalar * np.exp(-np.dot(np.dot(x_sub_u, inv(cov)), x_sub_u.T) / 2.)

  def predict(self,X,y):
    proj = self.project(X)
    gaussian_likelihoods = []
    classes = sorted(list(self.g_means.keys()))
    for x in proj:
      row = []
      for c in classes:
        res = self.priors[c] * self.gaussian_distribution(x, self.g_means[c], self.g_covariance[c])
        row.append(res)

      gaussian_likelihoods.append(row)
    gaussian_likelihoods = np.asarray(gaussian_likelihoods)
    
    # Get the largest predicted value
    predictions = np.argmax(gaussian_likelihoods, axis=1)
    return np.sum(predictions == y) / len(y), predictions

  def project(self,X):
    return np.dot(X, self.W)

  def _compute_means(self, data):
    class_mean = {}
    for classes in sorted(data):
        class_mean[classes] = np.average(data[classes], axis=0)
    return class_mean


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = create_dataset(dataset_dir)

    X_train, X_test = X_train / 255.0, X_test / 255.0

    train_dataset = DataSet(X_train, y_train)
    data, y_true = train_dataset.get_all_data()

    classifier = Fischer()
    classifier.fit(train_dataset.get_data_as_dict())

    accuracy, y_pred = classifier.predict(data, y_true)
    confusion = confusion_matrix(y_pred, y_true)
    
    print("Confusion Matrix: \n", confusion)
    print("Train acc:", accuracy)