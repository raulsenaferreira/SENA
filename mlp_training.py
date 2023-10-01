import numpy as np
import utils
from sklearn.neural_network import MLPClassifier
#from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.metrics import accuracy_score

filename = 'models/sgd_fashion_mnist.sav' # 'models/sgd_mnist.sav'
#scaler = StandardScaler()

(x_train, y_train), (x_test, y_test) = utils.load_data('mnist', num_samples=None)  # utils.load_data('mnist', num_samples=None)
'''
print('training...')
nsamples, nx, ny, ndim = x_train.shape
train_dataset = x_train.reshape((nsamples,nx*ny*ndim))

#x_train_scaled = scaler.fit_transform(train_dataset.astype(np.float64))

mlp_clf = MLPClassifier(random_state=42, max_iter=300) # instantiate
mlp_clf.fit(train_dataset, y_train) # train the classifier

# save the model to disk
pickle.dump(mlp_clf, open(filename, 'wb'))

'''
print('testing...')
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))

nsamples, nx, ny, ndim = x_test.shape
test_dataset = x_test.reshape((nsamples, nx * ny * ndim))

#x_test_scaled = scaler.fit_transform(test_dataset.astype(np.float64))

pred = loaded_model.predict(test_dataset)

print(accuracy_score(pred, y_test))
print(loaded_model.predict_proba(test_dataset[0].reshape(1, -1)))
print(loaded_model.predict_proba(test_dataset[100].reshape(1, -1)))

monitored_layer = -1
#print('shape of weights from the layer model', np.shape(loaded_model.coefs_[monitored_layer]))
#print('weights from the layer model', loaded_model.coefs_[monitored_layer])
