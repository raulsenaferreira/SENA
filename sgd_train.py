import numpy as np
import utils
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.metrics import accuracy_score



filename = 'models/sgd_mnist.sav'
scaler = StandardScaler()

(x_train, y_train), (x_test, y_test) = utils.load_data('mnist', num_samples=None)
'''
print('training...')
nsamples, nx, ny, ndim = x_train.shape
train_dataset = x_train.reshape((nsamples,nx*ny*ndim))

x_train_scaled = scaler.fit_transform(train_dataset.astype(np.float64))

sgd_clf = SGDClassifier(random_state=42) # instantiate
sgd_clf.fit(x_train_scaled, y_train) # train the classifier

# save the model to disk
pickle.dump(sgd_clf, open(filename, 'wb'))
'''

print('testing...')
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))

nsamples, nx, ny, ndim = x_test.shape
test_dataset = x_test.reshape((nsamples,nx*ny*ndim))

x_test_scaled = scaler.fit_transform(test_dataset.astype(np.float64))

pred = loaded_model.predict(x_test_scaled)

print(accuracy_score(pred, y_test))

print('features from the image', np.shape(loaded_model.coef_))