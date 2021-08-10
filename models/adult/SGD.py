import logging
import joblib
import numpy as np
import time

import os
import sys
import tensorflow as tf
import math

from baseline_constants import ACCURACY_KEY

from model import Model

from sklearn.linear_model import SGDClassifier
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score, log_loss

logger = logging.getLogger(__name__)


class ClientModel(Model):

    def __init__(self, seed, lr, num_classes, input_dim, cfg=None):

        self.num_classes = num_classes
        self.input_dim = input_dim
        self.cfg = cfg
        self.lr = lr

        self.model_name = os.path.abspath(__file__)

        self.model_type = 'Sklearn-linear-SGD_classif'
        self.global_model = SGDClassifier(loss='log', penalty='l2', warm_start = True)

        if self.cfg.sensitive_attribute == 'gender':
            self.sensitive_attribute = 0
        else: #cfg.sensitive_attribute == 'race'
            self.sensitive_attribute = 2 #1

        super(ClientModel, self).__init__(seed, lr)
    
    def create_model(self):

        return self.global_model

    def process_x(self, raw_x_batch):
        return np.array(raw_x_batch)

    def process_y(self, raw_y_batch):
        return np.array(raw_y_batch)

    def train(self, data, num_epochs=1, batch_size=10):
        # Extract x_train and y_train, by default,
        # label is stored in the last column
        x = self.process_x(data['x'])
        y = self.process_y(data['y'])

        if self.is_fitted(self.global_model):
            params_old = self.global_model.coef_[0]
            loss_old = log_loss(y, self.global_model.predict(x))
        else: 
            params_old = np.zeros(self.input_dim)
            loss_old = 0

        # Compute the classes based on labels if classification problem
        if self.classes is None:
            logger.warning(
                "Obtaining class labels based on local dataset. "
                "This may cause failures during aggregation "
                "when parties have distinctive class labels. ")
            self.classes = self.get_classes(labels=y)

        sample_weight = np.ones(x.shape[0])
        # sklearn `partial_fit` uses max_iter = 1,
        # manually start the local training cycles
        for iter in range(num_epochs):
            self.global_model.partial_fit(x, y,
                                    classes=self.classes,
                                    sample_weight=sample_weight)
        
        update = self.global_model.coef_[0]
        acc, loss, disp_imp = self.test(data)

        grad = []
        for i in update:
            grad.append((params_old[i] - update[i]) / self.lr) #TO DO

        return update, acc, loss, disp_imp, grad, loss_old

    def test(self, data):
        x = self.process_x(data['x'])
        y = self.process_y(data['y'])

        y_pred = self.global_model.predict(x)

        acc = accuracy_score(y, y_pred)
        loss = log_loss(y, y_pred)

        privileged = np.ones(len(x))
        unprivileged = np.zeros(len(x))

        unpriv_samples = np.equal(x[:,self.sensitive_attribute],unprivileged)
        priv_samples = np.equal(x[:,self.sensitive_attribute],privileged)

        unpriv_pred = y_pred[unpriv_samples]
        term1 = tf.count_nonzero(unpriv_pred) / len(unpriv_pred)


        priv_pred = y_pred[priv_samples]
        term2 = tf.count_nonzero(priv_pred) / len(priv_pred)

        disp_imp = np.divide(term1,term2)

        return acc, loss, disp_imp

    def set_params(self, model_params):
        self.global_model.set_params(model_params)

    def get_params(self):
        return self.global_model.get_params()


    def predict_proba(self, x):
        """
        Perform prediction for the given input.

        :param x: Samples with shape as expected by the model.
        :type x: `np.ndarray`
        :return: Array of predictions
        :rtype: `np.ndarray`
        """
        return self.model.predict_proba(x)

    def get_classes(self, labels=None):
        """
        Returns an array of shape (n_classes,). If self.classes is not None,
        return self.classes, else obtains the array based on provided labels.

        :param labels: Provided class labels to obtain the array of classes.
        :type labels: `numpy.ndarray`
        :return: An array of shape `(n_classes,)`.
        :rtype: `numpy.ndarray`
        """
        if self.classes is not None:
            return self.classes
        elif hasattr(self.model, "classes_"):
            return self.model.classes_
        elif labels is not None:
            return np.unique(labels)
        else:
            raise NotFittedError(
                "The scikit-learn model has not been initialized with "
                "`classes_` attribute, "
                "please either manually specify `classes_` attribute as "
                "an array of shape (n_classes,) or "
                "provide labels to obtain the array of classes. ")


    def save_model(self, filename=None, path=None):
        """
        Save a sklearn model to file in the format specific
        to the framework requirement.

        :param filename: Name of the file where to store the model.
        :type filename: `str`
        :param path: Path of the folder where to store the model. If no path \
        is specified, the model will be stored in the default data location of \
        the library `DATA_PATH`.
        :type path: `str`
        :return: filename
        """
        if filename is None:
            file = self.model_name if self.model_name else self.model_type
            filename = '{}_{}.pickle'.format(file, time.time())            

        full_path = super().get_model_absolute_path(filename)

        # check if the classes_ attribute exists for SGDClassifier
        # this attribute is required for prediction and scoring
        if isinstance(self.model, SGDClassifier) and \
                not hasattr(self.model, "classes_"):
            logger.warning(
                "The classification model to be saved has no `classes_` "
                "attribute and cannot be used for prediction!")

        with open(full_path, 'wb') as f:
            joblib.dump(self.global_model, f)
        logger.info('Model saved in path: %s.', full_path)
        return filename

    @staticmethod
    def load_model_from_spec(model_spec):
        """
        Loads model from provided model_spec, where model_spec is a `dict`
        that contains the following items: model_spec['model_definition']
        contains the model definition as
        type sklearn.linear_model.SGDClassifier
        or sklearn.linear_model.SGDRegressor.

        :param model_spec: Model specification contains \
        a compiled sklearn model.
        :param model_spec: `dict`
        :return: model
        :rtype: `sklearn.cluster`
        """
        model = None
        try:
            if 'model_definition' in model_spec:
                model_file = model_spec['model_definition']
                model_absolute_path = config.get_absolute_path(model_file)

                with open(model_absolute_path, 'rb') as f:
                    model = joblib.load(f)

                if not issubclass(type(model), (SGDClassifier)):
                    raise ValueError('Provided compiled model in model_spec '
                                     'should be of type sklearn.linear_model.'
                                     'Instead it is: ' + str(type(model)))
        except Exception as ex:
            raise ModelInitializationException('Model specification was '
                                               'badly formed. '+ str(ex))
        return model

    def is_fitted(self):
        """
        Return a boolean value indicating if the model is fitted or not.

        :return: res
        :rtype: `bool`
        """
        try:
            check_is_fitted(self.global_model, ("coef_", "intercept_"))
        except NotFittedError as ex:
            logger.warning(
                "Model has no attribute `coef_` and `intercept_`, "
                "and hence is not fitted yet.")
            return False
        return True

