import numpy as np
from sklearn.neural_network import BernoulliRBM
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neural_network import MLPClassifier
import joblib


class RBMStackedAutoencoder(BaseEstimator, TransformerMixin):
    def __init__(self, layers, batch_size=10, learning_rate=0.01, n_iter=10):
        self.layers = layers
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.rbms = []

    def fit(self, X, y=None):
        input_data = X
        for i, layer in enumerate(self.layers):
            print(
                f"Training RBM layer {i + 1}/{len(self.layers)} with {layer} hidden units."
            )
            rbm = BernoulliRBM(
                n_components=layer,
                batch_size=self.batch_size,
                learning_rate=self.learning_rate,
                n_iter=self.n_iter,
                verbose=True,
            )
            rbm.fit(input_data)
            input_data = rbm.transform(input_data)
            self.rbms.append(rbm)
        return self

    def transform(self, X):
        for rbm in self.rbms:
            X = rbm.transform(X)
        return X

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)


class FineTunedAutoencoder(BaseEstimator, TransformerMixin):
    def __init__(
        self, pre_trained_rbm, hidden_layers=(100,), learning_rate=0.01, max_iter=200
    ):
        self.pre_trained_rbm = pre_trained_rbm
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.mlp = None

    def fit(self, X, y):
        # Transform data using pre-trained RBM
        X_transformed = self.pre_trained_rbm.transform(X)

        # Initialize MLP with weights from RBM
        self.mlp = MLPClassifier(
            hidden_layer_sizes=self.hidden_layers,
            learning_rate_init=self.learning_rate,
            max_iter=self.max_iter,
            verbose=True,
        )
        self.mlp.fit(X_transformed, y)
        return self

    def predict(self, X):
        X_transformed = self.pre_trained_rbm.transform(X)
        return self.mlp.predict(X_transformed)

    def score(self, X, y):
        X_transformed = self.pre_trained_rbm.transform(X)
        return self.mlp.score(X_transformed, y)


if __name__ == "__main__":
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split

    # Load MNIST data
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True)
    X = X / 255.0
    y = y.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Pre-train with RBM
    rbm_layers = [256, 128, 64]
    pre_trained_rbm = RBMStackedAutoencoder(
        layers=rbm_layers, batch_size=10, learning_rate=0.01, n_iter=10
    )
    pre_trained_rbm.fit(X_train)

    # Fine-tune with backpropagation
    fine_tuned_autoencoder = FineTunedAutoencoder(
        pre_trained_rbm=pre_trained_rbm,
        hidden_layers=(64,),
        learning_rate=0.01,
        max_iter=50,
    )
    fine_tuned_autoencoder.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(fine_tuned_autoencoder.mlp, "./model/rbm_fine_tuned.pkl")

    # Evaluate
    score = fine_tuned_autoencoder.score(X_test, y_test)
    print(f"Test accuracy: {score}")
