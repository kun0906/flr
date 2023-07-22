import warnings
import flwr as fl
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

import utils

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--n_clients", type=int, default=10)
parser.add_argument("--i_client", type=int, default=0)
parser.add_argument("--seed", type=int, default=42)  #
args = parser.parse_args()
print(args)
N_CLIENTS = args.n_clients
RANDOM_STATE=args.seed
I_CLIENT = args.i_client

if __name__ == "__main__":
    # Load MNIST dataset from https://www.openml.org/d/554
    (X_train, y_train), (X_test, y_test) = utils.load_mnist()
    # Load Credit_risk dataset from:
    # (X_train, y_train), (X_test, y_test) = utils.load_credit_risk()

    # Split train set into 10 partitions and randomly use one for training.
    rng = np.random.RandomState(seed=RANDOM_STATE)
    # partition_id = rng.choice(n_clients)
    (X_train, y_train) = utils.partition(X_train, y_train, N_CLIENTS)[I_CLIENT]

    # Create LogisticRegression Model
    model = LogisticRegression(
        penalty="l2",
        max_iter=1,  # local epoch
        warm_start=True,  # prevent refreshing weights when fitting
        random_state=RANDOM_STATE,
    )

    # Setting initial parameters, akin to model.compile for keras models
    utils.set_initial_params(model)

    # Define Flower client
    class MnistClient(fl.client.NumPyClient):
        def get_parameters(self, config):  # type: ignore
            return utils.get_model_parameters(model)

        def fit(self, parameters, config):  # type: ignore
            utils.set_model_params(model, parameters)
            # Ignore convergence failure due to low local epochs
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
            print(f"Training finished for round {config['server_round']} at client: {I_CLIENT}")
            return utils.get_model_parameters(model), len(X_train), {}

        def evaluate(self, parameters, config):  # type: ignore
            utils.set_model_params(model, parameters)
            loss = log_loss(y_test, model.predict_proba(X_test))
            accuracy = model.score(X_test, y_test)
            return loss, len(X_test), {"accuracy": accuracy}

    # Start Flower client
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=MnistClient())
