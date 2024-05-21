import torch
from tqdm import tqdm

from pykoopman_ae.system_extraction import get_koopman_model
from pykoopman_ae.dataset_generator import get_temporal_dataset


class Trainer:
    """
    A class to train Koopman models using different neural network architectures (MLP, TCN, LSTM, GRU).

    Attributes:
        model (torch.nn.Module): The neural network model to be trained.
        trajectory (torch.Tensor): The trajectory data with shape (batch_size, num_features, sequence_length).
        input (torch.Tensor or None): The input data with shape (batch_size, sequence_length, num_inputs),
                                      or None if no input data is provided.
        loss_function (torch.nn.Module): The loss function to be used. Defaults to MSELoss if None.
        optimizer (torch.optim.Optimizer): The optimizer to be used. Defaults to AdamW if None.
        dynamic_loss_window (int): The window size for computing dynamic loss. Defaults to 10 if None.
        num_epochs (int): The number of epochs for training. Defaults to 10 if None.
        batch_size (int): The batch size for training. Defaults to 256 if None.
        weight_reconstruction_loss (float): The weight for the reconstruction loss. Defaults to 10 if None.
        weight_prediction_loss (float): The weight for the prediction loss. Defaults to 1 if None.
        weight_dynamics_loss (float): The weight for the dynamics loss. Defaults to 1 if None.
        device (torch.device): The device (CPU or GPU) on which the model parameters are located.
    """

    def __init__(
        self,
        model,
        trajectory,
        input=None,
        loss_function=None,
        optimizer=None,
        dynamic_loss_window=None,
        num_epochs=None,
        batch_size=None,
        weight_reconstruction_loss=None,
        weight_prediction_loss=None,
        weight_dynamics_loss=None,
    ):
        """
        Initializes the Trainer with the given model, data, and training parameters.

        Args:
            model (torch.nn.Module): The neural network model to be trained.
            trajectory (torch.Tensor): The trajectory data.
            input (torch.Tensor or None): The input data, or None if no input data is provided.
            loss_function (torch.nn.Module, optional): The loss function to be used. Defaults to MSELoss.
            optimizer (torch.optim.Optimizer, optional): The optimizer to be used. Defaults to AdamW.
            dynamic_loss_window (int, optional): The window size for computing dynamic loss. Defaults to 10.
            num_epochs (int, optional): The number of epochs for training. Defaults to 10.
            batch_size (int, optional): The batch size for training. Defaults to 256.
            weight_reconstruction_loss (float, optional): The weight for the reconstruction loss. Defaults to 10.
            weight_prediction_loss (float, optional): The weight for the prediction loss. Defaults to 1.
            weight_dynamics_loss (float, optional): The weight for the dynamics loss. Defaults to 1.
        """

        self.model = model
        self.trajectory = trajectory
        self.input = input
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.dynamic_loss_window = dynamic_loss_window
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.weight_reconstruction_loss = weight_reconstruction_loss
        self.weight_prediction_loss = weight_prediction_loss
        self.weight_dynamics_loss = weight_dynamics_loss
        self.device = next(self.model.parameters()).device

        if self.loss_function == None:
            self.loss_function = torch.nn.MSELoss()

        if self.optimizer == None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=0.0001, weight_decay=1e-6
            )

        if self.dynamic_loss_window == None:
            self.dynamic_loss_window = 10

        if self.num_epochs == None:
            self.num_epochs = 10

        if self.batch_size == None:
            self.batch_size = 256

        if self.weight_reconstruction_loss == None:
            self.weight_reconstruction_loss = 10

        if self.weight_prediction_loss == None:
            self.weight_prediction_loss = 1

        if self.weight_dynamics_loss == None:
            self.weight_dynamics_loss = 1

    def learn_koopman_model(self):
        """
        Trains the Koopman model with control input.

        Returns:
            float: The final training loss.
        """
        if self.model.model_type == "MLP":
            loss = train_nontemporal_model_with_control_input(self)
        else:
            loss = train_temporal_model_with_control_input(self)
        return loss

    def learn_koopman_eigendynamics(self):
        """
        Trains the Koopman model without control input.

        Returns:
            float: The final training loss.
        """
        if self.model.model_type == "MLP":
            loss = train_nontemporal_model_without_control_input(self)
        else:
            loss = train_temporal_model_without_control_input(self)
        return loss

    def learn_input_matrix(self):
        """
        Trains the input matrix of the Koopman model.

        Returns:
            float: The final training loss.
        """
        if self.model.model_type == "MLP":
            loss = train_nontemporal_b_block_with_control_input(self)
        else:
            loss = train_temporal_b_block_with_control_input(self)
        return loss


def train_nontemporal_model_with_control_input(trainer):
    """
    Trains a non-temporal Koopman model with control input using the provided trainer configuration.

    Args:
        trainer (Trainer): An instance of the Trainer class containing the model, data, and training parameters.

    Returns:
        torch.Tensor: A tensor containing the mean loss for each epoch.
    """

    loss_epoch_mean = []

    for epoch in range(trainer.num_epochs):

        for i in tqdm(range(trainer.trajectory.shape[0])):

            X_traj = trainer.trajectory[i, :, :-1].T
            Y_traj = trainer.trajectory[i, :, 1:].T
            U_traj = trainer.input[i, :, :-1].T

            losses = []
            for j in range(
                0, len(X_traj) - trainer.dynamic_loss_window, trainer.batch_size
            ):

                X_batch = X_traj[j : j + trainer.batch_size]
                encoded_k = trainer.model.encoder(X_batch)
                encoded_k = torch.concat([X_batch, encoded_k], axis=1)

                if trainer.model.decoder_trainable:
                    decoded_k = trainer.model.decoder(encoded_k)
                else:
                    decoded_k = torch.matmul(trainer.model.c_block, encoded_k.T).T

                U_batch = U_traj[j : j + trainer.batch_size]
                inp_k = trainer.model.b_block(U_batch)

                Y_pred = trainer.model(X_batch, U_batch)
                Y_batch = Y_traj[j : j + trainer.batch_size]

                for k in range(trainer.dynamic_loss_window - 1):
                    time_shifted_m_steps = encoded_k[: -k - 1]
                    u_input_m_steps = inp_k[: -k - 1]
                    for l in range(k + 1):
                        time_shifted_m_steps = (
                            trainer.model.k_block(time_shifted_m_steps)
                            + u_input_m_steps
                        )
                    encoded_m_steps = trainer.model.encoder(X_batch[k + 1 :])
                    encoded_m_steps = torch.concat(
                        [X_batch[k + 1 :], encoded_m_steps], axis=1
                    )
                    if k == 0:
                        loss_dynamics = trainer.loss_function(
                            time_shifted_m_steps, encoded_m_steps
                        )
                    else:
                        loss_dynamics += trainer.loss_function(
                            time_shifted_m_steps, encoded_m_steps
                        )
                loss_dynamics = loss_dynamics / (trainer.dynamic_loss_window - 1)

                loss_reconstruction = trainer.loss_function(decoded_k, X_batch)
                loss_prediction = trainer.loss_function(Y_pred, Y_batch)
                loss = (
                    trainer.weight_reconstruction_loss * loss_reconstruction
                    + trainer.weight_prediction_loss * loss_prediction
                    + trainer.weight_dynamics_loss * loss_dynamics
                )

                trainer.optimizer.zero_grad()
                loss.backward()
                trainer.optimizer.step()
                losses.append(loss.item())

        mean_epoch_loss = torch.mean(torch.tensor(losses)).item()
        loss_epoch_mean.append(mean_epoch_loss)

        print(f"Finished epoch {epoch+1}, mean loss for the epoch = {mean_epoch_loss}")

    return torch.tensor(loss_epoch_mean)


def train_temporal_model_with_control_input(trainer):
    """
    Trains a temporal Koopman model with control input using the provided trainer configuration.

    Args:
        trainer (Trainer): An instance of the Trainer class containing the model, data, and training parameters.

    Returns:
        torch.Tensor: A tensor containing the mean loss for each epoch.
    """

    X, Y, U = get_temporal_dataset(
        trajectory=trainer.trajectory,
        input=trainer.input,
        time_window=trainer.model.time_window,
    )

    if trainer.model.model_type == "TCN":
        if X.shape[-1] != trainer.model.time_window:
            X = X.transpose(-2, -1)

    X = X.to(trainer.device)
    Y = Y.to(trainer.device)
    U = U.to(trainer.device)

    loss_epoch_mean = []

    for epoch in range(trainer.num_epochs):

        for i in tqdm(range(X.shape[0])):
            X_input = X[i]
            Y_input = Y[i]
            U_input = U[i]

            losses = []
            for k in range(
                0, X_input.shape[0] - trainer.dynamic_loss_window, trainer.batch_size
            ):

                X_batch = X_input[k : k + trainer.batch_size, :, :]
                Y_batch = Y_input[k : k + trainer.batch_size, :]
                U_batch = U_input[k : k + trainer.batch_size, :]

                if trainer.model.model_type == "TCN":
                    tcn_k = trainer.model.tcn(X_batch)
                    encoded_k = trainer.model.encoder(tcn_k)
                    encoded_k = torch.concat([X_batch[:, :, -1], encoded_k], axis=1)

                if trainer.model.model_type == "LSTM":
                    lstm_k = trainer.model.lstm(X_batch)[0][:, -1, :]
                    encoded_k = trainer.model.encoder(lstm_k)
                    encoded_k = torch.concat([X_batch[:, -1, :], encoded_k], axis=1)

                if trainer.model.model_type == "GRU":
                    gru_k = trainer.model.gru(X_batch)[0][:, -1, :]
                    encoded_k = trainer.model.encoder(gru_k)
                    encoded_k = torch.concat([X_batch[:, -1, :], encoded_k], axis=1)

                if trainer.model.decoder_trainable:
                    decoded_k = trainer.model.decoder(encoded_k)
                else:
                    decoded_k = torch.matmul(trainer.model.c_block, encoded_k.T).T

                inp_k = trainer.model.b_block(U_batch)

                Y_pred = trainer.model(X_batch, U_batch)
                Y_batch = Y_input[k : k + trainer.batch_size]

                for k in range(trainer.dynamic_loss_window - 1):
                    time_shifted_m_steps = encoded_k[: -k - 1]
                    u_input_m_steps = inp_k[: -k - 1]
                    for l in range(k + 1):
                        time_shifted_m_steps = (
                            trainer.model.k_block(time_shifted_m_steps)
                            + u_input_m_steps
                        )

                    if trainer.model.model_type == "TCN":
                        encoded_m_steps = trainer.model.encoder(tcn_k[k + 1 :])
                        encoded_m_steps = torch.concat(
                            [X_batch[k + 1 :, :, -1], encoded_m_steps], axis=1
                        )

                    if trainer.model.model_type == "LSTM":
                        encoded_m_steps = trainer.model.encoder(lstm_k[k + 1 :])
                        encoded_m_steps = torch.concat(
                            [X_batch[k + 1 :, -1, :], encoded_m_steps], axis=1
                        )

                    if trainer.model.model_type == "GRU":
                        encoded_m_steps = trainer.model.encoder(gru_k[k + 1 :])
                        encoded_m_steps = torch.concat(
                            [X_batch[k + 1 :, -1, :], encoded_m_steps], axis=1
                        )

                    if k == 0:
                        loss_dynamics = trainer.loss_function(
                            time_shifted_m_steps, encoded_m_steps
                        )
                    else:
                        loss_dynamics += trainer.loss_function(
                            time_shifted_m_steps, encoded_m_steps
                        )
                loss_dynamics = loss_dynamics / (trainer.dynamic_loss_window - 1)

                if trainer.model.model_type == "TCN":
                    loss_reconstruction = trainer.loss_function(
                        decoded_k, X_batch[:, :, -1]
                    )
                else:
                    loss_reconstruction = trainer.loss_function(
                        decoded_k, X_batch[:, -1, :]
                    )
                loss_prediction = trainer.loss_function(Y_pred, Y_batch)
                loss = (
                    trainer.weight_reconstruction_loss * loss_reconstruction
                    + trainer.weight_prediction_loss * loss_prediction
                    + trainer.weight_dynamics_loss * loss_dynamics
                )

                trainer.optimizer.zero_grad()
                loss.backward()
                trainer.optimizer.step()
                losses.append(loss.item())

        mean_epoch_loss = torch.mean(torch.tensor(losses)).item()
        loss_epoch_mean.append(mean_epoch_loss)

        print(f"Finished epoch {epoch+1}, mean loss for the epoch = {mean_epoch_loss}")

    return torch.tensor(loss_epoch_mean)


def train_nontemporal_model_without_control_input(trainer):
    """
    Trains a non-temporal Koopman model without control input using the provided trainer configuration.

    Args:
        trainer (Trainer): An instance of the Trainer class containing the model, data, and training parameters.

    Returns:
        torch.Tensor: A tensor containing the mean loss for each epoch.
    """

    loss_epoch_mean = []

    for epoch in range(trainer.num_epochs):

        for i in tqdm(range(trainer.trajectory.shape[0])):

            X_traj = trainer.trajectory[i, :, :-1].T
            Y_traj = trainer.trajectory[i, :, 1:].T

            losses = []
            for j in range(
                0, len(X_traj) - trainer.dynamic_loss_window, trainer.batch_size
            ):

                X_batch = X_traj[j : j + trainer.batch_size]
                encoded_k = trainer.model.encoder(X_batch)
                encoded_k = torch.concat([X_batch, encoded_k], axis=1)

                if trainer.model.decoder_trainable:
                    decoded_k = trainer.model.decoder(encoded_k)
                else:
                    decoded_k = torch.matmul(trainer.model.c_block, encoded_k.T).T

                U_batch = torch.zeros((X_batch.shape[0], trainer.model.num_inputs)).to(
                    trainer.device
                )

                Y_pred = trainer.model(X_batch, U_batch)
                Y_batch = Y_traj[j : j + trainer.batch_size]

                for k in range(trainer.dynamic_loss_window - 1):
                    time_shifted_m_steps = encoded_k[: -k - 1]
                    for l in range(k + 1):
                        time_shifted_m_steps = trainer.model.k_block(
                            time_shifted_m_steps
                        )
                    encoded_m_steps = trainer.model.encoder(X_batch[k + 1 :])
                    encoded_m_steps = torch.concat(
                        [X_batch[k + 1 :], encoded_m_steps], axis=1
                    )
                    if k == 0:
                        loss_dynamics = trainer.loss_function(
                            time_shifted_m_steps, encoded_m_steps
                        )
                    else:
                        loss_dynamics += trainer.loss_function(
                            time_shifted_m_steps, encoded_m_steps
                        )

                loss_dynamics = loss_dynamics / (trainer.dynamic_loss_window - 1)
                loss_reconstruction = trainer.loss_function(decoded_k, X_batch)
                loss_prediction = trainer.loss_function(Y_pred, Y_batch)
                loss = (
                    trainer.weight_reconstruction_loss * loss_reconstruction
                    + trainer.weight_prediction_loss * loss_prediction
                    + trainer.weight_dynamics_loss * loss_dynamics
                )

                trainer.optimizer.zero_grad()
                loss.backward()
                trainer.optimizer.step()
                losses.append(loss.item())

        mean_epoch_loss = torch.mean(torch.tensor(losses)).item()
        loss_epoch_mean.append(mean_epoch_loss)

        print(f"Finished epoch {epoch+1}, mean loss for the epoch = {mean_epoch_loss}")

    return torch.tensor(loss_epoch_mean)


def train_temporal_model_without_control_input(trainer):
    """
    Trains a temporal Koopman model without control input using the provided trainer configuration.

    Args:
        trainer (Trainer): An instance of the Trainer class containing the model, data, and training parameters.

    Returns:
        torch.Tensor: A tensor containing the mean loss for each epoch.
    """

    X, Y, U = get_temporal_dataset(
        trajectory=trainer.trajectory,
        input=trainer.input,
        time_window=trainer.model.time_window,
    )

    if trainer.model.model_type == "TCN":
        if X.shape[-1] != trainer.model.time_window:
            X = X.transpose(-2, -1)

    X = X.to(trainer.device)
    Y = Y.to(trainer.device)

    loss_epoch_mean = []

    for epoch in range(trainer.num_epochs):

        for i in tqdm(range(X.shape[0])):
            X_input = X[i]
            Y_input = Y[i]

            losses = []
            for k in range(
                0, X_input.shape[0] - trainer.dynamic_loss_window, trainer.batch_size
            ):

                X_batch = X_input[k : k + trainer.batch_size, :, :]
                Y_batch = Y_input[k : k + trainer.batch_size, :]

                if trainer.model.model_type == "TCN":
                    tcn_k = trainer.model.tcn(X_batch)
                    encoded_k = trainer.model.encoder(tcn_k)
                    encoded_k = torch.concat([X_batch[:, :, -1], encoded_k], axis=1)

                if trainer.model.model_type == "LSTM":
                    lstm_k = trainer.model.lstm(X_batch)[0][:, -1, :]
                    encoded_k = trainer.model.encoder(lstm_k)
                    encoded_k = torch.concat([X_batch[:, -1, :], encoded_k], axis=1)

                if trainer.model.model_type == "GRU":
                    gru_k = trainer.model.gru(X_batch)[0][:, -1, :]
                    encoded_k = trainer.model.encoder(gru_k)
                    encoded_k = torch.concat([X_batch[:, -1, :], encoded_k], axis=1)

                if trainer.model.decoder_trainable:
                    decoded_k = trainer.model.decoder(encoded_k)
                else:
                    decoded_k = torch.matmul(trainer.model.c_block, encoded_k.T).T

                U_batch = torch.zeros((X_batch.shape[0], trainer.model.num_inputs)).to(
                    trainer.device
                )

                Y_pred = trainer.model(X_batch, U_batch)
                Y_batch = Y_input[k : k + trainer.batch_size]

                for k in range(trainer.dynamic_loss_window - 1):
                    time_shifted_m_steps = encoded_k[: -k - 1]
                    for l in range(k + 1):
                        time_shifted_m_steps = trainer.model.k_block(
                            time_shifted_m_steps
                        )

                    if trainer.model.model_type == "TCN":
                        encoded_m_steps = trainer.model.encoder(tcn_k[k + 1 :])
                        encoded_m_steps = torch.concat(
                            [X_batch[k + 1 :, :, -1], encoded_m_steps], axis=1
                        )

                    if trainer.model.model_type == "LSTM":
                        encoded_m_steps = trainer.model.encoder(lstm_k[k + 1 :])
                        encoded_m_steps = torch.concat(
                            [X_batch[k + 1 :, -1, :], encoded_m_steps], axis=1
                        )

                    if trainer.model.model_type == "GRU":
                        encoded_m_steps = trainer.model.encoder(gru_k[k + 1 :])
                        encoded_m_steps = torch.concat(
                            [X_batch[k + 1 :, -1, :], encoded_m_steps], axis=1
                        )

                    if k == 0:
                        loss_dynamics = trainer.loss_function(
                            time_shifted_m_steps, encoded_m_steps
                        )
                    else:
                        loss_dynamics += trainer.loss_function(
                            time_shifted_m_steps, encoded_m_steps
                        )

                loss_dynamics = loss_dynamics / (trainer.dynamic_loss_window - 1)
                if trainer.model.model_type == "TCN":
                    loss_reconstruction = trainer.loss_function(
                        decoded_k, X_batch[:, :, -1]
                    )
                else:
                    loss_reconstruction = trainer.loss_function(
                        decoded_k, X_batch[:, -1, :]
                    )
                loss_prediction = trainer.loss_function(Y_pred, Y_batch)
                loss = (
                    trainer.weight_reconstruction_loss * loss_reconstruction
                    + trainer.weight_prediction_loss * loss_prediction
                    + trainer.weight_dynamics_loss * loss_dynamics
                )

                trainer.optimizer.zero_grad()
                loss.backward()
                trainer.optimizer.step()
                losses.append(loss.item())

        mean_epoch_loss = torch.mean(torch.tensor(losses)).item()
        loss_epoch_mean.append(mean_epoch_loss)

        print(f"Finished epoch {epoch+1}, mean loss for the epoch = {mean_epoch_loss}")

    return torch.tensor(loss_epoch_mean)


def train_nontemporal_b_block_with_control_input(trainer):
    """
    Trains the b_block of a non-temporal Koopman model with control input using the provided trainer configuration.

    Args:
        trainer (Trainer): An instance of the Trainer class containing the model, data, and training parameters.

    Returns:
        torch.Tensor: A tensor containing the mean loss for each epoch.
    """

    K, B, C, enc = get_koopman_model(trainer.model)
    K = K.to(device=trainer.device)
    B = B.to(device=trainer.device)
    C = C.to(device=trainer.device)

    loss_epoch_mean = []

    for epoch in range(trainer.num_epochs):

        for i in tqdm(range(trainer.trajectory.shape[0])):
            X_traj = trainer.trajectory[i, :, :-1].T
            Y_traj = trainer.trajectory[i, :, 1:].T
            U_traj = trainer.input[i, :, :-1].T

            losses = []
            for j in range(
                0, len(X_traj) - trainer.dynamic_loss_window, trainer.batch_size
            ):

                X_batch = X_traj[j : j + trainer.batch_size]
                U_batch = U_traj[j : j + trainer.batch_size]
                Y_batch = Y_traj[j : j + trainer.batch_size]

                encoded_k = enc(X_batch)
                z_k_plus_one = torch.matmul(K, encoded_k.T).T + trainer.model.b_block(
                    U_batch
                )
                Y_pred = torch.matmul(C, z_k_plus_one.T).T

                for k in range(trainer.dynamic_loss_window - 1):
                    time_shifted_m_steps = encoded_k[: -k - 1]
                    for l in range(k + 1):
                        time_shifted_m_steps = torch.matmul(
                            K, time_shifted_m_steps.T
                        ).T + trainer.model.b_block(U_batch[k + 1 :])

                    encoded_m_steps = enc(X_batch[k + 1 :])
                    if k == 0:
                        loss_dynamics = trainer.loss_function(
                            torch.matmul(C, time_shifted_m_steps.T),
                            torch.matmul(C, encoded_m_steps.T),
                        )
                    else:
                        loss_dynamics += trainer.loss_function(
                            torch.matmul(C, time_shifted_m_steps.T),
                            torch.matmul(C, encoded_m_steps.T),
                        )

                loss_dynamics = loss_dynamics / (trainer.dynamic_loss_window - 1)
                loss_prediction = trainer.loss_function(Y_pred, Y_batch)
                loss = (
                    trainer.weight_prediction_loss * loss_prediction
                    + trainer.weight_dynamics_loss * loss_dynamics
                )

                trainer.optimizer.zero_grad()
                loss.backward()
                trainer.optimizer.step()
                losses.append(loss.item())

        mean_epoch_loss = torch.mean(torch.tensor(losses)).item()
        loss_epoch_mean.append(mean_epoch_loss)

        print(f"Finished epoch {epoch+1}, mean loss for the epoch = {mean_epoch_loss}")

    return torch.tensor(loss_epoch_mean)


def train_temporal_b_block_with_control_input(trainer):
    """
    Trains the b_block of a temporal Koopman model with control input using the provided trainer configuration.

    Args:
        trainer (Trainer): An instance of the Trainer class containing the model, data, and training parameters.

    Returns:
        torch.Tensor: A tensor containing the mean loss for each epoch.
    """

    X, Y, U = get_temporal_dataset(
        trajectory=trainer.trajectory,
        input=trainer.input,
        time_window=trainer.model.time_window,
    )

    if trainer.model.model_type == "TCN":
        if X.shape[-1] != trainer.model.time_window:
            X = X.transpose(-2, -1)

    X = X.to(trainer.device)
    Y = Y.to(trainer.device)
    U = U.to(trainer.device)

    K, B, C, enc = get_koopman_model(trainer.model)
    K = K.to(device=trainer.device)
    B = B.to(device=trainer.device)
    C = C.to(device=trainer.device)

    loss_epoch_mean = []

    for epoch in range(trainer.num_epochs):

        for i in tqdm(range(X.shape[0])):
            X_input = X[i]
            Y_input = Y[i]
            U_input = U[i]

            losses = []
            for j in range(
                0, X_input.shape[0] - trainer.dynamic_loss_window, trainer.batch_size
            ):

                X_batch = X_input[j : j + trainer.batch_size, :, :]
                Y_batch = Y_input[j : j + trainer.batch_size]
                U_batch = U_input[j : j + trainer.batch_size, :]

                encoded_k = enc(X_batch)
                z_k_plus_one = torch.matmul(K, encoded_k.T).T + trainer.model.b_block(
                    U_batch
                )
                Y_pred = torch.matmul(C, z_k_plus_one.T).T

                for k in range(trainer.dynamic_loss_window - 1):
                    time_shifted_m_steps = encoded_k[: -k - 1]
                    for l in range(k + 1):
                        time_shifted_m_steps = torch.matmul(
                            K, time_shifted_m_steps.T
                        ).T + trainer.model.b_block(U_batch[k + 1 :])

                    encoded_m_steps = enc(X_batch[k + 1 :])
                    if k == 0:
                        loss_dynamics = trainer.loss_function(
                            torch.matmul(C, time_shifted_m_steps.T),
                            torch.matmul(C, encoded_m_steps.T),
                        )
                    else:
                        loss_dynamics += trainer.loss_function(
                            torch.matmul(C, time_shifted_m_steps.T),
                            torch.matmul(C, encoded_m_steps.T),
                        )

                loss_dynamics = loss_dynamics / (trainer.dynamic_loss_window - 1)
                loss_prediction = trainer.loss_function(Y_pred, Y_batch)
                loss = (
                    trainer.weight_prediction_loss * loss_prediction
                    + trainer.weight_dynamics_loss * loss_dynamics
                )

                trainer.optimizer.zero_grad()
                loss.backward()
                trainer.optimizer.step()
                losses.append(loss.item())

        mean_epoch_loss = torch.mean(torch.tensor(losses)).item()
        loss_epoch_mean.append(mean_epoch_loss)

        print(f"Finished epoch {epoch+1}, mean loss for the epoch = {mean_epoch_loss}")

    return torch.tensor(loss_epoch_mean)
