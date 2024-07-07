import torch

from pykoopman_ae.default_params import *
from pykoopman_ae.params_test import params_test
from pykoopman_ae.system_extraction import get_koopman_model
from pykoopman_ae.training import Trainer


class MLP_AE(torch.nn.Module):
    """
    MLP_AE is a multi-layer perceptron based Koopman autoencoder model.
    It includes encoder, k_block, b_block, and an optional trainable decoder.

    Parameters:
        params (dict): A dictionary containing model parameters.
    """

    def __init__(self, params):
        """
        Initializes the MLP_AE model with the given parameters
                        or default parameters if not provided.

        Parameters:
            params (dict): A dictionary containing the parameters for the model.
        """

        # Use default parameters if not provided in params
        for key, value in default_params_mlp.items():
            if key not in params:
                params[key] = value

        self.model_type = "MLP"

        self.num_original_states = params["num_original_states"]
        self.num_lifted_states = params["num_lifted_states"]
        self.num_inputs = params["num_inputs"]

        self.encoder_layers = params["encoder_layers"]
        self.k_block_layers = params["k_block_layers"]
        self.b_block_layers = params["b_block_layers"]
        self.decoder_trainable = params["decoder_trainable"]
        self.decoder_layers = params["decoder_layers"]

        # Perform a test on parameters
        params_test(self)

        super().__init__()

        # Encoder Block
        encoder_layers = [
            torch.nn.Linear(
                self.num_original_states, self.encoder_layers[0], bias=False
            ),
            torch.nn.Tanh(),
        ]
        for i in range(len(self.encoder_layers) - 1):
            encoder_layers.extend(
                [
                    torch.nn.Linear(
                        self.encoder_layers[i], self.encoder_layers[i + 1], bias=False
                    ),
                    torch.nn.Tanh(),
                ]
            )
        encoder_layers.extend(
            [
                torch.nn.Linear(
                    self.encoder_layers[-1],
                    self.num_lifted_states - self.num_original_states,
                    bias=False,
                ),
                torch.nn.Tanh(),
            ]
        )
        (encoder_layers)
        self.encoder = torch.nn.Sequential(*encoder_layers)

        # K Block
        k_block_layers = [
            torch.nn.Linear(self.num_lifted_states, self.k_block_layers[0], bias=False)
        ]
        for i in range(len(self.k_block_layers) - 1):
            k_block_layers.extend(
                [
                    torch.nn.Linear(
                        self.k_block_layers[i], self.k_block_layers[i + 1], bias=False
                    )
                ]
            )
        k_block_layers.extend(
            [
                torch.nn.Linear(
                    self.k_block_layers[-1], self.num_lifted_states, bias=False
                ),
            ]
        )
        (k_block_layers)
        self.k_block = torch.nn.Sequential(*k_block_layers)

        # B Block
        b_block_layers = [
            torch.nn.Linear(self.num_inputs, self.b_block_layers[0], bias=False)
        ]
        for i in range(len(self.b_block_layers) - 1):
            b_block_layers.extend(
                [
                    torch.nn.Linear(
                        self.b_block_layers[i], self.b_block_layers[i + 1], bias=False
                    )
                ]
            )
        b_block_layers.extend(
            [
                torch.nn.Linear(
                    self.b_block_layers[-1], self.num_lifted_states, bias=False
                ),
            ]
        )
        (b_block_layers)
        self.b_block = torch.nn.Sequential(*b_block_layers)

        if self.decoder_trainable:
            # Decoder Block
            decoder_layers = [
                torch.nn.Linear(
                    self.num_lifted_states, self.decoder_layers[0], bias=False
                )
            ]
            for i in range(len(self.decoder_layers) - 1):
                decoder_layers.extend(
                    [
                        torch.nn.Linear(
                            self.decoder_layers[i],
                            self.decoder_layers[i + 1],
                            bias=False,
                        )
                    ]
                )
            decoder_layers.extend(
                [
                    torch.nn.Linear(
                        self.decoder_layers[-1], self.num_original_states, bias=False
                    ),
                ]
            )
            (decoder_layers)
            self.decoder = torch.nn.Sequential(*decoder_layers)

        else:
            # C Block
            self.c_block = torch.nn.Parameter(
                torch.eye(self.num_original_states, self.num_lifted_states)
            )
            self.c_block.requires_grad = False

    def forward(self, x, u):
        """
        Defines the forward pass of the MLP_AE model.

        Parameters:
            x (torch.Tensor): The input state tensor
                                with shape (batch_size, num_original_states).
            u (torch.Tensor): The control input tensor
                                with shape (batch_size, num_inputs).

        Returns:
            torch.Tensor: The decoded output state tensor
                                with shape (batch_size, num_original_states).
        """

        encoded = self.encoder(x)
        encoded = torch.concat([x, encoded], axis=1)

        inp = self.b_block(u)
        time_shifted = self.k_block(encoded) + inp

        if self.decoder_trainable:
            decoded = self.decoder(time_shifted)
        else:
            decoded = torch.matmul(self.c_block, time_shifted.T).T

        return decoded

    def learn_koopman_model(
        self,
        trajectory,
        input,
        loss_function=None,
        optimizer=None,
        dynamic_loss_window=None,
        num_epochs=None,
        batch_size=None,
        weight_reconstruction_loss=None,
        weight_prediction_loss=None,
        weight_dynamics_loss=None,
    ):

        ModelTrainer = Trainer(
            model=self,
            trajectory=trajectory,
            input=input,
            loss_function=loss_function,
            optimizer=optimizer,
            dynamic_loss_window=dynamic_loss_window,
            num_epochs=num_epochs,
            batch_size=batch_size,
            weight_reconstruction_loss=weight_reconstruction_loss,
            weight_prediction_loss=weight_prediction_loss,
            weight_dynamics_loss=weight_dynamics_loss,
        )

        loss = ModelTrainer.learn_koopman_model()

        return loss

    def learn_koopman_eigendynamics(
        self,
        trajectory,
        loss_function=None,
        optimizer=None,
        dynamic_loss_window=None,
        num_epochs=None,
        batch_size=None,
        weight_reconstruction_loss=None,
        weight_prediction_loss=None,
        weight_dynamics_loss=None,
    ):

        ModelTrainer = Trainer(
            model=self,
            trajectory=trajectory,
            loss_function=loss_function,
            optimizer=optimizer,
            dynamic_loss_window=dynamic_loss_window,
            num_epochs=num_epochs,
            batch_size=batch_size,
            weight_reconstruction_loss=weight_reconstruction_loss,
            weight_prediction_loss=weight_prediction_loss,
            weight_dynamics_loss=weight_dynamics_loss,
        )

        loss = ModelTrainer.learn_koopman_eigendynamics()

        return loss

    def learn_input_matrix(
        self,
        trajectory,
        input,
        loss_function=None,
        optimizer=None,
        dynamic_loss_window=None,
        num_epochs=None,
        batch_size=None,
        weight_prediction_loss=None,
        weight_dynamics_loss=None,
    ):

        ModelTrainer = Trainer(
            model=self,
            trajectory=trajectory,
            input=input,
            loss_function=loss_function,
            optimizer=optimizer,
            dynamic_loss_window=dynamic_loss_window,
            num_epochs=num_epochs,
            batch_size=batch_size,
            weight_prediction_loss=weight_prediction_loss,
            weight_dynamics_loss=weight_dynamics_loss,
        )

        loss = ModelTrainer.learn_input_matrix()

        return loss

    def get_koopman_model(self):
        """
        Extracts the lifted system matrices K, B, and C from the trained MLP_AE model.

        The lifted system matrices represent the Koopman dynamics (K),
        input matrix (B), and output matrix (C) of the lifted Koopman system model.
        Additionally, it provides a function to compute the lifted states from the original states.

        Returns:
                tuple: A tuple containing the following elements:
                        - K (torch.Tensor): The Koopman dynamics matrix
                                with shape (num_lifted_states, num_lifted_states).
                        - B (torch.Tensor): The input matrix
                                with shape (num_lifted_states, num_inputs).
                        - C (torch.Tensor): The output matrix
                                with shape (num_original_states, num_lifted_states).
                        - enc (function): A function that computes the lifted states from the original states.
                                The function takes a tensor `x` with shape
                                (batch_size, num_original_states) and returns
                                a tensor with shape (batch_size, num_lifted_states).
        """

        K, B, C, enc = get_koopman_model(self)
        return K, B, C, enc


class TCN_AE(torch.nn.Module):
    """
    TCN_AE is a Temporal Convolutional Network based Koopman autoencoder model.
    It includes encoder, k_block, b_block, and an optional trainable decoder.

    Parameters:
        params (dict): A dictionary containing model parameters.
    """

    def __init__(self, params):
        """
        Initializes the TCN_AE model with the given parameters
                        or default parameters if not provided.

        Parameters:
            params (dict): A dictionary containing the parameters for the model.
        """

        for key, value in default_params_tcn.items():
            if key not in params:
                params[key] = value

        self.model_type = "TCN"

        self.num_original_states = params["num_original_states"]
        self.num_lifted_states = params["num_lifted_states"]
        self.num_inputs = params["num_inputs"]

        self.encoder_layers = params["encoder_layers"]
        self.k_block_layers = params["k_block_layers"]
        self.b_block_layers = params["b_block_layers"]
        self.decoder_trainable = params["decoder_trainable"]
        self.decoder_layers = params["decoder_layers"]

        self.time_window = params["time_window"]
        self.tcn_channels = params["tcn_channels"]
        self.tcn_kernels = params["tcn_kernels"]

        params_test(self)

        super().__init__()

        # TCN Block
        tcn_layers = [
            torch.nn.Conv1d(
                in_channels=self.num_original_states,
                out_channels=self.tcn_channels[0],
                kernel_size=(self.tcn_kernels[0]),
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=False,
            ),
            torch.nn.Tanh(),
        ]
        for i in range(len(self.tcn_channels) - 1):
            tcn_layers.extend(
                [
                    torch.nn.Conv1d(
                        in_channels=self.tcn_channels[i],
                        out_channels=self.tcn_channels[i + 1],
                        kernel_size=(self.tcn_kernels[i + 1]),
                        stride=1,
                        padding=0,
                        dilation=1,
                        groups=1,
                        bias=False,
                    ),
                    torch.nn.Tanh(),
                ]
            )
        tcn_layers.extend([torch.nn.Flatten()])
        (tcn_layers)
        self.tcn = torch.nn.Sequential(*tcn_layers)

        tcn_out = self.time_window
        for k in self.tcn_kernels:
            tcn_out = tcn_out - k + 1

        assert (
            tcn_out > 0
        ), f"time_window: Kernel Sizes are large, increase time_window or reduce tcn_kernels."

        # Encoder Block
        encoder_layers = [
            torch.nn.Linear(
                self.tcn_channels[-1] * tcn_out,
                self.encoder_layers[0],
                bias=False,
            ),
            torch.nn.Tanh(),
        ]
        for i in range(len(self.encoder_layers) - 1):
            encoder_layers.extend(
                [
                    torch.nn.Linear(
                        self.encoder_layers[i], self.encoder_layers[i + 1], bias=False
                    ),
                    torch.nn.Tanh(),
                ]
            )
        encoder_layers.extend(
            [
                torch.nn.Linear(
                    self.encoder_layers[-1],
                    self.num_lifted_states - self.num_original_states,
                    bias=False,
                ),
                torch.nn.Tanh(),
            ]
        )
        (encoder_layers)
        self.encoder = torch.nn.Sequential(*encoder_layers)

        # K Block
        k_block_layers = [
            torch.nn.Linear(self.num_lifted_states, self.k_block_layers[0], bias=False)
        ]
        for i in range(len(self.k_block_layers) - 1):
            k_block_layers.extend(
                [
                    torch.nn.Linear(
                        self.k_block_layers[i], self.k_block_layers[i + 1], bias=False
                    )
                ]
            )
        k_block_layers.extend(
            [
                torch.nn.Linear(
                    self.k_block_layers[-1], self.num_lifted_states, bias=False
                ),
            ]
        )
        (k_block_layers)
        self.k_block = torch.nn.Sequential(*k_block_layers)

        # B Block
        b_block_layers = [
            torch.nn.Linear(self.num_inputs, self.b_block_layers[0], bias=False)
        ]
        for i in range(len(self.b_block_layers) - 1):
            b_block_layers.extend(
                [
                    torch.nn.Linear(
                        self.b_block_layers[i], self.b_block_layers[i + 1], bias=False
                    )
                ]
            )
        b_block_layers.extend(
            [
                torch.nn.Linear(
                    self.b_block_layers[-1], self.num_lifted_states, bias=False
                ),
            ]
        )
        (b_block_layers)
        self.b_block = torch.nn.Sequential(*b_block_layers)

        if self.decoder_trainable:
            # Decoder Block
            decoder_layers = [
                torch.nn.Linear(
                    self.num_lifted_states, self.decoder_layers[0], bias=False
                )
            ]
            for i in range(len(self.decoder_layers) - 1):
                decoder_layers.extend(
                    [
                        torch.nn.Linear(
                            self.decoder_layers[i],
                            self.decoder_layers[i + 1],
                            bias=False,
                        )
                    ]
                )
            decoder_layers.extend(
                [
                    torch.nn.Linear(
                        self.decoder_layers[-1], self.num_original_states, bias=False
                    ),
                ]
            )
            (decoder_layers)
            self.decoder = torch.nn.Sequential(*decoder_layers)

        else:
            # C Block
            self.c_block = torch.nn.Parameter(
                torch.eye(self.num_original_states, self.num_lifted_states)
            )
            self.c_block.requires_grad = False

    def forward(self, x, u):
        """
        Defines the forward pass of the TCN_AE model.

        Args:
            x (torch.Tensor): The input state tensor
                                with shape (batch_size, time_window, num_original_states).
            u (torch.Tensor): The input control tensor
                                with shape (batch_size, num_inputs).

        Returns:
            torch.Tensor: The output tensor with shape (batch_size, num_original_states).
        """

        if x.shape[-1] != self.time_window:
            x = x.transpose(-2, -1)

        tcn_output = self.tcn(x)

        encoded = self.encoder(tcn_output)
        encoded = torch.concat([x[:, :, -1], encoded], axis=1)

        inp = self.b_block(u)
        time_shifted = self.k_block(encoded) + inp

        if self.decoder_trainable:
            decoded = self.decoder(time_shifted)
        else:
            decoded = torch.matmul(self.c_block, time_shifted.T).T

        return decoded

    def learn_koopman_model(
        self,
        trajectory,
        input,
        loss_function=None,
        optimizer=None,
        dynamic_loss_window=None,
        num_epochs=None,
        batch_size=None,
        weight_reconstruction_loss=None,
        weight_prediction_loss=None,
        weight_dynamics_loss=None,
    ):

        ModelTrainer = Trainer(
            model=self,
            trajectory=trajectory,
            input=input,
            loss_function=loss_function,
            optimizer=optimizer,
            dynamic_loss_window=dynamic_loss_window,
            num_epochs=num_epochs,
            batch_size=batch_size,
            weight_reconstruction_loss=weight_reconstruction_loss,
            weight_prediction_loss=weight_prediction_loss,
            weight_dynamics_loss=weight_dynamics_loss,
        )

        loss = ModelTrainer.learn_koopman_model()

        return loss

    def learn_koopman_eigendynamics(
        self,
        trajectory,
        loss_function=None,
        optimizer=None,
        dynamic_loss_window=None,
        num_epochs=None,
        batch_size=None,
        weight_reconstruction_loss=None,
        weight_prediction_loss=None,
        weight_dynamics_loss=None,
    ):

        ModelTrainer = Trainer(
            model=self,
            trajectory=trajectory,
            loss_function=loss_function,
            optimizer=optimizer,
            dynamic_loss_window=dynamic_loss_window,
            num_epochs=num_epochs,
            batch_size=batch_size,
            weight_reconstruction_loss=weight_reconstruction_loss,
            weight_prediction_loss=weight_prediction_loss,
            weight_dynamics_loss=weight_dynamics_loss,
        )

        loss = ModelTrainer.learn_koopman_eigendynamics()

        return loss

    def learn_input_matrix(
        self,
        trajectory,
        input,
        loss_function=None,
        optimizer=None,
        dynamic_loss_window=None,
        num_epochs=None,
        batch_size=None,
        weight_prediction_loss=None,
        weight_dynamics_loss=None,
    ):

        ModelTrainer = Trainer(
            model=self,
            trajectory=trajectory,
            input=input,
            loss_function=loss_function,
            optimizer=optimizer,
            dynamic_loss_window=dynamic_loss_window,
            num_epochs=num_epochs,
            batch_size=batch_size,
            weight_prediction_loss=weight_prediction_loss,
            weight_dynamics_loss=weight_dynamics_loss,
        )

        loss = ModelTrainer.learn_input_matrix()

        return loss

    def get_koopman_model(self):
        """
        Extracts the lifted system matrices K, B, and C from the trained TCN_AE model.

        The lifted system matrices represent the Koopman dynamics (K),
        input matrix (B), and output matrix (C) of the lifted Koopman system model.
        Additionally, it provides a function to compute the lifted states from the original states.

        Returns:
                tuple: A tuple containing the following elements:
                        - K (torch.Tensor): The Koopman dynamics matrix
                                with shape (num_lifted_states, num_lifted_states).
                        - B (torch.Tensor): The input matrix
                                with shape (num_lifted_states, num_inputs).
                        - C (torch.Tensor): The output matrix
                                with shape (num_original_states, num_lifted_states).
                        - enc (function): A function that computes the lifted states from the original states.
                                The function takes a tensor `x` with shape
                                (batch_size, time_window, num_original_states) and returns
                                a tensor with shape (batch_size, num_lifted_states).
        """

        K, B, C, enc = get_koopman_model(self)
        return K, B, C, enc


class LSTM_AE(torch.nn.Module):
    def __init__(self, params):

        for key, value in default_params_lstm.items():
            if key not in params:
                params[key] = value

        self.model_type = "LSTM"

        self.num_original_states = params["num_original_states"]
        self.num_lifted_states = params["num_lifted_states"]
        self.num_inputs = params["num_inputs"]

        self.encoder_layers = params["encoder_layers"]
        self.k_block_layers = params["k_block_layers"]
        self.b_block_layers = params["b_block_layers"]
        self.decoder_trainable = params["decoder_trainable"]
        self.decoder_layers = params["decoder_layers"]

        self.time_window = params["time_window"]
        self.lstm_hidden_size = params["lstm_hidden_size"]
        self.lstm_num_layers = params["lstm_num_layers"]

        params_test(self)

        super().__init__()

        # LSTM Block
        self.lstm = torch.nn.LSTM(
            input_size=self.num_original_states,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_num_layers,
            batch_first=True,
        )

        # Encoder Block
        encoder_layers = [
            torch.nn.Linear(self.lstm_hidden_size, self.encoder_layers[0], bias=False),
            torch.nn.Tanh(),
        ]
        for i in range(len(self.encoder_layers) - 1):
            encoder_layers.extend(
                [
                    torch.nn.Linear(
                        self.encoder_layers[i], self.encoder_layers[i + 1], bias=False
                    ),
                    torch.nn.Tanh(),
                ]
            )
        encoder_layers.extend(
            [
                torch.nn.Linear(
                    self.encoder_layers[-1],
                    self.num_lifted_states - self.num_original_states,
                    bias=False,
                ),
                torch.nn.Tanh(),
            ]
        )
        (encoder_layers)
        self.encoder = torch.nn.Sequential(*encoder_layers)

        # K Block
        k_block_layers = [
            torch.nn.Linear(self.num_lifted_states, self.k_block_layers[0], bias=False)
        ]
        for i in range(len(self.k_block_layers) - 1):
            k_block_layers.extend(
                [
                    torch.nn.Linear(
                        self.k_block_layers[i], self.k_block_layers[i + 1], bias=False
                    )
                ]
            )
        k_block_layers.extend(
            [
                torch.nn.Linear(
                    self.k_block_layers[-1], self.num_lifted_states, bias=False
                ),
            ]
        )
        (k_block_layers)
        self.k_block = torch.nn.Sequential(*k_block_layers)

        # B Block
        b_block_layers = [
            torch.nn.Linear(self.num_inputs, self.b_block_layers[0], bias=False)
        ]
        for i in range(len(self.b_block_layers) - 1):
            b_block_layers.extend(
                [
                    torch.nn.Linear(
                        self.b_block_layers[i], self.b_block_layers[i + 1], bias=False
                    )
                ]
            )
        b_block_layers.extend(
            [
                torch.nn.Linear(
                    self.b_block_layers[-1], self.num_lifted_states, bias=False
                ),
            ]
        )
        (b_block_layers)
        self.b_block = torch.nn.Sequential(*b_block_layers)

        if self.decoder_trainable:
            # Decoder Block
            decoder_layers = [
                torch.nn.Linear(
                    self.num_lifted_states, self.decoder_layers[0], bias=False
                )
            ]
            for i in range(len(self.decoder_layers) - 1):
                decoder_layers.extend(
                    [
                        torch.nn.Linear(
                            self.decoder_layers[i],
                            self.decoder_layers[i + 1],
                            bias=False,
                        )
                    ]
                )
            decoder_layers.extend(
                [
                    torch.nn.Linear(
                        self.decoder_layers[-1], self.num_original_states, bias=False
                    ),
                ]
            )
            (decoder_layers)
            self.decoder = torch.nn.Sequential(*decoder_layers)

        else:
            # C Block
            self.c_block = torch.nn.Parameter(
                torch.eye(self.num_original_states, self.num_lifted_states)
            )
            self.c_block.requires_grad = False

    def forward(self, x, u):

        lstm_output, _status = self.lstm(x)
        lstm_output = lstm_output[:, -1, :]

        encoded = self.encoder(lstm_output)
        encoded = torch.concat([x[:, -1, :], encoded], axis=1)

        inp = self.b_block(u)
        time_shifted = self.k_block(encoded) + inp

        if self.decoder_trainable:
            decoded = self.decoder(time_shifted)
        else:
            decoded = torch.matmul(self.c_block, time_shifted.T).T

        return decoded

    def learn_koopman_model(
        self,
        trajectory,
        input,
        loss_function=None,
        optimizer=None,
        dynamic_loss_window=None,
        num_epochs=None,
        batch_size=None,
        weight_reconstruction_loss=None,
        weight_prediction_loss=None,
        weight_dynamics_loss=None,
    ):

        ModelTrainer = Trainer(
            model=self,
            trajectory=trajectory,
            input=input,
            loss_function=loss_function,
            optimizer=optimizer,
            dynamic_loss_window=dynamic_loss_window,
            num_epochs=num_epochs,
            batch_size=batch_size,
            weight_reconstruction_loss=weight_reconstruction_loss,
            weight_prediction_loss=weight_prediction_loss,
            weight_dynamics_loss=weight_dynamics_loss,
        )

        loss = ModelTrainer.learn_koopman_model()

        return loss

    def learn_koopman_eigendynamics(
        self,
        trajectory,
        loss_function=None,
        optimizer=None,
        dynamic_loss_window=None,
        num_epochs=None,
        batch_size=None,
        weight_reconstruction_loss=None,
        weight_prediction_loss=None,
        weight_dynamics_loss=None,
    ):

        ModelTrainer = Trainer(
            model=self,
            trajectory=trajectory,
            loss_function=loss_function,
            optimizer=optimizer,
            dynamic_loss_window=dynamic_loss_window,
            num_epochs=num_epochs,
            batch_size=batch_size,
            weight_reconstruction_loss=weight_reconstruction_loss,
            weight_prediction_loss=weight_prediction_loss,
            weight_dynamics_loss=weight_dynamics_loss,
        )

        loss = ModelTrainer.learn_koopman_eigendynamics()

        return loss

    def learn_input_matrix(
        self,
        trajectory,
        input,
        loss_function=None,
        optimizer=None,
        dynamic_loss_window=None,
        num_epochs=None,
        batch_size=None,
        weight_prediction_loss=None,
        weight_dynamics_loss=None,
    ):

        ModelTrainer = Trainer(
            model=self,
            trajectory=trajectory,
            input=input,
            loss_function=loss_function,
            optimizer=optimizer,
            dynamic_loss_window=dynamic_loss_window,
            num_epochs=num_epochs,
            batch_size=batch_size,
            weight_prediction_loss=weight_prediction_loss,
            weight_dynamics_loss=weight_dynamics_loss,
        )

        loss = ModelTrainer.learn_input_matrix()

        return loss

    def get_koopman_model(self):
        """
        Extracts the lifted system matrices K, B, and C from the trained LSTM_AE model.

        The lifted system matrices represent the Koopman dynamics (K),
        input matrix (B), and output matrix (C) of the lifted Koopman system model.
        Additionally, it provides a function to compute the lifted states from the original states.

        Returns:
                tuple: A tuple containing the following elements:
                        - K (torch.Tensor): The Koopman dynamics matrix
                                with shape (num_lifted_states, num_lifted_states).
                        - B (torch.Tensor): The input matrix
                                with shape (num_lifted_states, num_inputs).
                        - C (torch.Tensor): The output matrix
                                with shape (num_original_states, num_lifted_states).
                        - enc (function): A function that computes the lifted states from the original states.
                                The function takes a tensor `x` with shape
                                (batch_size, time_window, num_original_states) and returns
                                a tensor with shape (batch_size, num_lifted_states).
        """

        K, B, C, enc = get_koopman_model(self)
        return K, B, C, enc


class GRU_AE(torch.nn.Module):
    def __init__(self, params):

        for key, value in default_params_gru.items():
            if key not in params:
                params[key] = value

        self.model_type = "GRU"

        self.num_original_states = params["num_original_states"]
        self.num_lifted_states = params["num_lifted_states"]
        self.num_inputs = params["num_inputs"]

        self.encoder_layers = params["encoder_layers"]
        self.k_block_layers = params["k_block_layers"]
        self.b_block_layers = params["b_block_layers"]
        self.decoder_trainable = params["decoder_trainable"]
        self.decoder_layers = params["decoder_layers"]

        self.time_window = params["time_window"]
        self.gru_hidden_size = params["gru_hidden_size"]
        self.gru_num_layers = params["gru_num_layers"]

        params_test(self)

        super().__init__()

        # GRU Block
        self.gru = torch.nn.GRU(
            input_size=self.num_original_states,
            hidden_size=self.gru_hidden_size,
            num_layers=self.gru_num_layers,
            batch_first=True,
        )

        # Encoder Block
        encoder_layers = [
            torch.nn.Linear(self.gru_hidden_size, self.encoder_layers[0], bias=False),
            torch.nn.Tanh(),
        ]
        for i in range(len(self.encoder_layers) - 1):
            encoder_layers.extend(
                [
                    torch.nn.Linear(
                        self.encoder_layers[i], self.encoder_layers[i + 1], bias=False
                    ),
                    torch.nn.Tanh(),
                ]
            )
        encoder_layers.extend(
            [
                torch.nn.Linear(
                    self.encoder_layers[-1],
                    self.num_lifted_states - self.num_original_states,
                    bias=False,
                ),
                torch.nn.Tanh(),
            ]
        )
        (encoder_layers)
        self.encoder = torch.nn.Sequential(*encoder_layers)

        # K Block
        k_block_layers = [
            torch.nn.Linear(self.num_lifted_states, self.k_block_layers[0], bias=False)
        ]
        for i in range(len(self.k_block_layers) - 1):
            k_block_layers.extend(
                [
                    torch.nn.Linear(
                        self.k_block_layers[i], self.k_block_layers[i + 1], bias=False
                    )
                ]
            )
        k_block_layers.extend(
            [
                torch.nn.Linear(
                    self.k_block_layers[-1], self.num_lifted_states, bias=False
                ),
            ]
        )
        (k_block_layers)
        self.k_block = torch.nn.Sequential(*k_block_layers)

        # B Block
        b_block_layers = [
            torch.nn.Linear(self.num_inputs, self.b_block_layers[0], bias=False)
        ]
        for i in range(len(self.b_block_layers) - 1):
            b_block_layers.extend(
                [
                    torch.nn.Linear(
                        self.b_block_layers[i], self.b_block_layers[i + 1], bias=False
                    )
                ]
            )
        b_block_layers.extend(
            [
                torch.nn.Linear(
                    self.b_block_layers[-1], self.num_lifted_states, bias=False
                ),
            ]
        )
        (b_block_layers)
        self.b_block = torch.nn.Sequential(*b_block_layers)

        if self.decoder_trainable:
            # Decoder Block
            decoder_layers = [
                torch.nn.Linear(
                    self.num_lifted_states, self.decoder_layers[0], bias=False
                )
            ]
            for i in range(len(self.decoder_layers) - 1):
                decoder_layers.extend(
                    [
                        torch.nn.Linear(
                            self.decoder_layers[i],
                            self.decoder_layers[i + 1],
                            bias=False,
                        )
                    ]
                )
            decoder_layers.extend(
                [
                    torch.nn.Linear(
                        self.decoder_layers[-1], self.num_original_states, bias=False
                    ),
                ]
            )
            (decoder_layers)
            self.decoder = torch.nn.Sequential(*decoder_layers)

        else:
            # C Block
            self.c_block = torch.nn.Parameter(
                torch.eye(self.num_original_states, self.num_lifted_states)
            )
            self.c_block.requires_grad = False

    def forward(self, x, u):

        gru_output, _status = self.gru(x)
        gru_output = gru_output[:, -1, :]

        encoded = self.encoder(gru_output)
        encoded = torch.concat([x[:, -1, :], encoded], axis=1)

        inp = self.b_block(u)
        time_shifted = self.k_block(encoded) + inp

        if self.decoder_trainable:
            decoded = self.decoder(time_shifted)
        else:
            decoded = torch.matmul(self.c_block, time_shifted.T).T

        return decoded

    def learn_koopman_model(
        self,
        trajectory,
        input,
        loss_function=None,
        optimizer=None,
        dynamic_loss_window=None,
        num_epochs=None,
        batch_size=None,
        weight_reconstruction_loss=None,
        weight_prediction_loss=None,
        weight_dynamics_loss=None,
    ):

        ModelTrainer = Trainer(
            model=self,
            trajectory=trajectory,
            input=input,
            loss_function=loss_function,
            optimizer=optimizer,
            dynamic_loss_window=dynamic_loss_window,
            num_epochs=num_epochs,
            batch_size=batch_size,
            weight_reconstruction_loss=weight_reconstruction_loss,
            weight_prediction_loss=weight_prediction_loss,
            weight_dynamics_loss=weight_dynamics_loss,
        )

        loss = ModelTrainer.learn_koopman_model()

        return loss

    def learn_koopman_eigendynamics(
        self,
        trajectory,
        loss_function=None,
        optimizer=None,
        dynamic_loss_window=None,
        num_epochs=None,
        batch_size=None,
        weight_reconstruction_loss=None,
        weight_prediction_loss=None,
        weight_dynamics_loss=None,
    ):

        ModelTrainer = Trainer(
            model=self,
            trajectory=trajectory,
            loss_function=loss_function,
            optimizer=optimizer,
            dynamic_loss_window=dynamic_loss_window,
            num_epochs=num_epochs,
            batch_size=batch_size,
            weight_reconstruction_loss=weight_reconstruction_loss,
            weight_prediction_loss=weight_prediction_loss,
            weight_dynamics_loss=weight_dynamics_loss,
        )

        loss = ModelTrainer.learn_koopman_eigendynamics()

        return loss

    def learn_input_matrix(
        self,
        trajectory,
        input,
        loss_function=None,
        optimizer=None,
        dynamic_loss_window=None,
        num_epochs=None,
        batch_size=None,
        weight_prediction_loss=None,
        weight_dynamics_loss=None,
    ):

        ModelTrainer = Trainer(
            model=self,
            trajectory=trajectory,
            input=input,
            loss_function=loss_function,
            optimizer=optimizer,
            dynamic_loss_window=dynamic_loss_window,
            num_epochs=num_epochs,
            batch_size=batch_size,
            weight_prediction_loss=weight_prediction_loss,
            weight_dynamics_loss=weight_dynamics_loss,
        )

        loss = ModelTrainer.learn_input_matrix()

        return loss

    def get_koopman_model(self):
        """
        Extracts the lifted system matrices K, B, and C from the trained GRU_AE model.

        The lifted system matrices represent the Koopman dynamics (K),
        input matrix (B), and output matrix (C) of the lifted Koopman system model.
        Additionally, it provides a function to compute the lifted states from the original states.

        Returns:
                tuple: A tuple containing the following elements:
                        - K (torch.Tensor): The Koopman dynamics matrix
                                with shape (num_lifted_states, num_lifted_states).
                        - B (torch.Tensor): The input matrix
                                with shape (num_lifted_states, num_inputs).
                        - C (torch.Tensor): The output matrix
                                with shape (num_original_states, num_lifted_states).
                        - enc (function): A function that computes the lifted states from the original states.
                                The function takes a tensor `x` with shape
                                (batch_size, time_window, num_original_states) and returns
                                a tensor with shape (batch_size, num_lifted_states).
        """

        K, B, C, enc = get_koopman_model(self)
        return K, B, C, enc
