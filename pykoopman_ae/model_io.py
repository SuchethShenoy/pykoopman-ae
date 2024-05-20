import torch

from pykoopman_ae.system_extraction import get_koopman_system

import torch


def save_model_with_params(model, params, path):
    """
    Saves the model's state dictionary and the model parameters to a specified path.

    Args:
        model (torch.nn.Module): The model to be saved.
        params (dict): The parameters dictionary used to initialize the model.
        path (str): The file path to save the model and parameters.

    Returns:
        None
    """
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "params": params,
        },
        path,
    )
    print("Model and parameters saved successfully at", path)


def load_model_with_params(path, model_class):
    """
    Loads the model's state dictionary and parameters from a specified path and initializes
    the model using the loaded parameters.

    Args:
        path (str): The file path to load the model and parameters from.
        model_class (class): The class of the model to be instantiated.

    Returns:
        tuple: A tuple containing the following elements:
            - model (torch.nn.Module): The model loaded with the state dictionary and parameters.
            - params (dict): The parameters dictionary used to initialize the model.
    """
    checkpoint = torch.load(path)
    params = checkpoint["params"]
    model = model_class(params)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("Model and parameters loaded successfully from", path)
    return model, params


def save_koopman_system(model, params, path):
    """
    Saves the Koopman system matrices (K, B, C) and the encoder parameters.

    Args:
        model (torch.nn.Module): The trained model.
        path (str): The file path to save the Koopman system.
    """
    # Extract Koopman matrices and encoder function
    K, B, C, enc = get_koopman_system(model)

    # Save the matrices and encoder state dict
    torch.save(
        {
            "K": K,
            "B": B,
            "C": C,
            "encoder_state_dict": model.encoder.state_dict(),
            "model_params": params,
            "model_type": model.model_type,
            "time_window": model.time_window if hasattr(model, "time_window") else None,
        },
        path,
    )

    print(f"Koopman system saved successfully at {path}")


def load_koopman_system(path, model_class):
    """
    Loads the Koopman system matrices (K, B, C) and the encoder parameters.

    Args:
        path (str): The file path to load the Koopman system from.
        model_class (class): The class of the model to instantiate.

    Returns:
        tuple: A tuple containing the Koopman matrices (K, B, C) and the encoder function (enc).
    """
    # Load the checkpoint
    checkpoint = torch.load(path)

    # Extract matrices and parameters
    K = checkpoint["K"]
    B = checkpoint["B"]
    C = checkpoint["C"]
    model_params = checkpoint["model_params"]
    model_type = checkpoint["model_type"]
    time_window = checkpoint["time_window"]

    # Instantiate the model and load the encoder state dict
    model = model_class(model_params)
    model.encoder.load_state_dict(checkpoint["encoder_state_dict"])

    # Define the encoder function based on the model type
    def enc(x):
        if model_type == "MLP":
            encoded = model.encoder(x)
            lifted_states = torch.concat([x, encoded], axis=1)
            return lifted_states

        if model_type == "TCN":
            if x.shape[-1] != time_window:
                x = x.transpose(-2, -1)
            tcn_output = model.tcn(x)
            encoded = model.encoder(tcn_output)
            lifted_states = torch.concat([x[:, :, -1], encoded], axis=1)
            return lifted_states

        if model_type == "LSTM":
            lstm_output, _status = model.lstm(x)
            lstm_output = lstm_output[:, -1, :]
            encoded = model.encoder(lstm_output)
            lifted_states = torch.concat([x[:, -1, :], encoded], axis=1)
            return lifted_states

        if model_type == "GRU":
            gru_output, _status = model.gru(x)
            gru_output = gru_output[:, -1, :]
            encoded = model.encoder(gru_output)
            lifted_states = torch.concat([x[:, -1, :], encoded], axis=1)
            return lifted_states

    print(f"Koopman system loaded successfully from {path}")
    return K, B, C, enc
