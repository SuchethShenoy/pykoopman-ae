import torch

from pykoopman_ae.system_extraction import get_koopman_model


def save_koopman_model(model, params, path):
    """
    Saves the Koopman system matrices (K, B, C) and the encoder parameters.

    Args:
        model (torch.nn.Module): The trained model.
        path (str): The file path to save the Koopman system.
    """
    # Extract Koopman matrices and encoder function
    K, B, C, enc = get_koopman_model(model)

    # Save the matrices and encoder state dict
    torch.save(
        {
            "K": K,
            "B": B,
            "C": C,
            "model_state_dict": model.state_dict(),
            "encoder_state_dict": model.encoder.state_dict(),
            "model_params": params,
            "model_type": model.model_type,
            "time_window": model.time_window if hasattr(model, "time_window") else None,
        },
        path,
    )

    print(f"Koopman system saved successfully at {path}")


def load_koopman_model(path, model_class):
    """
    Loads the Koopman system matrices (K, B, C) and the encoder parameters.

    Args:
        path (str): The file path to load the Koopman system from.
        model_class (class): The class of the model to instantiate.

    Returns:
        tuple: A tuple containing the Koopman matrices (K, B, C), encoder function (enc) and the model.
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
    model.load_state_dict(checkpoint["model_state_dict"])

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
    return K, B, C, enc, model
