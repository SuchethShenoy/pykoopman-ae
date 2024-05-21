import torch


def get_koopman_model(model):
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

    # Get K matrix (Koopman Dynamics)
    K = model.k_block[len(model.k_block) - 1].weight.cpu().detach()
    for i in range(len(model.k_block) - 1, 0, -1):
        K = torch.matmul(K, model.k_block[i - 1].weight.cpu().detach())

    # Get B matrix (Input Matrix)
    B = model.b_block[len(model.b_block) - 1].weight.cpu().detach()
    for i in range(len(model.b_block) - 1, 0, -1):
        B = torch.matmul(B, model.b_block[i - 1].weight.cpu().detach())

    # Get C matrix (Output Matrix)
    if model.decoder_trainable:
        C = model.decoder[len(model.decoder) - 1].weight.cpu().detach()
        for i in range(len(model.decoder) - 1, 0, -1):
            C = torch.matmul(C, model.decoder[i - 1].weight.cpu().detach())
    else:
        C = model.c_block.cpu()

    # Get Encoder as a function (Lifting Function)
    def enc(x):
        """
        Computes the lifted states from the original states.

        Args:
            x (torch.Tensor): The input tensor with shape (batch_size, num_original_states).

        Returns:
            torch.Tensor: The lifted states with shape (batch_size, num_lifted_states).
        """

        if model.model_type == "MLP":
            encoded = model.encoder(x)
            lifted_states = torch.concat([x, encoded], axis=1)
            return lifted_states

        if model.model_type == "TCN":
            if x.shape[-1] != model.time_window:
                x = x.transpose(-2, -1)
            tcn_output = model.tcn(x)
            encoded = model.encoder(tcn_output)
            lifted_states = torch.concat([x[:, :, -1], encoded], axis=1)
            return lifted_states

        if model.model_type == "LSTM":
            lstm_output, _status = model.lstm(x)
            lstm_output = lstm_output[:, -1, :]
            encoded = model.encoder(lstm_output)
            lifted_states = torch.concat([x[:, -1, :], encoded], axis=1)
            return lifted_states

        if model.model_type == "GRU":
            gru_output, _status = model.gru(x)
            gru_output = gru_output[:, -1, :]
            encoded = model.encoder(gru_output)
            lifted_states = torch.concat([x[:, -1, :], encoded], axis=1)
            return lifted_states

    return K, B, C, enc
