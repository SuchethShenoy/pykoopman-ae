default_params_mlp = {
    # Necessary parameters
    "num_original_states": None,
    "num_lifted_states": None,
    "num_inputs": None,
    # MLP Parameters
    "encoder_layers": [16, 64, 16],
    "k_block_layers": [32],
    "b_block_layers": [32],
    "decoder_trainable": False,
    "decoder_layers": [16, 64, 128, 64, 16],
}

default_params_tcn = {
    # Necessary parameters
    "num_original_states": None,
    "num_lifted_states": None,
    "num_inputs": None,
    # MLP Parameters
    "encoder_layers": [16, 64, 16],
    "k_block_layers": [32],
    "b_block_layers": [32],
    "decoder_trainable": False,
    "decoder_layers": [16, 64, 128, 64, 16],
    # TCN Parameters
    "time_window": 5,
    "tcn_channels": [10, 20],
    "tcn_kernels": [3, 2],
}

default_params_lstm = {
    # Necessary parameters
    "num_original_states": None,
    "num_lifted_states": None,
    "num_inputs": None,
    # MLP Parameters
    "encoder_layers": [16, 64, 16],
    "k_block_layers": [32],
    "b_block_layers": [32],
    "decoder_trainable": False,
    "decoder_layers": [16, 64, 128, 64, 16],
    # LSTM Parameters
    "time_window": 5,
    "lstm_hidden_size": 10,
    "lstm_num_layers": 10,
}

default_params_gru = {
    # Necessary parameters
    "num_original_states": None,
    "num_lifted_states": None,
    "num_inputs": None,
    # MLP Parameters
    "encoder_layers": [16, 64, 16],
    "k_block_layers": [32],
    "b_block_layers": [32],
    "decoder_trainable": False,
    "decoder_layers": [16, 64, 128, 64, 16],
    # GRU Parameters
    "time_window": 5,
    "gru_hidden_size": 10,
    "gru_num_layers": 10,
}
