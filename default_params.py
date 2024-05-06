default_params = {

    # Necessary parameters
    'num_original_states': None,
    'num_lifted_states': None,
    'num_inputs': None,

    # Common Parameters
    'encoder_layers': [16, 64, 16],
    'k_block_layers': [32],
    'b_block_layers': [32],
    'decoder_trainable': False,
    'decoder_layers': [16, 64, 128, 64, 16]
}
