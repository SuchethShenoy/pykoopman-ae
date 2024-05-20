def params_test(model):

    # Necessary Parameters Tests
    assert (
        model.num_original_states != None
    ), f"num_original_states: Number of original states needs to be specified."

    assert (
        model.num_lifted_states != None
    ), f"num_lifted_states: Number of lifted states needs to be specified."

    assert (
        model.num_inputs != None
    ), f"num_inputs: Number of inputs needs to be specified."

    assert (
        model.num_lifted_states > model.num_original_states
    ), f"Number of lifted states must be greater than the number of original states."

    # TCN Parameters Tests
    if model.model_type == "TCN":
        assert len(model.tcn_channels) == len(
            model.tcn_kernels
        ), f"Number of TCN layer channels should be equal to the number of TCN layer kernels."
