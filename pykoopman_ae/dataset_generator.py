import torch


def get_temporal_dataset(trajectory, input, time_window):
    """
    Generates temporal datasets for training time-series models by creating sequences of a specified
    time window length from the input trajectory data. Optionally includes input data if provided.

    Args:
        trajectory (torch.Tensor): The trajectory data with shape (batch_size, num_features, sequence_length).
        input (torch.Tensor or None): The input data with shape (batch_size, sequence_length, num_inputs),
                                      or None if no input data is provided.
        time_window (int): The length of the time window for creating sequences.

    Returns:
        tuple: A tuple containing the following elements:
            - X (torch.Tensor): The input sequences with shape
                                (batch_size, sequence_length-1, time_window, num_features).
            - Y (torch.Tensor): The target values with shape
                                (batch_size, sequence_length-1, num_features).
            - U (torch.Tensor or None): The input sequences corresponding to the input data,
                                        with shape (batch_size, sequence_length-1, num_inputs),
                                        or None if no input data is provided.
    """

    X = torch.zeros(
        (trajectory.shape[0], trajectory.shape[2] - 1, time_window, trajectory.shape[1])
    )
    Y = torch.zeros((trajectory.shape[0], trajectory.shape[2] - 1, trajectory.shape[1]))

    for i, traj in enumerate(trajectory):
        traj = traj.T
        for j in range(traj.shape[0] - 1):
            temp = torch.zeros((time_window, trajectory.shape[1]))
            if j < (time_window):
                temp[0 : time_window - j] = traj[0]
                temp[time_window - j :] = traj[1 : j + 1, :]
                X[i, j] = temp
                Y[i, j] = traj[j + 1]
            else:
                X[i, j] = traj[j - time_window + 1 : j + 1, :]
                Y[i, j] = traj[j + 1, :]

    if input != None:
        U = input.transpose(1, 2)[:, :-1, :]
    else:
        U = None

    return X, Y, U
