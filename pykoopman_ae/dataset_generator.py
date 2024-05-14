import torch

def get_temporal_dataset(trajectory, input, time_window):

    X = torch.zeros((trajectory.shape[0], 
						trajectory.shape[2]-1, 
						time_window, 
						trajectory.shape[1]))
    Y = torch.zeros((trajectory.shape[0], 
                    trajectory.shape[2]-1, 
                    trajectory.shape[1]))

    for i, traj in enumerate(trajectory):
        traj = traj.T
        for j in range(traj.shape[0]-1):
            temp = torch.zeros((time_window, trajectory.shape[1]))
            if j<(time_window):
                temp[0:time_window-j] = traj[0]
                temp[time_window-j:] = traj[1:j+1,:]
                X[i,j] = temp
                Y[i,j] = traj[j+1]
            else:
                X[i, j] = traj[j-time_window+1:j+1, :]
                Y[i, j] = traj[j+1, :]

    U = input.transpose(1,2)[:, :-1, :]

    return X, Y, U