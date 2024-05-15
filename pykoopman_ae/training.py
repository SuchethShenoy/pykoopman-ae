import torch
from tqdm import tqdm

from pykoopman_ae.system_extraction import get_koopman_system

class Trainer():

    def __init__(self, 
                model,
                trajectory,
                input=None,
                loss_function=None,
                optimizer=None,
                dynamic_loss_window=10,
                num_epochs=10,
                batch_size=256):
        
        self.model = model
        self.trajectory = trajectory
        self.input = input
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.dynamic_loss_window = dynamic_loss_window
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = next(self.model.parameters()).device

        if self.loss_function == None:
            self.loss_function = torch.nn.MSELoss()

        if self.optimizer == None:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), 
                                    lr = 0.0001, weight_decay=1e-6)
        
        
    def learn_koopman_model(self):

        if self.model.model_type == "MLP":
            loss = train_MLP_with_control_input(self)
            return loss

    
    def learn_koopman_eigendynamics(self):

        if self.model.model_type == "MLP":
            loss = train_MLP_without_control_input(self)
            return loss


    def learn_input_matrix(self):

        if self.model.model_type == "MLP":
            loss = train_b_block_with_control_input(self)
            return loss




def train_MLP_with_control_input(trainer):
    """
        Trains the MLP_AE model.

        Parameters:
            trajectory (torch.Tensor): The input trajectories 
				with shape (num_trajectories, num_features, length_trajectory).
            input (torch.Tensor): The control inputs corresponding to the trajectories 
				with shape (num_trajectories, num_inputs, length_trajectory).
            loss_function (callable): The loss function to use for training.
            optimizer (torch.optim.Optimizer): The optimizer to use for training.
            dynamic_loss_window (int): The window size for calculating the dynamic loss.
            num_epochs (int): The number of epochs to train the model.
            batch_size (int): The size of the batches for training.

        Returns:
            torch.Tensor: A tensor containing the training losses for each epoch.
        """

    loss_epoch_mean = []

    for epoch in range(trainer.num_epochs):

        for i in tqdm(range(trainer.trajectory.shape[0])):

            X_traj = trainer.trajectory[i, :, :-1].T
            Y_traj = trainer.trajectory[i, :, 1:].T
            U_traj = trainer.input[i, :, :-1].T
            
            losses = []
            for j in range(0, len(X_traj)-trainer.dynamic_loss_window, trainer.batch_size):

                X_batch = X_traj[j:j+trainer.batch_size]
                encoded_k = trainer.model.encoder(X_batch)
                encoded_k = torch.concat([X_batch, encoded_k], axis=1)

                if trainer.model.decoder_trainable:
                    decoded_k = trainer.model.decoder(encoded_k)
                else:
                    decoded_k = torch.matmul(trainer.model.c_block, encoded_k.T).T

                U_batch = U_traj[j:j+trainer.batch_size]
                inp_k = trainer.model.b_block(U_batch)

                Y_pred = trainer.model(X_batch, U_batch)
                Y_batch = Y_traj[j:j+trainer.batch_size]
                
                for k in range(trainer.dynamic_loss_window-1):
                    time_shifted_m_steps = encoded_k[:-k-1]
                    u_input_m_steps = inp_k[:-k-1]
                    for l in range(k+1):
                        time_shifted_m_steps = trainer.model.k_block(time_shifted_m_steps) + u_input_m_steps
                    encoded_m_steps = trainer.model.encoder(X_batch[k+1:])
                    encoded_m_steps = torch.concat([X_batch[k+1:], encoded_m_steps], axis=1)
                    if k==0:
                        loss_dynamics = trainer.loss_function(time_shifted_m_steps, encoded_m_steps)
                    else:
                        loss_dynamics += trainer.loss_function(time_shifted_m_steps, encoded_m_steps)
                loss_dynamics = loss_dynamics/(trainer.dynamic_loss_window-1)
                
                loss_reconstruction = trainer.loss_function(decoded_k, X_batch)
                loss_prediction = trainer.loss_function(Y_pred, Y_batch)
                loss = 10*loss_reconstruction + loss_prediction + loss_dynamics
                
                trainer.optimizer.zero_grad()
                loss.backward()
                trainer.optimizer.step()
                losses.append(loss.item())
        
        mean_epoch_loss = torch.mean(torch.tensor(losses)).item()
        loss_epoch_mean.append(mean_epoch_loss)
    
        print(f'Finished epoch {epoch+1}, mean loss for the epoch = {mean_epoch_loss}')

    return torch.tensor(loss_epoch_mean)


def train_MLP_without_control_input(trainer):

    loss_epoch_mean = []

    for epoch in range(trainer.num_epochs):

        for i in tqdm(range(trainer.trajectory.shape[0])):

            X_traj = trainer.trajectory[i, :, :-1].T
            Y_traj = trainer.trajectory[i, :, 1:].T

            losses = []
            for j in range(0, len(X_traj)-trainer.dynamic_loss_window, trainer.batch_size):

                X_batch = X_traj[j:j+trainer.batch_size]
                encoded_k = trainer.model.encoder(X_batch)
                encoded_k = torch.concat([X_batch, encoded_k], axis=1)
                
                if trainer.model.decoder_trainable:
                    decoded_k = trainer.model.decoder(encoded_k)
                else:
                    decoded_k = torch.matmul(trainer.model.c_block, encoded_k.T).T

                U_traj = torch.zeros((X_batch.shape[0], trainer.model.num_inputs)).to(trainer.device)

                Y_pred = trainer.model(X_batch, U_traj)
                Y_batch = Y_traj[j:j+trainer.batch_size]
                
                for k in range(trainer.dynamic_loss_window-1):
                    time_shifted_m_steps = encoded_k[:-k-1]
                    for l in range(k+1):
                        time_shifted_m_steps = trainer.model.k_block(time_shifted_m_steps)
                    encoded_m_steps = trainer.model.encoder(X_batch[k+1:])
                    encoded_m_steps = torch.concat([X_batch[k+1:], encoded_m_steps], axis=1)
                    if k==0:
                        loss_dynamics = trainer.loss_function(time_shifted_m_steps, encoded_m_steps)
                    else:
                        loss_dynamics += trainer.loss_function(time_shifted_m_steps, encoded_m_steps)
                loss_dynamics = loss_dynamics/(trainer.dynamic_loss_window-1)

                loss_reconstruction = trainer.loss_function(decoded_k, X_batch)
                loss_prediction = trainer.loss_function(Y_pred, Y_batch)
                loss = 10*loss_reconstruction + loss_prediction + loss_dynamics
                
                trainer.optimizer.zero_grad()
                loss.backward()
                trainer.optimizer.step()
                losses.append(loss.item())
    
        mean_epoch_loss = torch.mean(torch.tensor(losses)).item()
        loss_epoch_mean.append(mean_epoch_loss)
    
        print(f'Finished epoch {epoch+1}, mean loss for the epoch = {mean_epoch_loss}')

    return torch.tensor(loss_epoch_mean)


def train_b_block_with_control_input(trainer):

    K, B, C, enc = get_koopman_system(trainer.model)
    K = K.to(device=trainer.device)
    B = B.to(device=trainer.device)
    C = C.to(device=trainer.device)

    loss_epoch_mean = []

    for epoch in range(trainer.num_epochs):
        
        for i in range(trainer.trajectory.shape[0]):
            X_traj = trainer.trajectory[i, :, :-1].T
            Y_traj = trainer.trajectory[i, :, 1:].T
            U_traj = trainer.input[i, :, :-1].T

            losses = []
            for j in range(0, len(X_traj)-trainer.dynamic_loss_window, trainer.batch_size):
                
                X_batch = X_traj[j:j+trainer.batch_size]
                U_batch = U_traj[j:j+trainer.batch_size]
                encoded_k = enc(X_batch)
                z_k_plus_one = torch.matmul(K, encoded_k.T).T + trainer.model.b_block(U_batch)
                Y_pred = torch.matmul(C, z_k_plus_one.T).T
                Y_batch = Y_traj[j:j+trainer.batch_size]
                
                for k in range(trainer.dynamic_loss_window-1):

                    time_shifted_m_steps = encoded_k[:-k-1]
                    for l in range(k+1):
                        time_shifted_m_steps = torch.matmul(K, time_shifted_m_steps.T).T + trainer.model.b_block(U_batch[k+1:])

                    encoded_m_steps = enc(X_batch[k+1:])
                    if k==0:
                        loss_dynamics = trainer.loss_function(torch.matmul(C, time_shifted_m_steps.T), torch.matmul(C, encoded_m_steps.T))
                    else:
                        loss_dynamics += trainer.loss_function(torch.matmul(C, time_shifted_m_steps.T), torch.matmul(C, encoded_m_steps.T))

                loss_dynamics = loss_dynamics/(trainer.dynamic_loss_window-1)
                loss_prediction = trainer.loss_function(Y_pred, Y_batch)
                loss = loss_prediction + loss_dynamics

                trainer.optimizer.zero_grad()
                loss.backward()
                trainer.optimizer.step()
                losses.append(loss.item())

        mean_epoch_loss = torch.mean(torch.tensor(losses)).item()
        loss_epoch_mean.append(mean_epoch_loss)
    
        print(f'Finished epoch {epoch+1}, mean loss for the epoch = {mean_epoch_loss}')

    return torch.tensor(loss_epoch_mean)
