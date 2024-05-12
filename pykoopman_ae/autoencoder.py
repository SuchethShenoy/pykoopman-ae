import torch
from tqdm import tqdm

from pykoopman_ae.default_params import *
from pykoopman_ae.params_test import params_test

class MLP_AE(torch.nn.Module):
	
	def __init__(self, params):

		for key, value in default_params_mlp.items():
			if key not in params:
				params[key] = value

		self.model_type = 'MLP'

		self.num_original_states = params['num_original_states']
		self.num_lifted_states = params['num_lifted_states']
		self.num_inputs = params['num_inputs']

		self.encoder_layers = params['encoder_layers']
		self.k_block_layers = params['k_block_layers']
		self.b_block_layers = params['b_block_layers']
		self.decoder_trainable = params['decoder_trainable']
		self.decoder_layers = params['decoder_layers']

		params_test(self)

		super().__init__()
		
		# Encoder Block
		encoder_layers = [torch.nn.Linear(self.num_original_states, 
										self.encoder_layers[0], 
										bias=False), 
										torch.nn.Tanh()]								
		for i in range(len(self.encoder_layers) - 1):
			encoder_layers.extend([
				torch.nn.Linear(self.encoder_layers[i], 
								self.encoder_layers[i + 1], 
								bias=False),
				torch.nn.Tanh()
			])
		encoder_layers.extend([
			torch.nn.Linear(self.encoder_layers[-1], 
							self.num_lifted_states - self.num_original_states, 
							bias=False),
			torch.nn.Tanh()
			])
		(encoder_layers)
		self.encoder = torch.nn.Sequential(*encoder_layers)

		# K Block
		k_block_layers = [torch.nn.Linear(self.num_lifted_states, 
										self.k_block_layers[0], 
										bias=False)]								
		for i in range(len(self.k_block_layers) - 1):
			k_block_layers.extend([
				torch.nn.Linear(self.k_block_layers[i], 
								self.k_block_layers[i + 1], 
								bias=False)
			])
		k_block_layers.extend([
			torch.nn.Linear(self.k_block_layers[-1], 
							self.num_lifted_states, 
							bias=False),
			])
		(k_block_layers)
		self.k_block = torch.nn.Sequential(*k_block_layers)

		# B Block
		b_block_layers = [torch.nn.Linear(self.num_inputs, 
										self.b_block_layers[0], 
										bias=False)]								
		for i in range(len(self.b_block_layers) - 1):
			b_block_layers.extend([
				torch.nn.Linear(self.b_block_layers[i], 
								self.b_block_layers[i + 1], 
								bias=False)
			])
		b_block_layers.extend([
			torch.nn.Linear(self.b_block_layers[-1], 
							self.num_lifted_states, 
							bias=False),
			])
		(b_block_layers)
		self.b_block = torch.nn.Sequential(*b_block_layers)

		if self.decoder_trainable:
			# Decoder Block
			decoder_layers = [torch.nn.Linear(self.num_lifted_states, 
											self.decoder_layers[0], 
											bias=False)]								
			for i in range(len(self.decoder_layers) - 1):
				decoder_layers.extend([
					torch.nn.Linear(self.decoder_layers[i], 
									self.decoder_layers[i + 1], 
									bias=False)
				])
			decoder_layers.extend([
				torch.nn.Linear(self.decoder_layers[-1], 
								self.num_original_states, 
								bias=False),
				])
			(decoder_layers)
			self.decoder = torch.nn.Sequential(*decoder_layers)
		
		else:		
			# C Block
			self.c_block = torch.nn.Parameter(torch.eye(self.num_original_states, 
														self.num_lifted_states))
			self.c_block.requires_grad = False


	def forward(self, x, u):
		
		encoded = self.encoder(x)
		encoded = torch.concat([x, encoded], axis=1)
		
		inp = self.b_block(u)		
		time_shifted = self.k_block(encoded) + inp
		
		if self.decoder_trainable:
			decoded = self.decoder(time_shifted)
		else:
			decoded = torch.matmul(self.c_block, time_shifted.T).T
		
		return decoded

	def train(self, 
			trajectory, 
			input,
			loss_function,
			optimizer,
			dynamic_loss_window=10,
			num_epochs=10,
			batch_size=256):

		losses = []
		for epoch in range(num_epochs):

			for i in tqdm(range(trajectory.shape[0])):

				X_traj = trajectory[i, :, :-1].T
				Y_traj = trajectory[i, :, 1:].T
				U_traj = input[i, :, :-1].T

				for j in range(0, len(X_traj)-dynamic_loss_window, batch_size):

					X_batch = X_traj[j:j+batch_size]
					encoded_k = self.encoder(X_batch)
					encoded_k = torch.concat([X_batch, encoded_k], axis=1)

					if self.decoder_trainable:
						decoded_k = self.decoder(encoded_k)
					else:
						decoded_k = torch.matmul(self.c_block, encoded_k.T).T

					U_batch = U_traj[j:j+batch_size]
					inp_k = self.b_block(U_batch)

					Y_pred = self(X_batch, U_batch)
					Y_batch = Y_traj[j:j+batch_size]
					
					for k in range(dynamic_loss_window-1):
						time_shifted_m_steps = encoded_k[:-k-1]
						u_input_m_steps = inp_k[:-k-1]
						for l in range(k+1):
							time_shifted_m_steps = self.k_block(time_shifted_m_steps) + u_input_m_steps
						encoded_m_steps = self.encoder(X_batch[k+1:])
						encoded_m_steps = torch.concat([X_batch[k+1:], encoded_m_steps], axis=1)
						if k==0:
							loss_dynamics = loss_function(time_shifted_m_steps, encoded_m_steps)
						else:
							loss_dynamics += loss_function(time_shifted_m_steps, encoded_m_steps)
					loss_dynamics = loss_dynamics/(dynamic_loss_window-1)
					
					loss_reconstruction = loss_function(decoded_k, X_batch)
					loss_prediction = loss_function(Y_pred, Y_batch)
					loss = 10*loss_reconstruction + loss_prediction + loss_dynamics
					
					optimizer.zero_grad()
					loss.backward()
					optimizer.step()
					losses.append(loss.item())
		
			print(f'Finished epoch {epoch+1}, latest loss {loss}')

		return torch.tensor(losses)


class TCN_AE(torch.nn.Module):

	def __init__(self, params):

		for key, value in default_params_tcn.items():
			if key not in params:
				params[key] = value

		self.model_type = 'TCN'

		self.num_original_states = params['num_original_states']
		self.num_lifted_states = params['num_lifted_states']
		self.num_inputs = params['num_inputs']

		self.encoder_layers = params['encoder_layers']
		self.k_block_layers = params['k_block_layers']
		self.b_block_layers = params['b_block_layers']
		self.decoder_trainable = params['decoder_trainable']
		self.decoder_layers = params['decoder_layers']

		self.time_window = params['time_window']
		self.tcn_channels = params['tcn_channels']
		self.tcn_kernels = params['tcn_kernels']

		params_test(self)

		super().__init__()
		
		# TCN Block
		tcn_layers = [torch.nn.Conv1d(in_channels=self.num_original_states, 
									out_channels=self.tcn_channels[0], 
									kernel_size=(self.tcn_kernels[0]), 
									stride=1, padding=0, dilation=1, groups=1, 
									bias=False),
									torch.nn.Tanh()]								
		for i in range(len(self.tcn_channels) - 1):
			tcn_layers.extend([torch.nn.Conv1d(in_channels=self.tcn_channels[i], 
									out_channels=self.tcn_channels[i+1], 
									kernel_size=(self.tcn_kernels[i+1]), 
									stride=1, padding=0, dilation=1, groups=1, 
									bias=False),
									torch.nn.Tanh()])
		tcn_layers.extend([torch.nn.Flatten()])
		(tcn_layers)
		self.tcn = torch.nn.Sequential(*tcn_layers)
		
		# Encoder Block
		encoder_layers = [torch.nn.Linear(self.tcn_channels[-1]*self.tcn_kernels[-1], 
										self.encoder_layers[0], 
										bias=False), 
										torch.nn.Tanh()]								
		for i in range(len(self.encoder_layers) - 1):
			encoder_layers.extend([
				torch.nn.Linear(self.encoder_layers[i], 
								self.encoder_layers[i + 1], 
								bias=False),
				torch.nn.Tanh()
			])
		encoder_layers.extend([
			torch.nn.Linear(self.encoder_layers[-1], 
							self.num_lifted_states - self.num_original_states, 
							bias=False),
			torch.nn.Tanh()
			])
		(encoder_layers)
		self.encoder = torch.nn.Sequential(*encoder_layers)

		# K Block
		k_block_layers = [torch.nn.Linear(self.num_lifted_states, 
										self.k_block_layers[0], 
										bias=False)]								
		for i in range(len(self.k_block_layers) - 1):
			k_block_layers.extend([
				torch.nn.Linear(self.k_block_layers[i], 
								self.k_block_layers[i + 1], 
								bias=False)
			])
		k_block_layers.extend([
			torch.nn.Linear(self.k_block_layers[-1], 
							self.num_lifted_states, 
							bias=False),
			])
		(k_block_layers)
		self.k_block = torch.nn.Sequential(*k_block_layers)

		# B Block
		b_block_layers = [torch.nn.Linear(self.num_inputs, 
										self.b_block_layers[0], 
										bias=False)]								
		for i in range(len(self.b_block_layers) - 1):
			b_block_layers.extend([
				torch.nn.Linear(self.b_block_layers[i], 
								self.b_block_layers[i + 1], 
								bias=False)
			])
		b_block_layers.extend([
			torch.nn.Linear(self.b_block_layers[-1], 
							self.num_lifted_states, 
							bias=False),
			])
		(b_block_layers)
		self.b_block = torch.nn.Sequential(*b_block_layers)

		if self.decoder_trainable:
			# Decoder Block
			decoder_layers = [torch.nn.Linear(self.num_lifted_states, 
											self.decoder_layers[0], 
											bias=False)]								
			for i in range(len(self.decoder_layers) - 1):
				decoder_layers.extend([
					torch.nn.Linear(self.decoder_layers[i], 
									self.decoder_layers[i + 1], 
									bias=False)
				])
			decoder_layers.extend([
				torch.nn.Linear(self.decoder_layers[-1], 
								self.num_original_states, 
								bias=False),
				])
			(decoder_layers)
			self.decoder = torch.nn.Sequential(*decoder_layers)
		
		else:		
			# C Block
			self.c_block = torch.nn.Parameter(torch.eye(self.num_original_states, 
														self.num_lifted_states))
			self.c_block.requires_grad = False


	def forward(self, x, u):

		tcn_output = self.tcn(x)
		
		encoded = self.encoder(tcn_output)
		encoded = torch.concat([x[:,:,-1], encoded], axis=1)
		
		inp = self.b_block(u)		
		time_shifted = self.k_block(encoded) + inp
		
		if self.decoder_trainable:
			decoded = self.decoder(time_shifted)
		else:
			decoded = torch.matmul(self.c_block, time_shifted.T).T
		
		return decoded


class LSTM_AE(torch.nn.Module):

	def __init__(self, params):

		for key, value in default_params_lstm.items():
			if key not in params:
				params[key] = value

		self.model_type = 'LSTM'

		self.num_original_states = params['num_original_states']
		self.num_lifted_states = params['num_lifted_states']
		self.num_inputs = params['num_inputs']

		self.encoder_layers = params['encoder_layers']
		self.k_block_layers = params['k_block_layers']
		self.b_block_layers = params['b_block_layers']
		self.decoder_trainable = params['decoder_trainable']
		self.decoder_layers = params['decoder_layers']

		self.time_window = params['time_window']
		self.lstm_hidden_size = params['lstm_hidden_size']
		self.lstm_num_layers = params['lstm_num_layers']

		params_test(self)

		super().__init__()

		# LSTM Block
		self.lstm = torch.nn.LSTM(input_size=self.num_original_states, 
								hidden_size=self.lstm_hidden_size, 
								num_layers=self.lstm_num_layers, 
								batch_first=True)
		
		# Encoder Block
		encoder_layers = [torch.nn.Linear(self.lstm_hidden_size, 
										self.encoder_layers[0], 
										bias=False), 
										torch.nn.Tanh()]								
		for i in range(len(self.encoder_layers) - 1):
			encoder_layers.extend([
				torch.nn.Linear(self.encoder_layers[i], 
								self.encoder_layers[i + 1], 
								bias=False),
				torch.nn.Tanh()
			])
		encoder_layers.extend([
			torch.nn.Linear(self.encoder_layers[-1], 
							self.num_lifted_states - self.num_original_states, 
							bias=False),
			torch.nn.Tanh()
			])
		(encoder_layers)
		self.encoder = torch.nn.Sequential(*encoder_layers)

		# K Block
		k_block_layers = [torch.nn.Linear(self.num_lifted_states, 
										self.k_block_layers[0], 
										bias=False)]								
		for i in range(len(self.k_block_layers) - 1):
			k_block_layers.extend([
				torch.nn.Linear(self.k_block_layers[i], 
								self.k_block_layers[i + 1], 
								bias=False)
			])
		k_block_layers.extend([
			torch.nn.Linear(self.k_block_layers[-1], 
							self.num_lifted_states, 
							bias=False),
			])
		(k_block_layers)
		self.k_block = torch.nn.Sequential(*k_block_layers)

		# B Block
		b_block_layers = [torch.nn.Linear(self.num_inputs, 
										self.b_block_layers[0], 
										bias=False)]								
		for i in range(len(self.b_block_layers) - 1):
			b_block_layers.extend([
				torch.nn.Linear(self.b_block_layers[i], 
								self.b_block_layers[i + 1], 
								bias=False)
			])
		b_block_layers.extend([
			torch.nn.Linear(self.b_block_layers[-1], 
							self.num_lifted_states, 
							bias=False),
			])
		(b_block_layers)
		self.b_block = torch.nn.Sequential(*b_block_layers)

		if self.decoder_trainable:
			# Decoder Block
			decoder_layers = [torch.nn.Linear(self.num_lifted_states, 
											self.decoder_layers[0], 
											bias=False)]								
			for i in range(len(self.decoder_layers) - 1):
				decoder_layers.extend([
					torch.nn.Linear(self.decoder_layers[i], 
									self.decoder_layers[i + 1], 
									bias=False)
				])
			decoder_layers.extend([
				torch.nn.Linear(self.decoder_layers[-1], 
								self.num_original_states, 
								bias=False),
				])
			(decoder_layers)
			self.decoder = torch.nn.Sequential(*decoder_layers)
		
		else:		
			# C Block
			self.c_block = torch.nn.Parameter(torch.eye(self.num_original_states, 
														self.num_lifted_states))
			self.c_block.requires_grad = False


	def forward(self, x, u):

		lstm_output, _status = self.lstm(x)
		lstm_output = lstm_output[:,-1,:]
		
		encoded = self.encoder(lstm_output)
		encoded = torch.concat([x[:,-1,:], encoded], axis=1)
		
		inp = self.b_block(u)		
		time_shifted = self.k_block(encoded) + inp
		
		if self.decoder_trainable:
			decoded = self.decoder(time_shifted)
		else:
			decoded = torch.matmul(self.c_block, time_shifted.T).T
		
		return decoded


class GRU_AE(torch.nn.Module):

	def __init__(self, params):

		for key, value in default_params_gru.items():
			if key not in params:
				params[key] = value

		self.model_type = 'GRU'

		self.num_original_states = params['num_original_states']
		self.num_lifted_states = params['num_lifted_states']
		self.num_inputs = params['num_inputs']

		self.encoder_layers = params['encoder_layers']
		self.k_block_layers = params['k_block_layers']
		self.b_block_layers = params['b_block_layers']
		self.decoder_trainable = params['decoder_trainable']
		self.decoder_layers = params['decoder_layers']

		self.time_window = params['time_window']
		self.gru_hidden_size = params['gru_hidden_size']
		self.gru_num_layers = params['gru_num_layers']

		params_test(self)

		super().__init__()

		# GRU Block
		self.gru = torch.nn.GRU(input_size=self.num_original_states, 
								hidden_size=self.gru_hidden_size, 
								num_layers=self.gru_num_layers, 
								batch_first=True)
		
		# Encoder Block
		encoder_layers = [torch.nn.Linear(self.gru_hidden_size, 
										self.encoder_layers[0], 
										bias=False), 
										torch.nn.Tanh()]								
		for i in range(len(self.encoder_layers) - 1):
			encoder_layers.extend([
				torch.nn.Linear(self.encoder_layers[i], 
								self.encoder_layers[i + 1], 
								bias=False),
				torch.nn.Tanh()
			])
		encoder_layers.extend([
			torch.nn.Linear(self.encoder_layers[-1], 
							self.num_lifted_states - self.num_original_states, 
							bias=False),
			torch.nn.Tanh()
			])
		(encoder_layers)
		self.encoder = torch.nn.Sequential(*encoder_layers)

		# K Block
		k_block_layers = [torch.nn.Linear(self.num_lifted_states, 
										self.k_block_layers[0], 
										bias=False)]								
		for i in range(len(self.k_block_layers) - 1):
			k_block_layers.extend([
				torch.nn.Linear(self.k_block_layers[i], 
								self.k_block_layers[i + 1], 
								bias=False)
			])
		k_block_layers.extend([
			torch.nn.Linear(self.k_block_layers[-1], 
							self.num_lifted_states, 
							bias=False),
			])
		(k_block_layers)
		self.k_block = torch.nn.Sequential(*k_block_layers)

		# B Block
		b_block_layers = [torch.nn.Linear(self.num_inputs, 
										self.b_block_layers[0], 
										bias=False)]								
		for i in range(len(self.b_block_layers) - 1):
			b_block_layers.extend([
				torch.nn.Linear(self.b_block_layers[i], 
								self.b_block_layers[i + 1], 
								bias=False)
			])
		b_block_layers.extend([
			torch.nn.Linear(self.b_block_layers[-1], 
							self.num_lifted_states, 
							bias=False),
			])
		(b_block_layers)
		self.b_block = torch.nn.Sequential(*b_block_layers)

		if self.decoder_trainable:
			# Decoder Block
			decoder_layers = [torch.nn.Linear(self.num_lifted_states, 
											self.decoder_layers[0], 
											bias=False)]								
			for i in range(len(self.decoder_layers) - 1):
				decoder_layers.extend([
					torch.nn.Linear(self.decoder_layers[i], 
									self.decoder_layers[i + 1], 
									bias=False)
				])
			decoder_layers.extend([
				torch.nn.Linear(self.decoder_layers[-1], 
								self.num_original_states, 
								bias=False),
				])
			(decoder_layers)
			self.decoder = torch.nn.Sequential(*decoder_layers)
		
		else:		
			# C Block
			self.c_block = torch.nn.Parameter(torch.eye(self.num_original_states, 
														self.num_lifted_states))
			self.c_block.requires_grad = False


	def forward(self, x, u):

		gru_output, _status = self.gru(x)
		gru_output = gru_output[:,-1,:]
		
		encoded = self.encoder(gru_output)
		encoded = torch.concat([x[:,-1,:], encoded], axis=1)
		
		inp = self.b_block(u)		
		time_shifted = self.k_block(encoded) + inp
		
		if self.decoder_trainable:
			decoded = self.decoder(time_shifted)
		else:
			decoded = torch.matmul(self.c_block, time_shifted.T).T
		
		return decoded