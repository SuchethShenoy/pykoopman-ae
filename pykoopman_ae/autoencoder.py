import torch

from default_params import default_params

class MLP_AE(torch.nn.Module):
	
	def __init__(self, params):

		for key, value in default_params.items():
			if key not in params:
				params[key] = value

		self.num_original_states = params['num_original_states']
		self.num_lifted_states = params['num_lifted_states']
		self.num_inputs = params['num_inputs']
		
		assert self.num_original_states != None, \
			f"num_original_states: Number of original states needs to be specified"
		
		assert self.num_lifted_states != None, \
			f"num_lifted_states: Number of lifted states needs to be specified"

		assert self.num_inputs != None, \
			f"num_inputs: Number of inputs needs to be specified"

		assert self.num_lifted_states > self.num_original_states, \
			f"Number of lifted states must be greater than the number of original states"

		self.encoder_layers = params['encoder_layers']
		self.k_block_layers = params['k_block_layers']
		self.b_block_layers = params['b_block_layers']
		self.decoder_trainable = params['decoder_trainable']
		self.decoder_layers = params['decoder_layers']

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