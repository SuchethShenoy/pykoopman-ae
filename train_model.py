import torch
import numpy as np
import os
from pykoopman_ae import *
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--dataset", choices=["duffing", "vanderpol"], required=True)
parser.add_argument(
    "-m", "--model_type", choices=["MLP", "TCN", "GRU", "LSTM"], required=True
)
parser.add_argument(
    "-ls", "--lifted_states", type=int, choices=[4, 8, 16], required=True
)
parser.add_argument(
    "-tw", "--time_window", type=int, choices=[0, 4, 8, 16], required=True
)
parser.add_argument("-e", "--epochs", type=int, required=False)

args = parser.parse_args()

DATASET = args.dataset  # duffing or vanderpol
MODEL_TYPE = args.model_type  # MLP or TCN or LSTM or GRU
LIFTED_STATES = args.lifted_states  # 4 or 8 or 16
TIME_WINDOW = args.time_window  # 4 or 8 or 16 (0 for MLP)
EPOCHS = args.epochs

if EPOCHS == None:
    EPOCHS = 30

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

params = {
    # Parameters to vary
    "num_lifted_states": LIFTED_STATES,
    "time_window": TIME_WINDOW,
    # Necessary parameters
    "num_original_states": 2,
    "num_inputs": 1,
}

data_load_path = f"datasets/{DATASET}/trajectory_input_5000_5.npy"

trajectory_input = np.load(data_load_path)
trajectory = torch.tensor(trajectory_input[:, :-1, :], dtype=torch.float32).to(device)
input = torch.tensor(
    np.expand_dims(trajectory_input[:, -1, :], axis=1), dtype=torch.float32
).to(device)

if MODEL_TYPE == "MLP":
    model = MLP_AE(params=params)

if MODEL_TYPE == "TCN":
    model = TCN_AE(params=params)

if MODEL_TYPE == "LSTM":
    model = LSTM_AE(params=params)

if MODEL_TYPE == "GRU":
    model = GRU_AE(params=params)

model.to(device)

DYNAMIC_LOSS_WINDOW = 10
MODEL_NAME_TAG = f"{DATASET}_{MODEL_TYPE}_{LIFTED_STATES}ls_{TIME_WINDOW}tw"
MODEL_SAVE_DIR = f"models/{MODEL_NAME_TAG}"

if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)

optimizer = torch.optim.AdamW(model.parameters())

losses = []
for epoch in range(EPOCHS):
    print(f"Epoch: {epoch + 1}")
    loss = model.learn_koopman_eigendynamics(
        trajectory=trajectory, num_epochs=1, dynamic_loss_window=10, optimizer=optimizer
    )
    losses.append(loss.item())
    path = os.path.join(MODEL_SAVE_DIR, f"{MODEL_NAME_TAG}_{epoch+1}ep.pth")
    save_koopman_model(model, params, path)
    np.save(
        os.path.join(MODEL_SAVE_DIR, f"{MODEL_NAME_TAG}_loss.npy"), np.array(losses)
    )
