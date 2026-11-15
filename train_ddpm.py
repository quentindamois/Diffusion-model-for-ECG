import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from ecgDataset import ECGDataset
import numpy as np
import random
from scipy.stats import norm, uniform
from denoisingSeqAutoEncoder import AEncoder1DUNet
from tqdm import tqdm
df = pd.read_csv("mitbih_train.csv", header=None)

random_seed = 42

torch.manual_seed(random_seed)
random.seed(random_seed)

df_np_train, df_np_test = train_test_split(df.to_numpy(), test_size=0.2)

df_np_train = (df_np_train - np.min(df_np_train))/(np.max(df_np_train) - np.min(df_np_train))
df_np_test = (df_np_test - np.min(df_np_test))/(np.max(df_np_test) - np.min(df_np_test))
epochs = 1000

batch_size = 128

learning_rate = 1e-6

nb_step = 100
linear_Beta = np.linspace(1e-6, 2e-1, nb_step)
alpha = np.cumprod(1 - linear_Beta, dtype=np.float32)

if torch.xpu.is_available():
    device = torch.device("xpu")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

best_loss = np.inf
max_patience = 15
patience = 0
lr_scheduler_patience = 5

if __name__ == "__main__":
    ds_train = ECGDataset(df_np_train)
    ds_test = ECGDataset(df_np_test)
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size)
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=batch_size)

    model = AEncoder1DUNet()
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=lr_scheduler_patience)
    model.to(device)
    loss_compute = torch.nn.MSELoss()

    for epoch in range(epochs):
        sum_loss = 0
        sum_ttl_loss = 0
        sum_var_loss = 0
        sum
        for batch in tqdm(dl_train, desc=f"Training, epoch: {epoch}"):
            target_data = batch.to(device)
            #noise = norm.rvs(random_state=random_seed)
            noise = torch.randn_like(batch)
            random_time_step = int(uniform.rvs(0, nb_step - 1, random_state=random_seed))
            noisy_input = float(np.sqrt(1 -  alpha[random_time_step])) * batch + float(np.sqrt(alpha[random_time_step])) * noise
            noisy_input = noisy_input.to(device).unsqueeze(1)
            target_data = target_data.unsqueeze(1)
            denoised_prediction = model(noisy_input)
            loss = loss_compute(target_data, denoised_prediction)
            loss_var = torch.sum((torch.var(target_data.squeeze()) - torch.var(denoised_prediction.squeeze()))**2)
            ttl_loss = loss
            ttl_loss.backward()
            opt.step()
            opt.zero_grad(True)
            sum_loss += loss.item()
            sum_ttl_loss += ttl_loss.item()
            sum_var_loss += loss_var
        sum_loss_val = 0
        sum_ttl_loss_val = 0
        sum_var_loss_val = 0
        model.eval()
        with torch.no_grad():
            for batch in tqdm(dl_test, desc=f"Evaluation, epoch: {epoch}"):
                target_data = batch.to(device)
                noise = norm.rvs(random_state=random_seed)
                random_time_step = int(uniform.rvs(0, nb_step - 1, random_state=random_seed))
                noisy_input = float(np.sqrt(1 -  alpha[random_time_step])) * batch + float(np.sqrt(alpha[random_time_step]) * noise)
                noisy_input = noisy_input.to(device).unsqueeze(1)
                target_data = target_data.unsqueeze(1)
                denoised_prediction = model(noisy_input)
                loss = loss_compute(target_data, denoised_prediction)
                loss_var = torch.sum((torch.var(target_data.squeeze()) - torch.var(denoised_prediction.squeeze()))**2)
                sum_loss_val += loss.item()
                ttl_loss = loss
                sum_ttl_loss_val += ttl_loss.item()
                sum_var_loss_val += loss_var
        model.train()
        lr_scheduler.step(sum_loss_val)
        print(f"Epoch: {epoch}, Sum training loss: {sum_ttl_loss} (rec loss: {sum_loss}, var loss: {sum_var_loss}), Sum val loss: {sum_ttl_loss_val} (rec loss: {sum_loss_val}, var loss: {sum_var_loss_val})")
        if best_loss > sum_loss_val:
            best_loss = sum_loss_val
            torch.save(model.state_dict(), "ddpm_model.pt")
            print("New best model saved")
            patience = 0
        else:
            if patience < max_patience:
                patience += 1
                print(f"No improvement, patience = {patience}")
            else:
                print(f"Reached the maximum number of iteration with no improvement.")
                break
        



