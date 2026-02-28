import torch
import torch.nn as nn
import numpy as np
from torchmetrics import MeanAbsoluteError, MeanSquaredError, R2Score
from tqdm.auto import tqdm
from collections import defaultdict


class Trainer:
    def __init__(self,
                 device,
                 random_state: int = 77):

        torch.manual_seed(random_state)
        np.random.seed(random_state)

        self.device = device

        self.mse = MeanSquaredError().to(device)
        self.mae = MeanAbsoluteError().to(device)
        self.r2 = R2Score().to(device)

    def train_step(self,
                   train_dataloader: torch.utils.data.DataLoader,
                   model: nn.Module,
                   optimizer: torch.optim.Optimizer,
                   loss_fn: nn.Module):

        train_loss = torch.tensor(0.0, device=self.device)

        model = model.to(self.device)
        model.train()

        self.mse.reset()
        self.mae.reset()
        self.r2.reset()

        for batch, (X_train, y_train) in enumerate(train_dataloader):
            X_train, y_train = X_train.to(self.device), y_train.to(self.device)

            optimizer.zero_grad()

            y_pred = model(X_train).view(-1)
            y_train = y_train.view(-1)

            loss = loss_fn(y_pred, y_train)
            loss.backward()
            optimizer.step()

            train_loss += loss.detach()

            self.mse(y_pred, y_train)
            self.mae(y_pred, y_train)
            self.r2(y_pred, y_train)

        train_mse = self.mse.compute().item()
        train_mae = self.mae.compute().item()
        train_r2 = self.r2.compute().item()

        train_loss = train_loss.item() / len(train_dataloader)

        return train_loss, train_mse, train_mae, train_r2

    def valid_step(self,
                  valid_dataloader: torch.utils.data.DataLoader,
                  model: nn.Module,
                  loss_fn: nn.Module):

        valid_loss = torch.tensor(0.0, device=self.device)
        model = model.to(self.device)
        y_preds = []
        self.mse.reset()
        self.mae.reset()
        self.r2.reset()

        with torch.inference_mode():
            model.eval()
            for batch, (X_val, y_val) in enumerate(valid_dataloader):
                X_val, y_val = X_val.to(self.device), y_val.to(self.device)

                y_pred = model(X_val).view(-1)
                y_val = y_val.view(-1)
                y_preds.append(y_pred)

                loss = loss_fn(y_pred, y_val)
                valid_loss += loss.detach()

                self.mse(y_pred, y_val)
                self.mae(y_pred, y_val)
                self.r2(y_pred, y_val)

        valid_mse = self.mse.compute().item()
        valid_mae = self.mae.compute().item()
        valid_r2 = self.r2.compute().item()

        valid_loss = valid_loss.item() / len(valid_dataloader)

        return valid_loss, valid_mse, valid_mae, valid_r2, torch.cat(y_preds).cpu()

    def model_eval(self,
                   train_dataloader: torch.utils.data.DataLoader,
                   valid_dataloader: torch.utils.data.DataLoader,
                   model: nn.Module,
                   optimizer: torch.optim.Optimizer,
                   loss_fn: nn.Module,
                   num_epochs: int = 100,
                   patience: int = 5,
                   tolerance: float = 0.01):

        best_loss = float("inf")
        counter = 0

        results = defaultdict(list)

        for epoch in tqdm(range(num_epochs)):

            train_loss, train_mse, train_mae, train_r2 = self.train_step(
                train_dataloader=train_dataloader,
                model=model,
                optimizer=optimizer,
                loss_fn=loss_fn
            )

            valid_loss, valid_mse, valid_mae, valid_r2, _ = self.valid_step(
                valid_dataloader=valid_dataloader,
                model=model,
                loss_fn=loss_fn
            )

            if valid_loss < (best_loss - tolerance):
                best_loss = valid_loss
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping | No improvement since {patience} epochs")
                    break

            results["train_loss"].append(train_loss)
            results["valid_loss"].append(valid_loss)
            results["train_mse"].append(train_mse)
            results["valid_mse"].append(valid_mse)
            results["train_mae"].append(train_mae)
            results["valid_mae"].append(valid_mae)
            results["train_r2"].append(train_r2)
            results["valid_r2"].append(valid_r2)

            log_interval = max(1, int(num_epochs / 10))
            if epoch % log_interval == 0:
                print(f"Epoch: {epoch} | Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f}\n"
                      f"Train MSE: {train_mse:.4f} | Valid MSE: {valid_mse:.4f}\n"
                      f"Train MAE: {train_mae:.4f} | Valid MAE : {valid_mae:.4f}\n"
                      f"Train R2: {train_r2:.4f} | Valid R2 : {valid_r2:.4f}")

        return results