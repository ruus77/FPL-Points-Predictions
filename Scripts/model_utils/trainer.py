import torch
import torch.nn as nn
import numpy as np
from torchmetrics import MeanAbsoluteError, MeanSquaredError, R2Score
from tqdm.auto import tqdm
from collections import defaultdict
import copy


def set_seed(seed: int = 77):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(77)


class Trainer:
    def __init__(self,
                 device,
                 random_state: int = 77
                 ):

        torch.manual_seed(random_state)
        np.random.seed(random_state)
        self.minutes_idx = 18
        self.device = device
        self.mse = MeanSquaredError().to(device)
        self.mae = MeanAbsoluteError().to(device)
        self.r2_metric = R2Score().to(device)

    def train_step(self,
                   train_dataloader: torch.utils.data.DataLoader,
                   model: nn.Module,
                   optimizer: torch.optim.Optimizer,
                   loss_fn: nn.Module):

        train_loss = torch.tensor(0.0, device=self.device)
        model = model.to(self.device)
        model.train()
        self.penalty_rate = 5.0
        self.mse.reset()
        self.mae.reset()
        self.r2_metric.reset()

        for batch, (X_train, y_train) in enumerate(train_dataloader):
            X_train, y_train = X_train.to(self.device), y_train.to(self.device)

            optimizer.zero_grad()
            y_pred = model(X_train)

            minutes_ema = X_train[:, self.minutes_idx]
            minute_weights = (minutes_ema / 90.0) + 0.1

            point_weights = torch.where(y_train > 6.0, self.penalty_rate, 1.0).squeeze()

            final_weights = minute_weights * point_weights

            loss = loss_fn(y_pred, y_train.view_as(y_pred))
            loss = (loss.squeeze() * final_weights).mean()

            loss.backward()
            optimizer.step()

            train_loss += loss.detach()

            self.mse(y_pred, y_train)
            self.mae(y_pred, y_train)
            self.r2_metric(y_pred, y_train)

        train_mse = self.mse.compute().item()
        train_mae = self.mae.compute().item()
        train_r2 = self.r2_metric.compute().item()

        train_loss = train_loss.item() / len(train_dataloader)

        return train_loss, train_mse, train_mae, train_r2

    def test_step(self,
                  test_dataloader: torch.utils.data.DataLoader,
                  model: nn.Module,
                  loss_fn: nn.Module):

        test_loss = torch.tensor(0.0, device=self.device)
        model = model.to(self.device)
        y_preds = []

        self.mse.reset()
        self.mae.reset()
        self.r2_metric.reset()

        with torch.inference_mode():
            model.eval()
            for batch, (X_test, y_test) in enumerate(test_dataloader):
                X_test, y_test = X_test.to(self.device), y_test.to(self.device)

                y_pred = model(X_test)
                y_preds.append(y_pred)

                minutes_ema = X_test[:, self.minutes_idx]
                minute_weights = (minutes_ema / 90.0) + 0.1

                point_weights = torch.where(y_test > 6.0, self.penalty_rate, 1.0).squeeze()
                final_weights = minute_weights * point_weights

                loss = loss_fn(y_pred, y_test.view_as(y_pred))
                loss = (loss.squeeze() * final_weights).mean()

                test_loss += loss.detach()

                self.mse(y_pred, y_test)
                self.mae(y_pred, y_test)
                self.r2_metric(y_pred, y_test)

        test_mse = self.mse.compute().item()
        test_mae = self.mae.compute().item()
        test_r2 = self.r2_metric.compute().item()

        test_loss = test_loss.item() / len(test_dataloader)

        return test_loss, test_mse, test_mae, test_r2, torch.cat(y_preds).cpu()

    def model_eval(self,
                   train_dataloader: torch.utils.data.DataLoader,
                   test_dataloader: torch.utils.data.DataLoader,
                   model: nn.Module,
                   optimizer: torch.optim.Optimizer,
                   loss_fn: nn.Module,
                   num_epochs: int = 100,
                   patience: int = 5,
                   tolerance: float = 0.01):

        best_model = copy.deepcopy(model.state_dict())

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3,
        )

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

            test_loss, test_mse, test_mae, test_r2, _ = self.test_step(
                test_dataloader=test_dataloader,
                model=model,
                loss_fn=loss_fn
            )

            scheduler.step(test_loss)

            current_lr = optimizer.param_groups[0]['lr']

            results["train_loss"].append(train_loss)
            results["test_loss"].append(test_loss)
            results["train_mse"].append(train_mse)
            results["test_mse"].append(test_mse)
            results["train_mae"].append(train_mae)
            results["test_mae"].append(test_mae)
            results["train_r2"].append(train_r2)
            results["test_r2"].append(test_r2)
            results["lr"].append(current_lr)

            log_interval = max(1, int(num_epochs / 10))
            if epoch % log_interval == 0:
                print(
                    f"Epoch: {epoch} | LR: {current_lr:.6f} | Train Loss: {train_loss:.3f} | Test Loss: {test_loss:.3f}\n"
                    f"Train MSE: {train_mse:.3f} | Test MSE: {test_mse:.3f}\n"
                    f"Train MAE: {train_mae:.3f} | Test MAE : {test_mae:.3f}\n"
                    f"Train R2: {train_r2:.3f} | Test R2: {test_r2:.3f}\n")

            if test_loss < (best_loss - tolerance):
                best_loss = test_loss
                counter = 0
                best_model = copy.deepcopy(model.state_dict())
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping | No improvement since {patience} epochs")
                    break

        return results, best_model