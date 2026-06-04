import torch
import math
import os
import time
import copy
import numpy as np

from lib.logger import get_logger
from lib.metrics import All_Metrics
from lib.masking_utils import mask_input_data,combined_time_node_masking

class Trainer(object):
    def __init__(self, model, loss, optimizer, train_loader, val_loader, test_loader,
                 scaler, args, lr_scheduler=None):
        super(Trainer, self).__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scaler = scaler
        self.args = args
        self.lr_scheduler = lr_scheduler
        self.train_per_epoch = len(train_loader)                        #loader类重写了长度协议，len(loader)返回批次的个数
        if val_loader != None:
            self.val_per_epoch = len(val_loader)
        self.best_path = os.path.join(self.args.log_dir, 'best_model.pth')
        self.loss_figure_path = os.path.join(self.args.log_dir, 'loss.png')
        #log
        if not os.path.isdir(args.log_dir):
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.model, debug=args.debug)
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))
        #if not args.debug:
        #self.logger.info("Argument: %r", args)
        # for arg, value in sorted(vars(args).items()):
        #     self.logger.info("Argument %s: %r", arg, value)

    def val_epoch(self, epoch, val_dataloader):
        self.model.eval()
        total_predict_loss = 0
        total_recon_loss = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_dataloader):
                # data = data[..., :self.args.input_dim]
                raw_data = data[...,:self.args.input_dim]
                label = target[..., :self.args.output_dim]

                # 1. 主任务：预测
                output, time_varying_A = self.model(data, target, "predict")

                # 2. 辅助任务：重建
                time_masked_indices, node_masked_indices = combined_time_node_masking(time_varying_A, self.args.masking_rate, self.args.high_rate, self.args.masking_rate, self.args.high_rate)
                masked_data = mask_input_data(data, time_masked_indices, node_masked_indices)
                recon = self.model(masked_data, target, "reconstruction")

                if self.args.real_value:
                    output = self.scaler.inverse_transform(output)
                    recon = self.scaler.inverse_transform(recon)
                    raw_data = self.scaler.inverse_transform(raw_data)

                loss_predict = self.loss(output, label)
                loss_recon = self.compute_masked_recon_loss(
                    raw_data, recon, time_masked_indices, node_masked_indices,
                    masked_weight=self.args.lamb, unmasked_weight=(1-self.args.lamb)
                )

                total_predict_loss += loss_predict.item()
                total_recon_loss += loss_recon.item()

        avg_predict_loss = total_predict_loss / len(val_dataloader)
        avg_recon_loss = total_recon_loss / len(val_dataloader)
        val_loss = self.args.task_rate * avg_predict_loss + (1-self.args.task_rate ) * avg_recon_loss

        self.logger.info(f'**********Val Epoch {epoch}: Predict Loss: {avg_predict_loss:.6f}, '
                         f'Recon Loss: {avg_recon_loss:.6f}, Total Loss: {val_loss:.6f}')
        return val_loss

    def mask_train_epoch(self,epoch):
        self.model.train()
        total_predict_loss = 0
        total_recon_loss = 0
        total_loss = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            # data = data[..., :self.args.input_dim]
            raw_data = data[..., :self.args.input_dim]
            label = target[..., :self.args.output_dim]  # (..., 1)
            self.optimizer.zero_grad()

            #data and target shape: B, T, N, F; output shape: B, T, N, F
            output, time_varying_A = self.model(data, target, "predict")

            time_masked_indices, node_masked_indices = combined_time_node_masking(time_varying_A, self.args.masking_rate, self.args.high_rate, self.args.masking_rate, self.args.high_rate)
            masked_data = mask_input_data(data, time_masked_indices, node_masked_indices)
            recon_data = self.model(masked_data, target, "reconstruction")

            if self.args.real_value:
                output = self.scaler.inverse_transform(output)
                recon_data = self.scaler.inverse_transform(recon_data)
                raw_data = self.scaler.inverse_transform(raw_data)

            loss_predict = self.loss(output, label)
            loss_recon = self.compute_masked_recon_loss(
                X=raw_data,
                recon_X=recon_data,
                time_indices=time_masked_indices,
                node_indices=node_masked_indices,
                masked_weight=self.args.lamb,
                unmasked_weight=1-self.args.lamb
            )
            loss = self.args.task_rate*loss_predict + (1-self.args.task_rate)*loss_recon
            loss.backward()

            # add max grad clipping
            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            total_predict_loss += loss_predict.item()
            total_recon_loss += loss_recon.item()
            total_loss += loss.item()

            #log information
            if batch_idx % self.args.log_step == 0:
                self.logger.info('Train Epoch {}: {}/{} Loss: {:.6f}'.format(
                    epoch, batch_idx, self.train_per_epoch, loss.item()))
        train_epoch_loss = total_loss / self.train_per_epoch
        predict_epoch_loss = total_predict_loss / self.train_per_epoch
        recon_epoch_loss = total_recon_loss / self.train_per_epoch
        self.logger.info('**********Train Epoch {}: averaged Loss: {:.6f}, predict Loss: {:.6f}, recon Loss: {:.6f}'.format(epoch, train_epoch_loss, predict_epoch_loss, recon_epoch_loss))

        #learning rate decay
        if self.args.lr_decay:
            self.lr_scheduler.step()
            for i, param_group in enumerate(self.optimizer.param_groups):
                self.logger.info(f"[LR Scheduler] Group {i} new learning rate: {param_group['lr']}")

        return train_epoch_loss

    def train(self):
        self.logger.info(f'begin to train! embed_dim:{self.args.embed_dim} embed_head:{self.args.embed_head} embed_d_model:{self.args.embed_d_model} rnn_units:{self.args.rnn_units}')
        best_model = None
        best_loss = float('inf')
        not_improved_count = 0
        train_loss_list = []
        val_loss_list = []
        start_time = time.time()
        for epoch in range(1, self.args.epochs + 1):
            #epoch_time = time.time()
            train_epoch_loss = self.mask_train_epoch(epoch)
            #print(time.time()-epoch_time)
            #exit()
            if self.val_loader == None:
                val_dataloader = self.test_loader
            else:
                val_dataloader = self.val_loader
            val_epoch_loss = self.val_epoch(epoch, val_dataloader)

            #print('LR:', self.optimizer.param_groups[0]['lr'])
            train_loss_list.append(train_epoch_loss)
            val_loss_list.append(val_epoch_loss)
            if train_epoch_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break
            #if self.val_loader == None:
            #val_epoch_loss = train_epoch_loss
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                not_improved_count = 0
                # save the best state
                self.logger.info('*********************************Current best model saved!')
                best_model = copy.deepcopy(self.model.state_dict())
            else:
                not_improved_count += 1
                # early stop
                if self.args.early_stop:
                    if not_improved_count == self.args.early_stop_patience:
                        self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                        "Training stops.".format(self.args.early_stop_patience))
                        break

        training_time = time.time() - start_time
        self.logger.info("Total training time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_loss))

        #save the best model to file
        if not self.args.debug:
            torch.save(best_model, self.best_path)
            self.logger.info("Saving current best model to " + self.best_path)

        #test
        self.model.load_state_dict(best_model)
        #self.val_epoch(self.args.epochs, self.test_loader)
        self.test(self.model, self.args, self.test_loader, self.scaler, self.logger)

    @staticmethod
    def compute_masked_recon_loss(
            X: torch.Tensor,
            recon_X: torch.Tensor,
            time_indices: torch.Tensor,
            node_indices: torch.Tensor,
            masked_weight: float = 1.0,
            unmasked_weight: float = 0.1
    ) -> torch.Tensor:
        """
        计算重建损失，其中掩码部分和非掩码部分分别计算 L1 损失并加权求和。

        Args:
            X (Tensor): 原始数据 [B, T, N, 1]
            recon_X (Tensor): 重建数据 [B, T, N, 1]
            time_indices (Tensor): 掩码的时间索引 [B, num_time_masked]
            node_indices (Tensor): 掩码的节点索引 [B, num_time_masked, num_node_masked]
            masked_weight (float): 掩码部分损失的权重
            unmasked_weight (float): 非掩码部分损失的权重

        Returns:
            total_loss (Tensor): 加权总损失（masked + unmasked）
        """
        B, T, N, _ = X.shape
        device = X.device

        # 创建布尔掩码矩阵，标记被掩码的位置
        mask = torch.zeros(B, T, N, dtype=torch.bool, device=device)

        for b in range(B):
            for t_idx_in_batch, t in enumerate(time_indices[b]):
                nodes = node_indices[b, t_idx_in_batch]
                mask[b, t, nodes] = True

        # 计算 L1 element-wise 误差
        abs_diff = torch.abs(X - recon_X).squeeze(-1)  # [B, T, N]

        # 掩码和非掩码区域的损失
        masked_loss = abs_diff[mask].mean() if mask.any() else torch.tensor(0.0, device=device)
        unmasked_loss = abs_diff[~mask].mean() if (~mask).any() else torch.tensor(0.0, device=device)

        # 加权求和
        total_loss = masked_weight * masked_loss + unmasked_weight * unmasked_loss

        return total_loss

    @staticmethod
    def test(model, args, data_loader, scaler, logger, path=None):
        if path is not None:
            check_point = torch.load(path)
            model.load_state_dict(check_point['state_dict'])
            model.to(args.device)

        model.eval()
        y_pred, y_true = [], []
        total_recon_loss = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                # data = data[..., :args.input_dim]
                raw_data = data[..., :args.input_dim]
                label = target[..., :args.output_dim]

                output, time_varying_A = model(data, target, "predict")
                y_true.append(label)
                y_pred.append(output)

                # ======= 计算重建损失（可选） =======
                time_masked_indices, node_masked_indices = combined_time_node_masking(time_varying_A, args.masking_rate, args.high_rate, args.masking_rate, args.high_rate)
                masked_data = mask_input_data(data, time_masked_indices, node_masked_indices)
                recon = model(masked_data, target, "reconstruction")
                if args.real_value:
                    recon = scaler.inverse_transform(recon)
                    raw_data = scaler.inverse_transform(raw_data)

                recon_loss = Trainer.compute_masked_recon_loss(
                    raw_data, recon, time_masked_indices, node_masked_indices,
                    masked_weight=args.lamb, unmasked_weight=(1-args.lamb)
                )
                total_recon_loss += recon_loss.item()

        y_pred = torch.cat(y_pred)
        y_true = torch.cat(y_true)
        if args.real_value:
            y_pred = scaler.inverse_transform(y_pred)

        np.save(f'{args.log_dir}/{args.dataset}_true.npy', y_true.cpu().numpy())
        np.save(f'{args.log_dir}/{args.dataset}_pred.npy', y_pred.cpu().numpy())

        # 打印各 horizon 下的评估指标
        for t in range(y_true.shape[1]):
            mae, rmse, mape, _, _ = All_Metrics(
                y_pred[:, t], y_true[:, t], args.mae_thresh, args.mape_thresh
            )
            logger.info(f"Horizon {t + 1:02d}, MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape * 100:.4f}%")

        mae, rmse, mape, _, _ = All_Metrics(
            y_pred, y_true, args.mae_thresh, args.mape_thresh
        )
        logger.info(f"Average Horizon, MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape * 100:.4f}%")
        logger.info(f"[TEST] Average reconstruction loss: {total_recon_loss / len(data_loader):.6f}")

    @staticmethod
    def _compute_sampling_threshold(global_step, k):
        """
        Computes the sampling probability for scheduled sampling using inverse sigmoid.
        :param global_step:
        :param k:
        :return:
        """
        return k / (k + math.exp(global_step / k))
