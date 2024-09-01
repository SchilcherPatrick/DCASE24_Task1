import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torch
import torchaudio
from torch.utils.data import DataLoader
import argparse
import torch.nn.functional as F
import transformers
import wandb
import json
import torch.nn as nn
import os

from dataset.dcase24 import get_training_set, get_test_set, get_eval_set
from helpers.init import worker_init_fn
from models.baseline import get_model
from helpers.utils import mixstyle, filteraugment
from helpers.checkpoints import get_state_dict
from helpers import nessi
from models.cp_resnet import get_model as get_model_teacher
from models.passt import get_model as get_model_teacher_p
from models.helpers.ensemble import BEAEnsemble


class PLModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        # results from argparse, contains all configurations for our experiment
        self.config = config

        # module for resampling waveforms on the fly
        resample = torchaudio.transforms.Resample(
            orig_freq=self.config.orig_sample_rate,
            new_freq=self.config.sample_rate
        )
        resample_p = torchaudio.transforms.Resample(
            orig_freq=self.config.orig_sample_rate,
            new_freq=self.config.sample_rate_p
        )

        # module to preprocess waveforms into log mel spectrograms
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            win_length=config.window_length,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            f_min=config.f_min,
            f_max=config.f_max
        )
        mel_p = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate_p,
            n_fft=config.n_fft_p,
            win_length=config.window_length_p,
            hop_length=config.hop_length_p,
            n_mels=config.n_mels_p,
            f_min=config.f_min_p,
            f_max=config.f_max_P
        )

        freqm = torchaudio.transforms.FrequencyMasking(config.freqm, iid_masks=True)
        timem = torchaudio.transforms.TimeMasking(config.timem, iid_masks=True)

        self.mel = torch.nn.Sequential(
            resample,
            mel
        )
        self.mel_p = torch.nn.Sequential(
            resample_p,
            mel_p
        )

        self.mel_augment = torch.nn.Sequential(
            freqm,
            timem
        )

        self.cp_resnet_teachers = []
        if config.cpresnet_teachers:
            # load all cpresnet models
            cp_resnet_ids = config.cpresnet_teachers.split(",")
            cp_resnet_projects = config.passt_teachers_project.split(",")
            if len(cp_resnet_projects) == 1:
                cp_resnet_projects = cp_resnet_projects*len(cp_resnet_ids) # repeat for each id
            assert len(cp_resnet_projects) == len(cp_resnet_ids), "Mismatched amount of CPResNet ids and project names"
            
            for i, run_id in enumerate(cp_resnet_ids):
                state_dict = get_state_dict(config.entity, 
                                            config.cpresnet_teachers_project,
                                            run_id)
                
                cpresnet_model = get_model_teacher(base_channels=config.base_channels)
                cpresnet_model.load_state_dict(state_dict)
                self.cp_resnet_teachers.append(f"cpresnet_{i:02d}_{run_id}")
                #set model as class attribute so it gets recognized by pytorch
                setattr(self, f"cpresnet_{i:02d}_{run_id}", cpresnet_model)

        self.num_cp_resnet_teachers = len(self.cp_resnet_teachers)

        self.cpm_teachers = []
        if config.cpm_teachers:
            # load all cpm teacher models
            cpm_teacher_ids = config.cpm_teachers.split(",")
            cpm_teacher_projects = config.passt_teachers_project.split(",")
            if len(cpm_teacher_projects) == 1:
                cpm_teacher_projects = cpm_teacher_projects*len(cpm_teacher_ids) # repeat for each id
            assert len(cpm_teacher_projects) == len(cpm_teacher_ids), "Mismatched amount of CPM ids and project names"

            for i, run_id in enumerate(cpm_teacher_ids):
                state_dict = get_state_dict(config.entity, 
                                            config.cpm_teachers_project,
                                            run_id)
                
                cpm_model = get_model(n_classes=config.n_classes,
                               in_channels=config.in_channels,
                               base_channels=config.base_channels,
                               channels_multiplier=config.channels_multiplier,
                               expansion_rate=config.expansion_rate
                               )
                cpm_model.load_state_dict(state_dict)
                self.cpm_teachers.append(f"cpm_{i:02d}_{run_id}")
                #set model as class attribute so it gets recognized by pytorch
                setattr(self, f"cpm_{i:02d}_{run_id}", cpm_model)

        self.num_cpm_teachers = len(self.cpm_teachers)

        self.passt_teachers = []
        if config.passt_teachers:
            # load all passt models
            passt_ids = config.passt_teachers.split(",")
            passt_projects = config.passt_teachers_project.split(",")
            if len(passt_projects) == 1:
                passt_projects = passt_projects*len(passt_ids) # repeat for each id
            assert len(passt_projects) == len(passt_ids), "Mismatched amount of PaSST ids and project names"

            for i, run_id in enumerate(passt_ids):
                state_dict = get_state_dict(config.entity, 
                                            config.passt_teachers_project,
                                            run_id)
                
                passt_model = get_model_teacher_p(arch=config.arch,
                             pretrained=False,
                             n_classes=config.n_classes,
                             input_fdim=config.input_fdim,
                             s_patchout_t=config.s_patchout_t,
                             s_patchout_f=config.s_patchout_f) 
                passt_model.load_state_dict(state_dict)
                self.passt_teachers.append(f"passt_{i:02d}_{run_id}")
                #set model as class attribute so it gets recognized by pytorch
                setattr(self, f"passt_{i:02d}_{run_id}", passt_model)

        self.num_passt_teachers = len(self.passt_teachers)
        
        
        self.model = get_model(n_classes=config.n_classes,
                               in_channels=config.in_channels,
                               base_channels=config.base_channels,
                               channels_multiplier=config.channels_multiplier,
                               expansion_rate=config.expansion_rate
                               )
        if config.pretrained_student:
            # load pretrained student
            state_dict = get_state_dict(config.entity, 
                                        config.pretrained_student_project,
                                        config.pretrained_student)
            self.model.load_state_dict(state_dict)

        self.kl_div_loss = nn.KLDivLoss(log_target=True, reduction="none")  # KL Divergence loss for soft targets
        
        self.bea_ensemble = BEAEnsemble([getattr(self, t) for t in self.cp_resnet_teachers] + 
                                          [getattr(self, t) for t in self.cpm_teachers], 
                                          [getattr(self, t) for t in self.passt_teachers])
        
        self.device_ids = ['a', 'b', 'c', 's1', 's2', 's3', 's4', 's5', 's6']
        self.label_ids = ['airport', 'bus', 'metro', 'metro_station', 'park', 'public_square', 'shopping_mall',
                          'street_pedestrian', 'street_traffic', 'tram']
        # categorization of devices into 'real', 'seen' and 'unseen'
        self.device_groups = {'a': "real", 'b': "real", 'c': "real",
                              's1': "seen", 's2': "seen", 's3': "seen",
                              's4': "unseen", 's5': "unseen", 's6': "unseen"}

        # pl 2 containers:
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def mel_forward(self, x):
        """
        Process melspectogram for CPResNet and CPM models
        :param x: batch of raw audio signals (waveforms)
        :return: log mel spectrogram
        """
        x = self.mel(x)
        if self.training:
            x = self.mel_augment(x)
        x = (x + 1e-5).log()
        return x
    
    def mel_forward_p(self, x):
        """
        Process melspectogram for PaSST models
        :param x: batch of raw audio signals (waveforms)
        :return: log mel spectrogram
        """
        x = self.mel_p(x)
        if self.training:
            x = self.mel_augment(x)
        x = (x + 1e-5).log()
        return x

    def forward(self, x):
        """
        :param x: batch of raw audio signals (waveforms)
        :return: final model predictions
        """
        x = self.mel_forward(x)
        x = self.model(x)
        return x

    def configure_optimizers(self):
        """
        This is the way pytorch lightening requires optimizers and learning rate schedulers to be defined.
        The specified items are used automatically in the optimization loop (no need to call optimizer.step() yourself).
        :return: optimizer and learning rate scheduler
        """

        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )

        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1
        }
        return [optimizer], [lr_scheduler_config]
    
    def on_train_start(self) -> None:
        # set all teachers to eval mode
        for t in self.cp_resnet_teachers:
            model = getattr(self, t)
            model.to(self.device)
            model.eval()
        for t in self.cpm_teachers:
            model = getattr(self, t)
            model.to(self.device)
            model.eval()
        for t in self.passt_teachers:
            model = getattr(self, t)
            model.to(self.device)
            model.eval()

        return super().on_train_start()

    def training_step(self, train_batch, batch_idx):
        """
        :param train_batch: contains one batch from train dataloader
        :param batch_idx
        :return: loss to update model parameters
        """
        x, files, labels, devices, cities = train_batch

        x_cp = self.mel_forward(x)  # we convert the raw audio signals into log mel spectrograms
        x_p = self.mel_forward_p(x)
        
        # frequency mixstyle
        if self.config.mixstyle_p > 0:
            x_cp = mixstyle(x_cp, self.config.mixstyle_p, self.config.mixstyle_alpha)

        # filteraugment
        if self.config.filteraugment:
            x_cp = filteraugment(features=x_cp, 
                              db_range=[self.config.filt_aug_db_l, 
                                        self.config.filt_aug_db_h],
                              n_band = [self.config.filt_aug_n_band_l, 
                                        self.config.filt_aug_n_band_h], 
                              min_bw=self.config.filt_aug_min_bw, 
                              filter_type=self.config.filt_aug_type) 
            
        y_hat = self.model(x_cp)
        samples_loss = F.cross_entropy(y_hat, labels.long(), reduction="none")
        label_loss = samples_loss.mean()

        with torch.cuda.amp.autocast():
            y_hat_soft = F.log_softmax(y_hat / self.config.temperature, dim=-1)

        teacher_logits = self.bea_ensemble(x_cp, x_p)
        
        teacher_logits_soft = F.log_softmax(teacher_logits / self.config.temperature, dim=-1)
        kd_loss = self.kl_div_loss(y_hat_soft, teacher_logits_soft).mean()

        kd_loss = kd_loss * (self.config.temperature ** 2)
        loss = self.config.kd_lambda * label_loss + (1 - self.config.kd_lambda) * kd_loss

        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'])
        self.log("epoch", self.current_epoch)
        self.log("train/loss", loss.detach().cpu())
        self.log("train/label_loss", (label_loss * self.config.kd_lambda).detach().cpu())
        self.log("train/kd_loss", (kd_loss * (1 - self.config.kd_lambda)).detach().cpu())
        return loss

    def on_train_epoch_end(self):
        if self.current_epoch==self.config.n_epochs-1:
            torch.save(self.model.state_dict(), 'model_params.pth')

    def validation_step(self, val_batch, batch_idx):
        x, files, labels, devices, cities = val_batch

        y_hat = self.forward(x)
        samples_loss = F.cross_entropy(y_hat, labels.long(), reduction="none")

        # for computing accuracy
        _, preds = torch.max(y_hat, dim=1)
        n_correct_per_sample = (preds == labels)
        n_correct = n_correct_per_sample.sum()

        dev_names = [d.rsplit("-", 1)[1][:-4] for d in files]
        results = {'loss': samples_loss.mean(), "n_correct": n_correct,
                   "n_pred": torch.as_tensor(len(labels), device=self.device)}

        # log metric per device and scene
        for d in self.device_ids:
            results["devloss." + d] = torch.as_tensor(0., device=self.device)
            results["devcnt." + d] = torch.as_tensor(0., device=self.device)
            results["devn_correct." + d] = torch.as_tensor(0., device=self.device)
        for i, d in enumerate(dev_names):
            results["devloss." + d] = results["devloss." + d] + samples_loss[i]
            results["devn_correct." + d] = results["devn_correct." + d] + n_correct_per_sample[i]
            results["devcnt." + d] = results["devcnt." + d] + 1

        for l in self.label_ids:
            results["lblloss." + l] = torch.as_tensor(0., device=self.device)
            results["lblcnt." + l] = torch.as_tensor(0., device=self.device)
            results["lbln_correct." + l] = torch.as_tensor(0., device=self.device)
        for i, l in enumerate(labels):
            results["lblloss." + self.label_ids[l]] = results["lblloss." + self.label_ids[l]] + samples_loss[i]
            results["lbln_correct." + self.label_ids[l]] = \
                results["lbln_correct." + self.label_ids[l]] + n_correct_per_sample[i]
            results["lblcnt." + self.label_ids[l]] = results["lblcnt." + self.label_ids[l]] + 1
        results = {k: v.cpu() for k, v in results.items()}
        self.validation_step_outputs.append(results)

    def on_validation_epoch_end(self):
        # convert a list of dicts to a flattened dict
        outputs = {k: [] for k in self.validation_step_outputs[0]}
        for step_output in self.validation_step_outputs:
            for k in step_output:
                outputs[k].append(step_output[k])
        for k in outputs:
            outputs[k] = torch.stack(outputs[k])

        avg_loss = outputs['loss'].mean()
        acc = sum(outputs['n_correct']) * 1.0 / sum(outputs['n_pred'])

        logs = {'acc': acc, 'loss': avg_loss}

        # log metric per device and scene
        for d in self.device_ids:
            dev_loss = outputs["devloss." + d].sum()
            dev_cnt = outputs["devcnt." + d].sum()
            dev_corrct = outputs["devn_correct." + d].sum()
            logs["loss." + d] = dev_loss / dev_cnt
            logs["acc." + d] = dev_corrct / dev_cnt
            logs["cnt." + d] = dev_cnt
            # device groups
            logs["acc." + self.device_groups[d]] = logs.get("acc." + self.device_groups[d], 0.) + dev_corrct
            logs["count." + self.device_groups[d]] = logs.get("count." + self.device_groups[d], 0.) + dev_cnt
            logs["lloss." + self.device_groups[d]] = logs.get("lloss." + self.device_groups[d], 0.) + dev_loss

        for d in set(self.device_groups.values()):
            logs["acc." + d] = logs["acc." + d] / logs["count." + d]
            logs["lloss." + d] = logs["lloss." + d] / logs["count." + d]

        for l in self.label_ids:
            lbl_loss = outputs["lblloss." + l].sum()
            lbl_cnt = outputs["lblcnt." + l].sum()
            lbl_corrct = outputs["lbln_correct." + l].sum()
            logs["loss." + l] = lbl_loss / lbl_cnt
            logs["acc." + l] = lbl_corrct / lbl_cnt
            logs["cnt." + l] = lbl_cnt

        logs["macro_avg_acc"] = torch.mean(torch.stack([logs["acc." + l] for l in self.label_ids]))
        # prefix with 'val' for logging
        self.log_dict({"val/" + k: logs[k] for k in logs})
        self.validation_step_outputs.clear()

    def test_step(self, test_batch, batch_idx):
        x, files, labels, devices, cities = test_batch

        # maximum memory allowance for parameters: 128 KB
        # baseline has 61148 parameters -> we can afford 16-bit precision
        # since 61148 * 16 bit ~ 122 kB

        # assure fp16
        self.model.half()
        x = self.mel_forward(x)
        x = x.half()
        y_hat = self.model(x)
        samples_loss = F.cross_entropy(y_hat, labels.long(), reduction="none")

        # for computing accuracy
        _, preds = torch.max(y_hat, dim=1)
        n_correct_per_sample = (preds == labels)
        n_correct = n_correct_per_sample.sum()

        dev_names = [d.rsplit("-", 1)[1][:-4] for d in files]
        results = {'loss': samples_loss.mean(), "n_correct": n_correct,
                   "n_pred": torch.as_tensor(len(labels), device=self.device)}

        # log metric per device and scene
        for d in self.device_ids:
            results["devloss." + d] = torch.as_tensor(0., device=self.device)
            results["devcnt." + d] = torch.as_tensor(0., device=self.device)
            results["devn_correct." + d] = torch.as_tensor(0., device=self.device)
        for i, d in enumerate(dev_names):
            results["devloss." + d] = results["devloss." + d] + samples_loss[i]
            results["devn_correct." + d] = results["devn_correct." + d] + n_correct_per_sample[i]
            results["devcnt." + d] = results["devcnt." + d] + 1

        for l in self.label_ids:
            results["lblloss." + l] = torch.as_tensor(0., device=self.device)
            results["lblcnt." + l] = torch.as_tensor(0., device=self.device)
            results["lbln_correct." + l] = torch.as_tensor(0., device=self.device)
        for i, l in enumerate(labels):
            results["lblloss." + self.label_ids[l]] = results["lblloss." + self.label_ids[l]] + samples_loss[i]
            results["lbln_correct." + self.label_ids[l]] = \
                results["lbln_correct." + self.label_ids[l]] + n_correct_per_sample[i]
            results["lblcnt." + self.label_ids[l]] = results["lblcnt." + self.label_ids[l]] + 1
        self.test_step_outputs.append(results)

    def on_test_epoch_end(self):
        # convert a list of dicts to a flattened dict
        outputs = {k: [] for k in self.test_step_outputs[0]}
        for step_output in self.test_step_outputs:
            for k in step_output:
                outputs[k].append(step_output[k])
        for k in outputs:
            outputs[k] = torch.stack(outputs[k])

        avg_loss = outputs['loss'].mean()
        acc = sum(outputs['n_correct']) * 1.0 / sum(outputs['n_pred'])

        logs = {'acc': acc, 'loss': avg_loss}

        # log metric per device and scene
        for d in self.device_ids:
            dev_loss = outputs["devloss." + d].sum()
            dev_cnt = outputs["devcnt." + d].sum()
            dev_corrct = outputs["devn_correct." + d].sum()
            logs["loss." + d] = dev_loss / dev_cnt
            logs["acc." + d] = dev_corrct / dev_cnt
            logs["cnt." + d] = dev_cnt
            # device groups
            logs["acc." + self.device_groups[d]] = logs.get("acc." + self.device_groups[d], 0.) + dev_corrct
            logs["count." + self.device_groups[d]] = logs.get("count." + self.device_groups[d], 0.) + dev_cnt
            logs["lloss." + self.device_groups[d]] = logs.get("lloss." + self.device_groups[d], 0.) + dev_loss

        for d in set(self.device_groups.values()):
            logs["acc." + d] = logs["acc." + d] / logs["count." + d]
            logs["lloss." + d] = logs["lloss." + d] / logs["count." + d]

        for l in self.label_ids:
            lbl_loss = outputs["lblloss." + l].sum()
            lbl_cnt = outputs["lblcnt." + l].sum()
            lbl_corrct = outputs["lbln_correct." + l].sum()
            logs["loss." + l] = lbl_loss / lbl_cnt
            logs["acc." + l] = lbl_corrct / lbl_cnt
            logs["cnt." + l] = lbl_cnt

        logs["macro_avg_acc"] = torch.mean(torch.stack([logs["acc." + l] for l in self.label_ids]))
        # prefix with 'test' for logging
        self.log_dict({"test/" + k: logs[k] for k in logs})
        self.test_step_outputs.clear()

    def predict_step(self, eval_batch, batch_idx, dataloader_idx=0):
        x, files = eval_batch

        # assure fp16
        self.model.half()

        x = self.mel_forward(x)
        x = x.half()
        y_hat = self.model(x)

        return files, y_hat


def train(config):
    # init wandb before WandbLogger to log used artifacts in init of PLModule
    # run will be reused by WandbLogger
    run = wandb.init(
                project=config.project_name,
                tags=["DCASE24", "KD_BEA"],
                config=config,  # this logs all hyperparameters for us
                name=config.experiment_name
            )

    # logging is done using wandb
    wandb_logger = WandbLogger(
        project=config.project_name,
        tags=["DCASE24", "KD_BEA"],
        config=config,  # this logs all hyperparameters for us
        name=config.experiment_name,
        log_model=config.log_model
    )

    # train dataloader
    assert config.subset in {100, 50, 25, 10, 5}, "Specify an integer value in: {100, 50, 25, 10, 5} to use one of " \
                                                  "the given subsets."
    roll_samples = config.orig_sample_rate * config.roll_sec
    train_dl = DataLoader(dataset=get_training_set(config.subset, 
                                                   roll=roll_samples, 
                                                   dir_p=config.dir_p, 
                                                   resample_rate=config.sample_rate),
                          worker_init_fn=worker_init_fn,
                          num_workers=config.num_workers,
                          batch_size=config.batch_size,
                          shuffle=True)

    test_dl = DataLoader(dataset=get_test_set(),
                         worker_init_fn=worker_init_fn,
                         num_workers=config.num_workers,
                         batch_size=config.batch_size)

    # create pytorch lightening module
    pl_module = PLModule(config)

    # get model complexity from nessi and log results to wandb
    sample = next(iter(test_dl))[0][0].unsqueeze(0)

    shape = pl_module.mel_forward(sample).size()
    macs, params = nessi.get_torch_size(pl_module.model, input_size=shape)
    # log MACs and number of parameters for our model
    wandb_logger.experiment.config['MACs'] = macs
    wandb_logger.experiment.config['Parameters'] = params

    # create the pytorch lightening trainer by specifying the number of epochs to train, the logger,
    # on which kind of device(s) to train and possible callbacks
    trainer = pl.Trainer(max_epochs=config.n_epochs,
                         logger=wandb_logger,
                         accelerator='auto',
                         devices=1,
                         precision=config.precision,
                         callbacks=[pl.callbacks.ModelCheckpoint(save_last=True)])
    # start training and validation for the specified number of epochs
    trainer.fit(pl_module, train_dl, test_dl)

    # final test step
    # here: use the validation split
    trainer.test(ckpt_path='last', dataloaders=test_dl)

    wandb.finish()


def evaluate(config):
    import os
    from sklearn import preprocessing
    import pandas as pd
    import torch.nn.functional as F
    from dataset.dcase24 import dataset_config

    assert config.ckpt_id is not None, "A value for argument 'ckpt_id' must be provided."
    ckpt_dir = os.path.join(config.project_name, config.ckpt_id, "checkpoints")
    assert os.path.exists(ckpt_dir), f"No such folder: {ckpt_dir}"
    ckpt_file = os.path.join(ckpt_dir, "last.ckpt")
    assert os.path.exists(ckpt_file), f"No such file: {ckpt_file}. Implement your own mechanism to select" \
                                      f"the desired checkpoint."

    # create folder to store predictions
    os.makedirs("predictions", exist_ok=True)
    out_dir = os.path.join("predictions", config.ckpt_id)
    os.makedirs(out_dir, exist_ok=True)

    # load lightning module from checkpoint
    pl_module = PLModule.load_from_checkpoint(ckpt_file, config=config)
    trainer = pl.Trainer(logger=False,
                         accelerator='gpu',
                         devices=1,
                         precision=config.precision)

    # evaluate lightning module on development-test split
    test_dl = DataLoader(dataset=get_test_set(),
                         worker_init_fn=worker_init_fn,
                         num_workers=config.num_workers,
                         batch_size=config.batch_size)

    # get model complexity from nessi
    sample = next(iter(test_dl))[0][0].unsqueeze(0).to(pl_module.device)
    shape = pl_module.mel_forward(sample).size()
    macs, params = nessi.get_torch_size(pl_module.model, input_size=shape)

    print(f"Model Complexity: MACs: {macs}, Params: {params}")
    assert macs <= nessi.MAX_MACS, "The model exceeds the MACs limit and must not be submitted to the challenge!"
    assert params <= nessi.MAX_PARAMS_MEMORY, \
        "The model exceeds the parameter limit and must not be submitted to the challenge!"

    allowed_precision = int(nessi.MAX_PARAMS_MEMORY / params * 8)
    print(f"ATTENTION: According to the number of model parameters and the memory limits that apply in the challenge,"
          f" you are allowed to use at max the following precision for model parameters: {allowed_precision} bit.")

    # obtain and store details on model for reporting in the technical report
    info = {}
    info['MACs'] = macs
    info['Params'] = params
    res = trainer.test(pl_module, test_dl)
    info['test'] = res

    # generate predictions on evaluation set
    eval_dl = DataLoader(dataset=get_eval_set(),
                         worker_init_fn=worker_init_fn,
                         num_workers=config.num_workers,
                         batch_size=config.batch_size)

    predictions = trainer.predict(pl_module, dataloaders=eval_dl)
    # all filenames
    all_files = [item[len("audio/"):] for files, _ in predictions for item in files]
    # all predictions
    all_predictions = torch.cat([torch.as_tensor(p) for _, p in predictions], 0)
    all_predictions = F.softmax(all_predictions, dim=1)

    # write eval set predictions to csv file
    df = pd.read_csv(dataset_config['meta_csv'], sep="\t")
    le = preprocessing.LabelEncoder()
    le.fit_transform(df[['scene_label']].values.reshape(-1))
    class_names = le.classes_
    df = {'filename': all_files}
    scene_labels = [class_names[i] for i in torch.argmax(all_predictions, dim=1)]
    df['scene_label'] = scene_labels
    for i, label in enumerate(class_names):
        df[label] = all_predictions[:, i]
    df = pd.DataFrame(df)

    # save eval set predictions, model state_dict and info to output folder
    df.to_csv(os.path.join(out_dir, 'output.csv'), sep='\t', index=False)
    torch.save(pl_module.model.state_dict(), os.path.join(out_dir, "model_state_dict.pt"))
    with open(os.path.join(out_dir, "info.json"), "w") as json_file:
        json.dump(info, json_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DCASE 24 argument parser')

    # general
    parser.add_argument('--project_name', type=str, default="DCASE24_Task1")
    parser.add_argument('--experiment_name', type=str, default="Baseline")
    parser.add_argument('--num_workers', type=int, default=8)  # number of workers for dataloaders
    parser.add_argument('--precision', type=str, default="32")

    # evaluation
    parser.add_argument('--evaluate', action='store_true')  # predictions on eval set
    parser.add_argument('--ckpt_id', type=str, default=None)  # for loading trained model, corresponds to wandb id

    # teacher models
    parser.add_argument('--entity', type=str, required=True)
    parser.add_argument('--cpm_teachers_project', type=str, default="DCASE24_Task1") # single project or comma seperated wandb project names
    parser.add_argument('--cpm_teachers', type=str, default=None) # comma seperated wandb run ids
    parser.add_argument('--cpresnet_teachers_project', type=str, default="DCASE24_Task1") # single project or comma seperated wandb project names
    parser.add_argument('--cpresnet_teachers', type=str, default=None) # comma seperated wandb run ids
    parser.add_argument('--passt_teachers_project', type=str, default="DCASE24_Task1") # single project or comma seperated wandb project names
    parser.add_argument('--passt_teachers', type=str, default=None) # comma seperated wandb run ids

    parser.add_argument('--pretrained_student_project', type=str, default="DCASE24_Task1")
    parser.add_argument('--pretrained_student', type=str, default=None) # wandb id

    # passt model
    parser.add_argument('--arch', type=str, default='passt_s_swa_p16_128_ap476')  # pretrained passt model
    parser.add_argument('--input_fdim', type=int, default=128)
    parser.add_argument('--s_patchout_t', type=int, default=0)
    parser.add_argument('--s_patchout_f', type=int, default=6)

    # dataset
    # subset in {100, 50, 25, 10, 5}
    parser.add_argument('--subset', type=int, default=100)
    parser.add_argument('--orig_sample_rate', type=int, default=44100)

    # model
    parser.add_argument('--n_classes', type=int, default=10)  # classification model with 'n_classes' output neurons
    parser.add_argument('--in_channels', type=int, default=1)
    # adapt the complexity of the neural network (3 main dimensions to scale the baseline)
    parser.add_argument('--base_channels', type=int, default=32)
    parser.add_argument('--channels_multiplier', type=float, default=1.8)
    parser.add_argument('--expansion_rate', type=float, default=2.1)

    # wandb
    # upload the model to wandb model registry
    parser.add_argument('--log_model', action='store_true')

    # training
    parser.add_argument('--n_epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--mixstyle_p', type=float, default=0.6)  # frequency mixstyle
    parser.add_argument('--mixstyle_alpha', type=float, default=0.4)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--roll_sec', type=int, default=0.1)  # roll waveform over time
    parser.add_argument('--dir_p', type=float, default=0.6)  # prob. to apply device impulse response augmentation
    
    # filteraugment
    parser.add_argument('--filteraugment', action='store_true')
    parser.add_argument('--filt_aug_db_l', type=int, default=-6) # Lower bound of augmenting weights
    parser.add_argument('--filt_aug_db_h', type=int, default=6) # Upper bound of augmenting weights
    parser.add_argument('--filt_aug_n_band_l', type=int, default=3) # Lower bound of bins to augment
    parser.add_argument('--filt_aug_n_band_h', type=int, default=6) # Upper bound of bins to augment
    parser.add_argument('--filt_aug_min_bw', type=int, default=6) # distance between modified mel bins
    parser.add_argument('--filt_aug_type', type=str, default="linear") # either "linear" or "step"

    ## knowledge distillation
    parser.add_argument('--temperature', type=float, default=2.0)
    parser.add_argument('--kd_lambda', type=float, default=0.02)

    # peak learning rate (in cosinge schedule)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--warmup_steps', type=int, default=2000)

    # preprocessing
    parser.add_argument('--sample_rate', type=int, default=32000)
    parser.add_argument('--window_length', type=int, default=3072)  # in samples (corresponds to 96 ms)
    parser.add_argument('--hop_length', type=int, default=500)  # in samples (corresponds to ~16 ms)
    parser.add_argument('--n_fft', type=int, default=4096)  # length (points) of fft, e.g. 4096 point FFT
    parser.add_argument('--n_mels', type=int, default=256)  # number of mel bins
    parser.add_argument('--freqm', type=int, default=48)  # mask up to 'freqm' spectrogram bins
    parser.add_argument('--timem', type=int, default=0)  # mask up to 'timem' spectrogram frames
    parser.add_argument('--f_min', type=int, default=0)  # mel bins are created for freqs. between 'f_min' and 'f_max'
    parser.add_argument('--f_max', type=int, default=None)
 
    # preprocessing for passt model
    parser.add_argument('--sample_rate_p', type=int, default=32000)
    parser.add_argument('--window_length_p', type=int, default=800)  # in samples (corresponds to 96 ms)
    parser.add_argument('--hop_length_p', type=int, default=320)  # in samples (corresponds to ~16 ms)
    parser.add_argument('--n_fft_p', type=int, default=1024)  # length (points) of fft, e.g. 4096 point FFT
    parser.add_argument('--n_mels_p', type=int, default=128)  # number of mel bins
    parser.add_argument('--freqm_p', type=int, default=48)  # mask up to 'freqm' spectrogram bins
    parser.add_argument('--timem_p', type=int, default=0)  # mask up to 'timem' spectrogram frames
    parser.add_argument('--f_min_p', type=int, default=0)  # mel bins are created for freqs. between 'f_min' and 'f_max'
    parser.add_argument('--f_max_P', type=int, default=None)
    

    args = parser.parse_args()
    if args.evaluate:
        evaluate(args)
    else:
        train(args)
