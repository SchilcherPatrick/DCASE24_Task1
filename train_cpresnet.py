import lightning
from lightning.pytorch.loggers import WandbLogger
import torch
import torchaudio
from torch.utils.data import DataLoader
import argparse
import torch.nn.functional as F
import transformers
import wandb
import json

from dataset.dcase24 import get_training_set, get_test_set, get_eval_set
from helpers.init import worker_init_fn
from models.cp_resnet import get_model
from helpers.utils import mixstyle, filteraugment
from helpers.checkpoints import get_state_dict
from helpers import nessi


class PLModule(lightning.LightningModule):
    def __init__(self, config):
        super().__init__()
        # results from argparse, contains all configurations for our experiment
        self.config = config

        # module for resampling waveforms on the fly
        resample = torchaudio.transforms.Resample(
            orig_freq=self.config.orig_sample_rate,
            new_freq=self.config.sample_rate
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

        freqm = torchaudio.transforms.FrequencyMasking(config.freqm, iid_masks=True)
        timem = torchaudio.transforms.TimeMasking(config.timem, iid_masks=True)

        self.mel = torch.nn.Sequential(
            resample,
            mel
        )

        self.mel_augment = torch.nn.Sequential(
            freqm,
            timem
        )

        # setup cp-resnet model
        self.model = get_model(base_channels=config.base_channels)

        # Load pretrained cp-resnet model
        if config.pretrained_ckpt is not None:
            self.model.load_state_dict(get_state_dict(config.pretrained_entity,
                                                      config.pretrained_project, 
                                                      config.pretrained_ckpt))

        self.device_ids = ['a', 'b', 'c', 's1', 's2', 's3', 's4', 's5', 's6']
        self.label_ids = ['airport', 'bus', 'metro', 'metro_station', 
                          'park', 'public_square', 'shopping_mall',
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
        :param x: batch of raw audio signals (waveforms)
        :return: log mel spectrogram
        """
        x = self.mel(x)
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
        This is the way pytorch lightening requires optimizers and 
        learning rate schedulers to be defined.
        The specified items are used automatically in the optimization 
        loop (no need to call optimizer.step() yourself).

        :return: optimizer and learning rate scheduler
        """

        optimizer = torch.optim.AdamW(self.parameters(), 
                                      lr=self.config.lr, 
                                      weight_decay=self.config.weight_decay)
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

    def training_step(self, train_batch, batch_idx):
        """
        :param train_batch: contains one batch from train dataloader
        :param batch_idx
        :return: loss to update model parameters
        """
        x, files, labels, devices, cities = train_batch
        # we convert the raw audio signals into log mel spectrograms
        x = self.mel_forward(x)  

        # frequency mixstyle
        if self.config.mixstyle_p > 0:
            x = mixstyle(x, self.config.mixstyle_p, self.config.mixstyle_alpha)

        # filteraugment
        if self.config.filteraugment:
            x = filteraugment(features=x, 
                              db_range=[self.config.filt_aug_db_l, 
                                        self.config.filt_aug_db_h],
                              n_band = [self.config.filt_aug_n_band_l, 
                                        self.config.filt_aug_n_band_h], 
                              min_bw=self.config.filt_aug_min_bw, 
                              filter_type=self.config.filt_aug_type ) 

        y_hat = self.model(x)
        samples_loss = F.cross_entropy(y_hat, labels.long(), reduction="none")
        loss = samples_loss.mean()

        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'])
        self.log("epoch", self.current_epoch)
        self.log("train/loss", loss.detach().cpu())
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

        x = self.mel_forward(x)
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

        x = self.mel_forward(x)
        y_hat = self.model(x)

        return files, y_hat


def train(config):
    if config.pretrained_ckpt:
        # init wandb before WandbLogger to log used artifacts in init of PLModule
        # run will be reused by WandbLogger
        run = wandb.init(
                    project=config.project_name,
                    tags=["DCASE24", "CPResNet"],
                    config=config,  # this logs all hyperparameters for us
                    name=config.experiment_name
                )

    wandb_logger = WandbLogger(
        project=config.project_name,
        tags=["DCASE24", "CPResNet"],
        config=config,  # this logs all hyperparameters for us
        name=config.experiment_name,
        log_model=config.log_model
    )

    # train dataloader
    assert config.subset in {100, 50, 25, 10, 5}, "Specify an integer value in: {100, 50, 25, 10, 5} to use one of " \
                                                  "the given subsets."
    roll_samples = config.orig_sample_rate * config.roll_sec
    train_ds = get_training_set(config.subset, 
                                roll=roll_samples, 
                                dir_p=config.dir_p, 
                                resample_rate=config.sample_rate)
    train_dl = DataLoader(dataset=train_ds,
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
    shape = pl_module.cpu().mel_forward(sample).size()
    macs, params = nessi.get_torch_size(pl_module.model, input_size=shape)
    # log MACs and number of parameters for our model
    wandb_logger.experiment.config['MACs'] = macs
    wandb_logger.experiment.config['Parameters'] = params

    # create the pytorch lightening trainer by specifying the number of epochs to train, the logger,
    # on which kind of device(s) to train and possible callbacks
    trainer = lightning.Trainer(max_epochs=config.n_epochs,
                         logger=wandb_logger,
                         accelerator='auto',
                         devices=1,
                         precision=config.precision,
                         callbacks=[lightning.pytorch.callbacks.ModelCheckpoint(save_last=True)])
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
    trainer = lightning.Trainer(logger=False,
                         accelerator='auto',
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
    parser.add_argument('--experiment_name', type=str, default="CPResNet")
    parser.add_argument('--num_workers', type=int, default=8)  # number of workers for dataloaders
    parser.add_argument('--precision', type=str, default="32")

    # evaluation
    parser.add_argument('--evaluate', action='store_true')  # predictions on eval set
    parser.add_argument('--ckpt_id', type=str, default=None)  # for loading trained model, corresponds to wandb id

    # dataset
    # subset in {100, 50, 25, 10, 5}
    parser.add_argument('--subset', type=int, default=100)
    parser.add_argument('--orig_sample_rate', type=int, default=44100)

    # model
    parser.add_argument('--n_classes', type=int, default=10)  # classification model with 'n_classes' output neurons
    parser.add_argument('--in_channels', type=int, default=1)
    # adapt the complexity of the neural network (3 main dimensions to scale the baseline)
    parser.add_argument('--base_channels', type=int, default=32)

    # wandb
    # upload the model to wandb model registry
    parser.add_argument('--log_model', action='store_true')

    parser.add_argument('--pretrained_ckpt', type=str, default=None)  # wandb checkpoint id of audioset pretrained model
    parser.add_argument('--pretrained_entity', type=str, default="DCASE24") # wandb entity the pretrained model is stored in
    # wandb project to which the pretrained model is assigned to
    # defaults to "project_name" if not specified
    parser.add_argument('--pretrained_project', type=str, default=None) 

    # training
    parser.add_argument('--n_epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--mixstyle_p', type=float, default=0.7)  # frequency mixstyle
    parser.add_argument('--mixstyle_alpha', type=float, default=0.6)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--roll_sec', type=float, default=0.1)  # roll waveform over time
    parser.add_argument('--dir_p', type=float, default=0.7)  # prob. to apply device impulse response augmentation

    # filteraugment
    parser.add_argument('--filteraugment', action='store_true')
    parser.add_argument('--filt_aug_db_l', type=int, default=-6) # Lower bound of augmenting weights
    parser.add_argument('--filt_aug_db_h', type=int, default=6) # Upper bound of augmenting weights
    parser.add_argument('--filt_aug_n_band_l', type=int, default=3) # Lower bound of bins to augment
    parser.add_argument('--filt_aug_n_band_h', type=int, default=6) # Upper bound of bins to augment
    parser.add_argument('--filt_aug_min_bw', type=int, default=6) # distance between modified mel bins
    parser.add_argument('--filt_aug_type', type=str, default="linear") # either "linear" or "step"

    # peak learning rate (in cosinge schedule)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--warmup_steps', type=int, default=2000)

    # preprocessing
    parser.add_argument('--sample_rate', type=int, default=32000)
    parser.add_argument('--window_length', type=int, default=3072)  # in samples (corresponds to 96 ms)
    parser.add_argument('--hop_length', type=int, default=750)  # in samples (corresponds to ~23 ms)
    parser.add_argument('--n_fft', type=int, default=4096)  # length (points) of fft, e.g. 4096 point FFT
    parser.add_argument('--n_mels', type=int, default=256)  # number of mel bins
    parser.add_argument('--freqm', type=int, default=48)  # mask up to 'freqm' spectrogram bins
    parser.add_argument('--timem', type=int, default=0)  # mask up to 'timem' spectrogram frames
    parser.add_argument('--f_min', type=int, default=0)  # mel bins are created for freqs. between 'f_min' and 'f_max'
    parser.add_argument('--f_max', type=int, default=None)

    args = parser.parse_args()

    if args.pretrained_project is None:
        args.pretrained_project = args.project_name

    if args.evaluate:
        evaluate(args)
    else:
        train(args)