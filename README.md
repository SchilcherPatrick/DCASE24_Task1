# DCASE24 Task1
This project is mainly based on the [DCASE24 Task1 Baseline](https://github.com/CPJKU/dcase2024_task1_baseline)
## Usage instructions
### Setup the Python environment
Create and activate a [conda](https://docs.anaconda.com/free/miniconda/index.html) environment:

```
conda create -n <env-name> -f environment.yml python=3.11
conda activate <env-name>
```


### Download the dataset

1. Download the dataset from [here](https://zenodo.org/records/6337421).
2. Create a new directory in the *dataset* directory called *dataset*.
3. Extract the files and copy the *audio* folder and *meta.csv* to the new directory *dataset/dataset/*.
4. Furthermore, in order to make DIR augmentation work, download the audio files from [here](https://github.com/fschmid56/cpjku_dcase23/tree/main/datasets/dirs), create a folder *dataset/dataset/dirs* and move the files into it.
     
Optional:

5. Create a new folder in the *dataset/dataset* directory called *split_setup*.
6. Download split files from [here](https://github.com/CPJKU/dcase2024_task1_baseline/releases/tag/files) and extract them into newly created *split_setup* directory.

The final folder structure in the *dataset* directory should look like this:
```
dataset
├───dataset
│   ├───audio
|   |   ├───airport-barcelona-0-0-0-a.wav
|   |   ├───airport-barcelona-0-0-1-a.wav
|   |   └───...
│   ├───split_setup
|   |   ├───split5.csv
|   |   ├───split10.csv
|   |   ├───split25.csv
|   |   ├───split50.csv
|   |   ├───split100.csv
|   |   └───test100.csv
|   └───meta.csv
└───dcase24.py
```

## Training

### Training models
To train the different types of models the corresponding training files *training_cpm.py*, *training_cpresnet.py* and *training_passt.py* are provided. 
All files share the arguments below if not specified otherwise, with some varying default values.

#### General
| Parameter | Value-Type | Default value | Description |
|-----------|------------|---------------|-------------|
| `--project_name` | str | DCASE24_Task1 | The Weights & Biases project this run belongs to. |
| `--experiment_name` | str | Baseline, CPResNet, PaSST | The experiment name to be used in Weights & Biases |
| `--num_workers` | int | 8 | number of workers for dataloaders |
| `--precision` | str | 32 | precision of the float datatype used |

#### Evaluation
| Parameter | Value-Type | Default value | Description |
|-----------|------------|---------------|-------------|
| `--evaluate` | bool | false | run predictions on eval set |
| `--ckpt_id` | str | None | for loading trained model, corresponds to wandb checkpoint id |

#### Dataset
| Parameter | Value-Type | Default value | Description |
|-----------|------------|---------------|-------------|
| `--orig_sample_rate` | int | 44100 | Sample rate of the original audio recordings |
| `--subset` | int | 100 | Dataset split to be used for training. Can be either 100, 50, 25, 10 or 5 |

#### Model 
| Parameter | Value-Type | Default value | Description |
|-----------|------------|---------------|-------------|
| `--n_classes` | int | 10 | classification model with 'n_classes' output neurons |
| `--in_channels` | int | 1 | classification model with 'in_channels' input channels |
| `--base_channels` | int | 32 |  |
| `--channels_multiplier` | float | 1.8 | multiplier for the amount of channels for each stage (CPM only) |
| `--expansion_rate` | float | 2.1 | expansion rate of inverted bottleneck in CPM Blocks (CPM only) |
| `--arch` | str | passt_s_swa_p16_128_ap476 | PaSST only |
| `--input_fdim` | int | 128 | PaSST only |
| `--s_patchout_t` | int | 0 | PaSST only |
| `--s_patchout_f` | int | 6 | PaSST only |
| `--pretrained` | bool | false | PaSST only |


#### W&B
| Parameter | Value-Type | Default value | Description |
|-----------|------------|---------------|-------------|
| `--log_model` | bool | false | Upload the trained model to W&B model registry |
| `--pretrained_ckpt` | str | None | W&B checkpoint id of pretrained model (CPM and CPResNet only)  |
| `--pretrained_entity` | str | "DCASE24" |  W&B entity/team the pretrained model is stored in (CPM and CPResNet only) |
| `--pretrained_project` | str | None |  W&B project the checkpoint run is assigned to. Defaults to project_name (CPM and CPResNet only) |


#### Training
| Parameter | Value-Type | Default value | Description |
|-----------|------------|---------------|-------------|
| `--n_epochs` | int | 150 |  |
| `--batch_size` | int | 256 |  |
| `--mixstyle_p` | float | 0.7 | frequency mixstyle |
| `--mixstyle_alpha` | float | 0.6 |  |
| `--weight_decay` | float | 0.0001, 0.001 | |
| `--roll_sec` | float | 0.1 | roll waveform over time |
| `--dir_p` | flaot | 0.7 | Probability of a DIR augmentation being applied |

#### Filter-Augmentation
| Parameter | Value-Type | Default value | Description |
|-----------|------------|---------------|-------------|
| `--filteraugment` | bool | false | use filter-augmention |
| `--filt_aug_db_l` | int | -6 | lower bound of augmentation weights |
| `--filt_aug_db_h` | int | 6 | upper bound of augmentation weights |
| `--filt_aug_n_band_l` | int | 3 | lower bound of bands to augment |
| `--filt_aug_n_band_h` | int | 6 | upper bound of bands to augment |
| `--filt_aug_min_bw` | int | 6 | minimal distance between two augmented bands |
| `--filt_aug_type` | str | "linear" | Possible values: "linear", "step" |


#### peak learning rate (in cosinge schedule)
| Parameter | Value-Type | Default value | Description |
|-----------|------------|---------------|-------------|
| `--lr` | float | 0.005, 0.001, 0.00001 | learning rate |
| `--warmup_steps` | int | 2000 |  |

#### Preprocessing
| Parameter | Value-Type | Default value | Description |
|-----------|------------|---------------|-------------|
| `--sample_rate` | int | 32000 |  |
| `--window_length` | int | 3072, 800 | in samples (corresponds to 96/25 ms) |
| `--hop_length` | int | 500, 320 | in samples (corresponds to ~16/10 ms) |
| `--n_fft` | int | 4096, 1024 | length (points) of fft, e.g. 4096/1024 point FFT |
| `--n_mels` | int | 256, 128 | number of mel bins |
| `--freqm` | int | 48 | mask up to 'freqm' spectrogram bins |
| `--timem` | int | 0 | mask up to 'timem' spectrogram frames |
| `--f_min` | int | 0 | mel bins are created for freqs. between 'f_min' and 'f_max' |
| `--f_max` | int | None |  |

### Knowledge Distillation
For knowledge distillation the following files are provided: 
- *kd_mean.py*: The average teacher logits are used
- *kd_bea.py*:  Teacher logits get ensembled using BEA
- *kd_mixed.py*: CPM and CPResNet teachers get ensembled using BEA. This ensemble and the PaSST techer logits get averaged. 
The same common arguments from the training scripts are used, including the PaSST-specific arguments. With additional parameters specified below:

#### Teacher Models
| Parameter | Value-Type | Default value | Description |
|-----------|------------|---------------|-------------|
| `--temperature` | float | 2.0 |  |
| `--kd_lambda` | float | 0.02 |  |


#### Teacher Models
| Parameter | Value-Type | Default value | Description |
|-----------|------------|---------------|-------------|
| `--entity` | str | None | REQUIRED. W&B entity/team the models are stored in |
| `--cpm_teachers_project` | str | DCASE24_Task1 | single project or comma seperated wandb project names the CPM model runs belong to |
| `--cpm_teachers` | str | None | comma seperated wandb run ids of CPM Teachers |
| `--cpresnet_teachers_project` | str | DCASE24_Task1 | single project or comma seperated wandb project names the CPResNEt model runs belong to |
| `--cpresnet_teachers` | str | None | comma seperated wandb run ids of CPResNet Teachers |
| `--passt_teachers_project` | str | DCASE24_Task1 | single project or comma seperated wandb project names the PaSST model runs belong to |
| `--passt_teachers` | str | None | comma seperated wandb run ids of PaSST Teachers |

#### Pretrained Student Model
| Parameter | Value-Type | Default value | Description |
|-----------|------------|---------------|-------------|
| `--pretrained_student_project` | str | DCASE24_Task1 | W&B project name the pretrained student model run belongs to |
| `--pretrained_student` | str | None | W&B run id of pretrained student model |

#### PaSST Preprocessing 
| Parameter | Value-Type | Default value | Description |
|-----------|------------|---------------|-------------|
| `--sample_rate_p` | int | 32000 |  |
| `--window_length_p` | int |  800 | in samples (corresponds to 25 ms) |
| `--hop_length_p` | int | 320 | in samples (corresponds to ~10 ms) |
| `--n_fft_p` | int | 1024 | length (points) of fft, e.g. 1024 point FFT |
| `--n_mels_p` | int | 128 | number of mel bins |
| `--freqm_p` | int | 48 | mask up to 'freqm' spectrogram bins |
| `--timem_p` | int | 0 | mask up to 'timem' spectrogram frames |
| `--f_min_p` | int | 0 | mel bins are created for freqs. between 'f_min' and 'f_max' |
| `--f_max_p` | int | None |  |
