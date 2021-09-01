# Audio representation using space filling curves

## Setup
<a name="setup"></a>

Packages needed:
* torch, torchaudio, torchvision, PIL
* torchmetrics
* scipy, numpy, matplotlib, seaborn, pandas
* pytorch-lightning
* librosa, soundfile
* tqdm
* h5py
* nltk
* gmpy2

To set up the environment make sure that conda (or Miniconda) is installed (see [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) for the installation).
Open a terminal and set the current directory to the location of this file.

> Use `cd` to change directory and `pwd` to print the path to the working directory.

Then create a specific environment using
```
conda env create -f requirements.yml
```
and activate the environment using
```
conda activate test
```
> You can use another name than `test` by changing the requirements.yml file. You can remove the environment by calling ```conda remove --name test --all```

## Directory structure

The code of the models is in the `models` folder. All models extend the class `BaseModel` defined in the `torch_model.py` file by redefining the forward pass (`_forward_impl` method). The `BaseModel` class implements a pytorch lightning module and defines the training/validation/test loops (mixup is defined here) and the optimization routine.

The `preprocessing` folder contains all code relative to the data downloading+loading and data preprocessing (including the computation of audio representations such as MFCCs, SFC mappings). The `downloadData.py` file downloads and extracts the data in the `data` folder, if the files are not found (Note that `data/LibriSpeech/word_labels` folder must be downloaded manually [here](https://imperialcollegelondon.app.box.com/s/yd541e9qsmctknaj6ggj5k2cnb4mabkc?page=1)). `LibriSpeech.py` and `SpeechCommands.py` implement classes to load the corresponding dataset. `loadData.py` offers a simpler method to load the dataset as a torch.Tensor object or as a torch.utils.data.DataLoader object. The `KFold` class implemented in `Ksplit.py` splits the data into K folds and updates the training/validation set in the case of cross-validation each time the `update_folds` method is called. 

The `preprocess.py` file implements the preprocessing steps for the MFCC and the space filling curve approach (including random shifts+centering). The `representation.py` file implements the space filling curve mapping using mainly L-systems. All curves extend the abstract `Curve` class and implements the `build_curve` method.

> The curves that are currently implemented are: Hilbert, Z, Gray, H, OptimalRSFC, Scan, Sweep, Diagonal.

The data itself (not preprocessed) is contained in the `data` folder, where each dataset (uncompress) is contained in a folder according to its name (the data should appear here once the code is run). The logs of the models are saved in the `lightning_logs` folder according to the following tree structure
```
    ├──lightning_logs
        ├── librispeech              # dataset name
        │   ├── efficientnet         # model name
        │   │   ├── mfcc             # method name
        │   │   ├── sfc              # method name
        │   │   │   ├── Hilbert      # curve name
        │   │   │   ├── Z
        │   │   │   └── ...  
        │   ├── mixnet               # model name
        │   │   ├── mfcc            
        │   │   ├── sfc  
        │   │   │   ├── Hilbert
        │   │   │   ├── Z
        │   │   │   └── ...  
        │   ├── mobilenetv3          # model name
        │   └── ...                  # etc.
        └── ...
```
The results and weights of the models (*.ckpt files) are saved using the similar tree structure in the `results` folder.
The jupyter notebook `audio_repr.ipynb` contains the code to produce the images and the figures are saved in the `figure` folder.

## Training
<a name="training"></a>

To train the model, simply execute the `run.py` file. For example,

```
python run.py --lr 0.5 --pretrained True --name_curve Hilbert
```

The commands are

* Training params:
    * `gpus`: number of GPUs (default all)
    * `batch_size`: batch size (default 256)
    * `sgd`: whether to use sgd or AdamW (default True)
    * `lr`: learning rate (default 0.005)
    * `weight_decay`: intensity of the weight decay (default 0.0)
    * `momentum`: specify the momentum of the optimization routine (as defined in torch.optim) (default 0.0)d
    * `max_epochs`: maximum number of epochs before quitting (default 300)
    * `deterministic`: enables cudnn.deterministic. (by default True for reproducibility)
    * `track_grad_norm`: track $l_p$ norm of the gradient (default p=2, set -1 for no tracking)
    * `patience`: number of waiting epochs of the early stopping callback (default 10)
* Model params:
    * `model_name`: name of the model: mobilenetv3 (default), res8, squeezenet, shufflenet, mixnet, efficientnet
    * `method`: either mfcc or sfc (i.e. space filling curve, default)
    * `name_curve`: name of the curve, either Hilbert (default), Z, Gray, OptR, H, Peano, Sweep, Scan, Diagonal
    * `width_mult`: expand or reduce the number of channels of the model by the factor `width_mult` (default 1.0, must be positive)    
* Data params:
    * `dataset`: the name of the dataset, either speechcommands (default), librispeech, urban8k, ESC50
    * `length`: the length of the input samples (default = 16000)
    * `sr`: the sampling rate (default=16000)
    * `center`: center the audio signal in the middle of the sample (default False)
    * `shifting`: data augmentation by randomly shifting in time the audio clips (default False)
    * `mixup`: use mixup data augmentation (deault false)
    * `alpha`: hyperparameter of the beta distribution of the mixing parameter of mixup (default 0.2)
* $K$ fold training:
    * `use_fold`: if true, split the data into $K$ folds, specified by the argument `K` and use K-1 folds for training and 1 for testing.
    * `output_file`: *.csv file to store the output of the cross-validation (default KFold.csv)
    
Additional model specific parameters can be tuned. You can see them in the `add_model_specific_args` method of the corresponding model. We always use the default parameters. 

The training checkpoints are stored in lightning_logs/[model_name]/[method]/[name_curve]

Possibility to examine the training steps using tensorboard. Open a bash terminal and enter

```tensorboard --logdir lightning_logs/[model]/[method]/[name_curve]```

The best model is stored in results/[model_name]/[method]/[name_curve]/sc-epoch=...val_accuracy=...ckpt

## Evaluation
<a name="evaluation"></a>

To evaluate the model on the validation set or on the test set, run `evaluate.py`. For example,

```
python run.py --filename sc-epoch=10-val_accuracy=0.800.ckpt --name_curve Hilbert
```

The possible commands are:

* Training params:
    * `gpus`: number of GPUs (default 1)
    * `batch_size`: batch size (default 256)
* Model params:
    * `filename`: mandatory checkpoint file
    * `use_test`: whether to evaluate the model on the test or the validation set (default False, i.e. validation set)
    * `hparams_file`: hyperparameters file of the model (optional, default None)
    * `model_name`: name of the model (see [training section](#training))
    * `method`: either mfcc or sfc (i.e. space filling curve, default)
    * `name_curve`: name of the curve (see [training section](#training))
    * `width_mult`: the width multiplier (see [training section](#training))
* Input params:
    * `center`: center the audio signal in the middle of the sample (default False)
* Data params:
    * `dataset`: the name of the dataset (see [training section](#training))
    * `length`: the length of the input samples (default = 16000)
    * `sr`: the sampling rate (default=16000)

The commands should be the same as the ones used for the `run.py` file.


*Authors*: Alessandro Mari, Arash Salarian
