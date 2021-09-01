from argparse import ArgumentParser
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from preprocessing import loadData, preprocess, representation
import torch
import os
from preprocessing.helpers import boolean_string
from models.model_selection import ModelSelection


def main(args,selected_model):
    dict_args = vars(args)
    # seed for reproducibility
    seed_everything(42)

    name_curve=args.name_curve
    # build the curve (if method=sfc otherwise return none)
    curve =representation.choose_curve(name_curve,args.length,method=args.method)
    x_size,y_size=representation.find_dimensions(length=args.length,curve=curve,method=args.method,sr=args.sr)
    if args.method != "sfc":
        name_curve=""

    # select the preprocessing method
    preparation=preprocess.prepare_curve if dict_args["method"] == "sfc" else preprocess.prepare_mfcc
    transformation=None

    # load the data
    train, val, test, KFold = loadData.load_data(curve=curve,preparation=preparation,transformation=transformation,
                                use_fold=None,folder="data",small=False,**dict_args)

    class_names=train.class_names
    num_classes=len(class_names)
    filename = dict_args.get("filename")
    if filename is None:
        raise ValueError("Unknown file name")
    print(f"Processing filename: {filename}")
    # build the path to the .ckpt file
    relative_path=os.path.join(args.dataset,args.model_name,args.method)
    if args.method!="mfcc":
        relative_path=os.path.join(relative_path,name_curve)

    # load the model
    model=selected_model.load(os.path.join("results",relative_path,filename),
                            class_names=class_names,num_classes=num_classes,**dict_args)

    # build the trainer of the model
    logger = TensorBoardLogger("results", name=relative_path, default_hp_metric=False, log_graph=True)
    trainer = Trainer.from_argparse_args(args, logger=logger,
                                         auto_select_gpus=True,  terminate_on_nan=True, num_sanity_val_steps=0,
                                         auto_scale_batch_size=None, auto_lr_find=False, checkpoint_callback=False, profiler="simple",
                                         # limit_train_batches=2  # Debugging
                                         )
    # test the model (use the validation set by default)
    if dict_args["use_test"]:
        res=trainer.test(model, test)
    else:
        res=trainer.test(model, val)
    return res[0]


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gpus', default=1)
    parser.add_argument('--batch_size', default=256,type=int)
    # figure out which model to use
    parser.add_argument('--dataset', type=str,
                        default='speechcommands', help='Either speechcommands or ...')
    parser.add_argument('--model_name', type=str,
                        default='mobilenetv3', help='Either mobilenetv3 or ...')
    parser.add_argument('--method', type=str,
                        default='sfc', help='Either sfc or mfcc')
    parser.add_argument('--name_curve', type=str,
                        default='Hilbert', help='Hilbert/Z/Gray/OptR/H/Sweep/Scan/Diagonal')
    parser.add_argument('--length', default=16000, type=int)

    parser.add_argument('--hparams_file', type=str, default=None)
    parser.add_argument('--filename', type=str, default=None)

    # preprocessing
    parser.add_argument('--center', default=False, type=boolean_string)

    # the sampling rate
    parser.add_argument('--sr', default=16000, type=int)

    # use the test set only at the end
    parser.add_argument('--use_test', type=boolean_string, default=False,
                        help="Either use test or validation set (True=test)")


    # parse temporary arguments
    temp_args, _ = parser.parse_known_args()

    # let the model add what it wants
    selected_model=ModelSelection(temp_args.model_name)
    parser=selected_model.parse_params(parser)

    args = parser.parse_args()
    main(args,selected_model)
