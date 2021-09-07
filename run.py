from argparse import ArgumentParser
from pytorch_lightning import Trainer, seed_everything, callbacks
from pytorch_lightning.loggers import TensorBoardLogger

from pytorch_lightning.plugins import DDPPlugin
import torch
import os
from copy import deepcopy
import pandas as pd

from preprocessing import loadData, preprocess, representation
from preprocessing.helpers import boolean_string
from models.model_selection import ModelSelection

def main(args,selected_model):
    dict_args = vars(args)
    # seed for reproducibility
    seed_everything(42)

    # build the curve (if method==sfc, otherwise returns none)
    name_curve=args.name_curve
    curve =representation.choose_curve(name_curve,args.length,method=args.method)
    x_size,y_size=representation.find_dimensions(length=args.length,curve=curve,method=args.method,sr=args.sr)
    if args.method != "sfc":
        name_curve=""

    # preprocess the inputs
    preparation=preprocess.prepare_curve if args.method == "sfc" else preprocess.prepare_mfcc
    transformation=None
    # load the data
    train, val, test,KFold = loadData.load_data(curve=curve,preparation=preparation,transformation=transformation,
                                                folder="data",**dict_args)
    
    class_names=train.class_names
    num_classes=len(class_names)
    # load the model
    model=selected_model(**dict_args, class_names=class_names, x_size=x_size, y_size=y_size,num_classes=num_classes)

    # build the path where the .ckpt is saved
    path=os.path.join(args.dataset,args.model_name,args.method)
    if name_curve !="":
        path=os.path.join(path,name_curve)

    if args.use_fold:
        # cross-validation training
        results=[]
        path=os.path.join(path,args.output_file.split(".csv")[0])
        os.makedirs(os.path.join('results',path))
        # save the splits
        KFold.save_split(os.path.join('results',path,"split.pkl"))
        for i in range(args.K):
            if i>0:
                KFold.update_folds()
            # store best model
            checkpoint_callback = callbacks.ModelCheckpoint(
                monitor='val_accuracy',
                dirpath=os.path.join('results',path),
                filename='sc-{epoch:02d}-{val_accuracy:.3f}'+"fold_"+str(i+1),
                save_top_k=1,
                mode='max',
                verbose=True,
            )
            # defines early stopping rule
            callbacks_ = [checkpoint_callback, callbacks.EarlyStopping(
                "val_loss", min_delta=0, patience=dict_args["patience"])]
            logger = TensorBoardLogger("lightning_logs", name=path, default_hp_metric=False, log_graph=False,version=f'fold_{i + 1}')
            _model=deepcopy(model)
            # train the model 
            # due to pytorch limitations use "dp" (data parallel mode). We cannot call multiple times the .fit method in ddp mode.
            trainer=Trainer.from_argparse_args(args, logger=logger, auto_select_gpus=True,  terminate_on_nan=True, num_sanity_val_steps=0, accumulate_grad_batches=1,
                                            auto_scale_batch_size=None, callbacks=callbacks_, checkpoint_callback=True, profiler="simple",accelerator="dp",
                                            #limit_train_batches=2  # Debugging
                                            )
            trainer.fit(_model, train, val)
            # store the results 
            test_res=trainer.test(test_dataloaders=test, ckpt_path="best")[0]
            results.append(test_res)
            partial_res=pd.DataFrame([test_res])
            # save the partial results
            partial_res.to_csv(os.path.join('results',path,"fold_"+str(i+1)+args.output_file),index=False)
        # save all the results  
        res=pd.DataFrame(results)
        print("Mean and std are ",res.agg(["mean","std"]))
        res.to_csv(os.path.join('results',path,args.output_file),index=False)
    else:
        # store best model wrt. accurracy
        checkpoint_callback = callbacks.ModelCheckpoint(
            monitor='val_accuracy',
            dirpath=os.path.join('results',path),
            filename='sc-{epoch:02d}-{val_accuracy:.3f}',
            save_top_k=1,
            mode='max',
            verbose=True,
        )
        # defines early stopping rule
        callbacks_ = [checkpoint_callback, callbacks.EarlyStopping("val_loss",
                                                 min_delta=0, patience=dict_args["patience"])]
        logger = TensorBoardLogger("lightning_logs", name=path, default_hp_metric=False, log_graph=True)

        # train the model
        trainer = Trainer.from_argparse_args(args, logger=logger, auto_select_gpus=True,  terminate_on_nan=True, num_sanity_val_steps=0, accumulate_grad_batches=1,
                                            auto_scale_batch_size=None, callbacks=callbacks_, checkpoint_callback=True, profiler="simple",accelerator="ddp", plugins=DDPPlugin(find_unused_parameters=False),
                                            #limit_train_batches=2  # Debugging
                                            )
        trainer.fit(model, train, val)
        # test the model on the validation set
        trainer.test(test_dataloaders=val, ckpt_path="best")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gpus', default=-1)
    parser.add_argument('--batch_size', default=256,type=int)
    # figure out which model to use
    parser.add_argument('--model_name', type=str,
                        default='mobilenetv3', help='Either mobilenetv3 or res8 or ...')
    parser.add_argument('--dataset', type=str,
                        default='speechcommands', help='Either speechcommands or ...')
    parser.add_argument('--method', type=str,
                        default='sfc', help='Either sfc or mfcc')
    parser.add_argument('--name_curve', type=str,
                        default='Hilbert', help='Hilbert/Z/Gray/OptR/H/Sweep/Scan/Diagonal')
    parser.add_argument('--length', default=16000, type=int)

    # training parameters
    parser.add_argument('--max_epochs', type=int, default=300)
    parser.add_argument('--deterministic', type=boolean_string, default=True)
    parser.add_argument('--track_grad_norm', type=int, default=2)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--sr', default=16000, type=int)

    # k fold training
    parser.add_argument("--use_fold",default=None,type=boolean_string)
    parser.add_argument("--K",type=int,default=10)
    parser.add_argument("--output_file",default="KFold.csv",type=str)

    # preprocessing
    parser.add_argument('--center', default=False, type=boolean_string)
    parser.add_argument('--shifting', default=False, type=boolean_string)

    # mixup
    parser.add_argument('--mixup', default=False, type=boolean_string)
    parser.add_argument('--alpha', default=0.2, type=float)

    # parse temporary arguments
    temp_args, _ = parser.parse_known_args()

    # let the model add what it wants
    selected_model=ModelSelection(temp_args.model_name)
    parser=selected_model.parse_params(parser)

    args = parser.parse_args()
    main(args,selected_model)
