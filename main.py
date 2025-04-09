from typing import final
import arguments

import time
import os
import torch
from torch.optim import AdamW, SGD
from torch_geometric.loader import RandomNodeLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import accuracy_score, classification_report, balanced_accuracy_score, roc_auc_score

# from utils import DirichletLoss
# from utils.trainers import model_train, model_predict,TrustWI_wo_train,TrustWI_wo_predict
from utils.trainers_dirichlet import TrustWI_train, TrustWI_predict
from utils import SEPARATOR
from utils import get_path_of_all
# from utils.utils import create_directory
from utils.utils import ce_loss
from utils.utils import setup_logger
from utils.model import TrustWI
from utils.dataset import GraphDataset


args = arguments.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.dataset_name == 'DoHBrw':
    MAPPING: final = {
    "Benign": 0,
    "Malicious": 1,
    }
elif args.dataset_name == 'TONIoT':
    MAPPING: final = {
        "normal": 0,
        "backdoor": 1,
        "ddos": 2,
        "dos": 3,
        "injection": 4,
        "mitm": 5,
        "password": 6,
        "ransomware": 7,
        "scanning": 8,
        "xss": 9
    }
elif args.dataset_name == 'CICIDS':
    MAPPING: final = {
    'BENIGN': 0, 
    'DoS GoldenEye': 1,
    'PortScan': 2,
    'DoS Slowhttptest': 3,
    'Web Attack  Brute Force': 4,
    'Bot': 5,
    'Web Attack  Sql Injection': 6,
    'Web Attack  XSS': 7,
    'Infiltration': 8,
    'DDoS': 9,
    'DoS slowloris': 10,
    'Heartbleed': 11,
    'FTP-Patator': 12,
    'DoS Hulk': 13,
    "SSH-Patator": 14,
    }

    
def get_parameter_number(model, logger):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Model size: {trainable_params/(1024*1024):.3f} M')

def main(dataset_name, model_name):
    # Get path of all file
    dataset_path = os.path.join(os.getcwd(), 'datasets', dataset_name)
    
    log_path, log_file_path, log_train_path, model_path, result_path, \
        confusion_matrix_path, detection_rate_path = get_path_of_all(
            dataset_name, small=args.n_small,num_neigh=args.n_neigh, num_node=args.n_node, num_edge=args.n_edge, alpha=args.alpha, hid=args.hid, pro_T=args.pro_T, model_name=model_name,v=args.v)
    # Init logger
    logger = setup_logger('logger', log_file_path)
    train_logger = setup_logger('training', log_train_path)

    # Create training, val and test dataset
    n_classes = len(MAPPING)
    train_dataset = GraphDataset(
        root=dataset_path,
        dataset_name=dataset_name,
        small=args.n_small,
        num_neighbors=args.n_neigh,
        pro_T=args.pro_T,
        knn_neigh=args.knn_neigh, 
        knn_metric=args.knn_metric,
        alpha = args.alpha,
        weakNodes=args.n_node,
        weakEdges=args.n_edge,
        binary=args.binary,
        augmentation=args.augmentation,
        val=False,
        test=False
    )
    val_dataset = GraphDataset(
        root=dataset_path,
        dataset_name=dataset_name,
        small=args.n_small,
        num_neighbors=args.n_neigh,
        pro_T=args.pro_T,
        knn_neigh=args.knn_neigh, 
        knn_metric=args.knn_metric,
        alpha = args.alpha,
        weakNodes=args.n_node,
        weakEdges=args.n_edge,
        binary=args.binary,
        val=True,
        test=False
    )
    test_dataset = GraphDataset(
        root=dataset_path,
        dataset_name=dataset_name,
        small=args.n_small,
        num_neighbors=args.n_neigh,
        pro_T=args.pro_T,
        knn_neigh=args.knn_neigh, 
        knn_metric=args.knn_metric,
        alpha = args.alpha,
        weakNodes=args.n_node,
        weakEdges=args.n_edge,
        binary=args.binary,
        val=False,
        test=True
    )
    logger.info("-" * 35 + "Start!" + "-" * 35)
    logger.info('Time: {}'.format(time.strftime('%Y-%m-%d-%H-%M')))
    logger.info(f'Number of features: {train_dataset.num_features}')
    logger.info(f'Number of classes: {n_classes}')
    logger.info(SEPARATOR)

    # Define train, val and test loader
    train_data = train_dataset[0]
    val_data = val_dataset[0]
    test_data = test_dataset[0]

    # Log info of train, val and test
    logger.info(train_data)
    logger.info(val_data)
    logger.info(test_data)
    logger.info(SEPARATOR)

    # Define train, val and test loader
    train_loader = RandomNodeLoader(train_data, num_parts=args.n_parts)
    val_loader = RandomNodeLoader(val_data, num_parts=args.n_parts)
    test_loader = RandomNodeLoader(test_data, num_parts=args.n_parts)

    # Define model
    logger.info(f'N. Convs: {args.pro_T}')
    logger.info(f'N. Hidden Channels: {args.hid}')
    logger.info(f'N. Num Parts: {args.n_parts}')
    logger.info(SEPARATOR)

    train_logger.info("-" * 35 + "Start!" + "-" * 35)
    train_logger.info('Time: {}'.format(time.strftime('%Y-%m-%d-%H-%M')))
    train_logger.info(f'N. Convs: {args.pro_T}')
    train_logger.info(f'N. Hidden Channels: {args.hid}')
    train_logger.info(f'N. Num Parts: {args.n_parts}')
    train_logger.info(SEPARATOR)

    model = TrustWI(
        dataset=train_data,
        num_classes=n_classes,
        hid=args.hid,
        dropout=0.5
    )

    model.to(device)
    logger.info(model)
    logger.info(get_parameter_number(model, logger))
    

    criterion = ce_loss

    '''initiate criteria and optimizer'''
    if model_name == 'attention' and dataset_name!='TONIoT':
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    elif model_name != 'gcn' and (dataset_name=='TONIoT' or dataset_name=='CICIDS2017'):
        optimizer = AdamW(model.parameters(), lr=args.lr)
    else:
        optimizer = SGD(model.parameters(), lr=args.lr, weight_decay=0.01)

    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, min_lr=1e-5, verbose=True)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
    logger.info(f'Criterion: {criterion}')
    logger.info(f'optimizer: {optimizer}')
    logger.info(SEPARATOR)
    

    # Check if model is trained

    model_save = f'TrustWI_{args.n_node}_{args.n_edge}_hid{args.hid}_T{args.pro_T}{"_aug" if args.augmentation else ""}{"_wopse" if args.pse else ""}_{args.v}'
    
    model_trained_path = os.path.join(model_path, f'{model_save}.h5')
    if os.path.exists(model_trained_path) and not args.re_train:
        model.load_state_dict(torch.load(model_trained_path, weights_only=True))
        print("load model..")
    else:
        TrustWI_train(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            model_path,
            n_classes,
            args.epochs,
            args.annealing_epoch,
            logger=train_logger,
            model_name=model_save,
            evaluation=True,
            val_dataloader=val_loader,
            patience=20
        )
        train_logger.info(SEPARATOR)
    y_true, y_pred, y_prob = TrustWI_predict(
        model,
        args.pse,
        test_loader,
        device
    )
    
    print(y_prob.shape, y_pred.shape) 
    print(y_true) 
    logger.info(f'Accuracy on test: {accuracy_score(y_true, y_pred)}')
    logger.info(f'Balanced accuracy on test: {balanced_accuracy_score(y_true, y_pred)}')
    if args.binary:
        logger.info(f'AUC (weighted) on test: {roc_auc_score(y_true, y_pred)}')
        logger.info(f'AUC (macro) on test: {roc_auc_score(y_true, y_pred)}') 
    else:
        logger.info(f'AUC (weighted) on test: {roc_auc_score(y_true, y_prob, average="weighted", multi_class="ovr")}')
        logger.info(f'AUC (macro) on test: {roc_auc_score(y_true, y_prob,average="macro", multi_class="ovr")}') 

    logger.info(f'\n{classification_report(y_true, y_pred, digits=4)}')

if __name__ == "__main__":
    dataset_name = args.dataset_name
    print(f"dataset: {dataset_name}")
    print("Learning rate: ", args.lr)
    print("Epochs: ", args.epochs)
    model_name = args.model_name
    main(dataset_name, model_name)
    print("Done!")
