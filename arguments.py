import argparse
from utils import str2bool

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='TONIoT', choices=["TONIoT","DoHBrw", "CICIDS", "CICMalMen2022"],
                    help='which dataset to use')
    parser.add_argument('--n_classes', dest='n_classes', action='store', choices=[10, 2, 15],
                        type=int, default=10, help='Num classes.')
    parser.add_argument('--b', dest='binary', action='store_true',
                         default=False, help='True if you want binary classification.')
    
    parser.add_argument('--pse', dest='pse', type=str2bool, default=True)
    parser.add_argument('--aug', dest='augmentation', action='store',
                        type=str2bool, default=False, help='Apply or not augmentation on data.')
    parser.add_argument('--n_small', dest='n_small', type=int, default=0) # , choices=[1000,10000, 2000, 15000]

    parser.add_argument('--v', dest='v', action='store',
                    type=str, default='0', choices=['v-l','v-s','v-p','v-add'])
    parser.add_argument('--model_name', dest='model_name', action='store',
                    type=str, default='uinweak', choices=["uinweak","attention", "simple", "gcn","gat"])
    parser.add_argument('--re', dest='re_train', action='store_true', default=False, help='True if you want to retrain the model.')
    
    
    parser.add_argument('--neigh', dest='n_neigh', type=int, default=2700, 
                        help='Defines the number of neighborhood nodes.')
    parser.add_argument('--node', dest='n_node', type=int, default=0, 
                        help='Defines the percentage of features deleted.')
    parser.add_argument('--edge', dest='n_edge', type=int, default=0, 
                        help='Defines the percentage of edges deleted.')


    parser.add_argument('--epo', dest='epochs', type=int, default=100)
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--hid', dest='hid', type=int, default=128)
    parser.add_argument('--n_convs', dest='num_convs', type=int, default=2)
    parser.add_argument('--knn_neigh', dest='knn_neigh', type=int, default=25)
    parser.add_argument('--knn_metric', dest='knn_metric', type=str, default='cosine', choices=['cosine','minkowski'])
    parser.add_argument('--T', dest='pro_T', type=int, default=20)
    parser.add_argument("--annealing_epoch", type=int, default=10)
    parser.add_argument('--n_parts', dest='n_parts', type=int, default=128, help='Define batch size (num_nodes / num_parts).')


    args = parser.parse_args()

    return args
