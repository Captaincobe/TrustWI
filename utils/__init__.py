import argparse
import os

SEPARATOR = '\n------------------------------------\n'
DETECTION_RATE = 'Detection Rate'
CONFUSION_MATRIX = 'Confusion Matrix'
BAR_STACKED = 'Bar Stacked'
LOGGER = 'logger.log'
TRAINING_LOGGER = 'training.log'
CWD = os.getcwd()

def get_path_of_all(name_dir,  small=0, num_neigh=None, num_node=None,num_edge=None, alpha=0.01, hid=0, n_convs=0, pro_T=0, model_name='uinweak',v=None):
    if small > 0:
        name_dir = os.path.join(name_dir, str(small))

    log_path = os.path.join(CWD, 'log', name_dir)
    if model_name == 'uinweak':
        # log_file_path = os.path.join(log_path, LOGGER if num_neigh is None else f'logger_{num_neigh}_{num_node}_{num_edge}_'
        #                                                                         f'hid_{hid}_convs_{n_convs}'
        #                                                                         f'{"_aug" if augmentation else ""}.log')
        
        log_file_path = os.path.join(log_path, LOGGER if num_neigh is None else f'logger_{num_node}_{num_edge}_'
                                                                                f'hid{hid}'
                                                                                f'_T{pro_T}_a{alpha}'
                                                                                f'{v}'
                                                                                '.log')
        log_train_path = os.path.join(log_path,
                                    TRAINING_LOGGER if num_neigh is None else f'training_{num_node}_{num_edge}_'
                                                                                f'hid_{hid}'
                                                                                f'_T{pro_T}_a{alpha}.log')
    elif model_name == 'simple' or model_name == 'attention':
        log_file_path = os.path.join(log_path, LOGGER if num_neigh is None else f'{model_name}_logger_'
                                                                                f'{num_node}_{num_edge}_'
                                                                                f'hid_{hid}_T{pro_T}'
                                                                                f'.log')
        log_train_path = os.path.join(log_path, TRAINING_LOGGER if num_neigh is None else f'{model_name}_training_'
                                                                                f'{num_node}_{num_edge}_'
                                                                                f'hid{hid}_T{pro_T}'
                                                                                f'.log')
    else:
        log_file_path = os.path.join(log_path, LOGGER if num_neigh is None else f'{model_name}_logger_'
                                                                                f'{num_node}_{num_edge}_'
                                                                                f'hid_{hid}_convs{pro_T}'
                                                                                f'.log')
        log_train_path = os.path.join(log_path, TRAINING_LOGGER if num_neigh is None else f'{model_name}_training_'
                                                                                f'{num_node}_{num_edge}_'
                                                                                f'hid{hid}_convs{pro_T}'
                                                                                f'.log')

    model_path = os.path.join(CWD,'model', name_dir)
    result_path = os.path.join(log_path,'results.csv')
    image_path = os.path.join(CWD,'image', name_dir)
    confusion_matrix_path = os.path.join(image_path, CONFUSION_MATRIX)
    detection_rate_path = os.path.join(image_path, DETECTION_RATE)

    return log_path, log_file_path, log_train_path, model_path, result_path, confusion_matrix_path, detection_rate_path


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
