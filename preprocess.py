# Load data
from scipy.io import arff
import os
import random
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder

from utils.utils import Z_Scaler

from typing import final

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='TONIoT', choices=["TONIoT", "DoHBrw", "CICIDS" ,"CICMalMen2022"],
                    help='which dataset to use')
parser.add_argument('--num_all', type=int, default=4000000)
parser.add_argument('--b', dest='binary', action='store_true',
                        default=False, help='True if you want binary classification.')
parser.add_argument('--train_test_ratio', type=float, default=0.25, help='test ratio')
parser.add_argument('--n_small', type=int, default=0)
parser.add_argument('--seed', type=int, default=17)


random_seed = 9999

args = parser.parse_args()

if args.dataset_name == 'DoHBrw':
    CSV_FILES: final = ['l2-benign.csv', 'l2-malicious.csv']
    FEATURES: final = [
    'src_ip',
    'dst_ip',
    'src_port',
    'dst_port',
    'Timestamp',
    'Duration',
    'FlowBytesSent',
    'FlowSentRate',
    'FlowBytesReceived',
    'FlowReceivedRate',
    'PacketLengthVariance',
    'PacketLengthStandardDeviation',
    'PacketLengthMean',
    'PacketLengthMedian',
    'PacketLengthMode',
    'PacketLengthSkewFromMedian',
    'PacketLengthSkewFromMode',
    'PacketLengthCoefficientofVariation',
    'PacketTimeVariance',
    'PacketTimeStandardDeviation',
    'PacketTimeMean',
    'PacketTimeMedian',
    'PacketTimeMode',
    'PacketTimeSkewFromMedian',
    'PacketTimeSkewFromMode',
    'PacketTimeCoefficientofVariation',
    'ResponseTimeTimeVariance',
    'ResponseTimeTimeStandardDeviation',
    'ResponseTimeTimeMean',
    'ResponseTimeTimeMedian',
    'ResponseTimeTimeMode',
    'ResponseTimeTimeSkewFromMedian',
    'ResponseTimeTimeSkewFromMode',
    'ResponseTimeTimeCoefficientofVariation',
    'label'
    ]
    FEATURES_TO_STANDARDIZE: final = [
    'Duration',
    'FlowBytesSent',
    'FlowSentRate',
    'FlowBytesReceived',
    'FlowReceivedRate',
    'PacketLengthVariance',
    'PacketLengthStandardDeviation',
    'PacketLengthMean',
    'PacketLengthMedian',
    'PacketLengthMode',
    'PacketLengthSkewFromMedian',
    'PacketLengthSkewFromMode',
    'PacketLengthCoefficientofVariation',
    'PacketTimeVariance',
    'PacketTimeStandardDeviation',
    'PacketTimeMean',
    'PacketTimeMedian',
    'PacketTimeMode',
    'PacketTimeSkewFromMedian',
    'PacketTimeSkewFromMode',
    'PacketTimeCoefficientofVariation',
    'ResponseTimeTimeVariance',
    'ResponseTimeTimeStandardDeviation',
    'ResponseTimeTimeMean',
    'ResponseTimeTimeMedian',
    'ResponseTimeTimeMode',
    'ResponseTimeTimeSkewFromMedian',
    'ResponseTimeTimeSkewFromMode',
    'ResponseTimeTimeCoefficientofVariation',
    ]
    MAPPING: final = {
    "Benign": 0,
    "Malicious": 1,
    }
elif args.dataset_name == 'TONIoT':
    CSV_FILES: final = ['Network_dataset_' + str(i+1) + '.csv' for i in range(23)]
    # CSV_FILES: final = ['TONIoT-train.csv','TONIoT-val.csv','TONIoT-test.csv']
    FEATURES: final = [
        'Timestamp','src_ip','src_port','dst_ip','dst_port','proto','service',
        'duration','src_bytes','dst_bytes','conn_state','missed_bytes','src_pkts','src_ip_bytes','dst_pkts',
        'dst_ip_bytes',
        'dns_query',
        'dns_qclass',
        'dns_qtype',
        'dns_rcode',
        'dns_AA',
        'dns_RD',
        'dns_RA',
        'dns_rejected',
        'ssl_version',
        'ssl_cipher',
        'ssl_resumed',
        'ssl_established',
        'ssl_subject',
        'ssl_issuer',
        'http_trans_depth',
        'http_method',
        'http_uri',
        'http_referrer',
        'http_version',
        'http_request_body_len',
        'http_response_body_len',
        'http_status_code',
        'http_user_agent',
        'http_orig_mime_types',
        'http_resp_mime_types',
        'weird_name',
        'weird_addl',
        'weird_notice',
        'label',
        'type',
    ]
    FEATURES_TO_STANDARDIZE: final = [
        'proto',
        'service',
        'duration',
        'src_bytes',
        'dst_bytes',
        'conn_state',
        'missed_bytes',
        'src_pkts',
        'src_ip_bytes',
        'dst_pkts',
        'dst_ip_bytes',
        'dns_query',    
        'dns_rcode',
        'dns_AA',
        'dns_RD',
        'dns_RA',
        'dns_rejected',
        'ssl_version',
        'ssl_cipher',
        'ssl_resumed',
        'ssl_established',
        'ssl_subject',
        'ssl_issuer',
        'http_trans_depth',
        'http_method',
        'http_uri',
        'http_referrer',
        'http_version',
        'http_request_body_len',
        'http_response_body_len',
        'http_status_code',
        'http_user_agent',
        'http_orig_mime_types',
        'http_resp_mime_types',
        'weird_name',
        'weird_addl',
        'weird_notice',
        ]
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
    CSV_FILES: final = ['dataset_1.csv', 'dataset_2.csv']
    FEATURES:final = [
        'Flow ID','src_ip','src_port','dst_ip','dst_port','Protocol','Timestamp', # 7
        'Flow Duration','Total Fwd Packets','Total Backward Packets','Total Length of Fwd Packets','Total Length of Bwd Packets', # 5
        'Fwd Packet Length Max','Fwd Packet Length Min','Fwd Packet Length Mean','Fwd Packet Length Std', # 4
        'Bwd Packet Length Max','Bwd Packet Length Min','Bwd Packet Length Mean','Bwd Packet Length Std', # 4
        'Flow Bytes/s','Flow Packets/s','Flow IAT Mean','Flow IAT Std','Flow IAT Max','Flow IAT Min', # 6
        'Fwd IAT Total','Fwd IAT Mean','Fwd IAT Std','Fwd IAT Max','Fwd IAT Min','Bwd IAT Total', # 6
        'Bwd IAT Mean','Bwd IAT Std','Bwd IAT Max','Bwd IAT Min','Fwd PSH Flags','Bwd PSH Flags','Fwd URG Flags', # 6
        'Bwd URG Flags','Fwd Header Length','Bwd Header Length', 'Fwd Packets/s','Bwd Packets/s','Min Packet Length', # 6
        'Max Packet Length','Packet Length Mean','Packet Length Std','Packet Length Variance',  # 4
        'FIN Flag Count','SYN Flag Count','RST Flag Count','PSH Flag Count','ACK Flag Count','URG Flag Count','CWE Flag Count','ECE Flag Count', # 8
        'Down/Up Ratio','Average Packet Size','Avg Fwd Segment Size','Avg Bwd Segment Size', # 4
        'Fwd Header Length2', 'Fwd Avg Bytes/Bulk','Fwd Avg Packets/Bulk','Fwd Avg Bulk Rate','Bwd Avg Bytes/Bulk','Bwd Avg Packets/Bulk','Bwd Avg Bulk Rate',
        'Subflow Fwd Packets','Subflow Fwd Bytes','Subflow Bwd Packets','Subflow Bwd Bytes','Init_Win_bytes_forward','Init_Win_bytes_backward',
        'act_data_pkt_fwd','min_seg_size_forward','Active Mean','Active Std','Active Max','Active Min','Idle Mean','Idle Std','Idle Max','Idle Min',
        'label'
    ]
    FEATURES_TO_STANDARDIZE: final = [
        'Protocol','Flow Duration','Total Fwd Packets','Total Backward Packets','Total Length of Fwd Packets','Total Length of Bwd Packets',
        'Fwd Packet Length Max','Fwd Packet Length Min','Fwd Packet Length Mean','Fwd Packet Length Std', 
        'Bwd Packet Length Max','Bwd Packet Length Min','Bwd Packet Length Mean','Bwd Packet Length Std',
        'Flow Bytes/s','Flow Packets/s','Flow IAT Mean','Flow IAT Std','Flow IAT Max','Flow IAT Min', 
        'Fwd IAT Total','Fwd IAT Mean','Fwd IAT Std','Fwd IAT Max','Fwd IAT Min','Bwd IAT Total', 
        'Bwd IAT Mean','Bwd IAT Std','Bwd IAT Max','Bwd IAT Min','Fwd PSH Flags','Fwd URG Flags',
        'Fwd Header Length','Bwd Header Length', 'Fwd Packets/s','Bwd Packets/s','Min Packet Length',
        'Max Packet Length','Packet Length Mean','Packet Length Std','Packet Length Variance', 
        'FIN Flag Count','SYN Flag Count','RST Flag Count','PSH Flag Count','ACK Flag Count','URG Flag Count','CWE Flag Count','ECE Flag Count',
        'Down/Up Ratio','Average Packet Size','Avg Fwd Segment Size','Avg Bwd Segment Size', 
        'Fwd Header Length2',
        'Subflow Fwd Packets','Subflow Fwd Bytes','Subflow Bwd Packets','Subflow Bwd Bytes','Init_Win_bytes_forward','Init_Win_bytes_backward',
        'act_data_pkt_fwd','min_seg_size_forward','Active Mean','Active Std','Active Max','Active Min','Idle Mean','Idle Std','Idle Max','Idle Min',
        # remove ['Bwd PSH Flags', 'Bwd URG Flags', 'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate']
    ]
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
elif args.dataset_name == 'UNSW15':
    CSV_FILES: final = ['UNSW-NB15_1.csv','UNSW-NB15_2.csv','UNSW-NB15_3.csv','UNSW-NB15_4.csv']
    FEATURES: final = [
    'srcip',
    'sport',
    'dstip',
    'dsport',
    'proto',
    'state',
    'dur',
    'sbytes',
    'dbytes',
    'sttl',
    'dttl',
    'sloss',
    'dloss',
    'service',
    'sload',
    'dload',
    'spkts',
    'dpkts',
    'swin',
    'dwin',
    'stcpb',
    'dtcpb',
    'smeansz',
    'dmeansz',
    'trans_depth',
    'res_bdy_len',
    'sjit',
    'djit',
    'stime',
    'ltime',
    'sintpkt',
    'dintpkt',
    'tcprtt',
    'synack',
    'ackdat',
    'is_sm_ips_ports',
    'ct_state_ttl',
    'ct_flw_http_mthd',
    'is_ftp_login',
    'ct_ftp_cmd',
    'ct_srv_src',
    'ct_srv_dst',
    'ct_dst_ltm',
    'ct_src_ltm',
    'ct_src_dport_ltm',
    'ct_dst_sport_ltm',
    'ct_dst_src_ltm',
    'attack_cat',
    'label'
    ]
    FEATURES_TO_STANDARDIZE: final = [
    'dur',
    'sbytes',
    'dbytes',
    'sttl',
    'dttl',
    'sloss',
    'dloss',
    'sload',
    'dload',
    'spkts',
    'dpkts',
    'swin',
    'dwin',
    'stcpb',
    'dtcpb',
    'smeansz',
    'dmeansz',
    'trans_depth',
    'res_bdy_len',
    'sjit',
    'djit',
    'sintpkt',
    'dintpkt',
    'tcprtt',
    'synack',
    'ackdat',
    'is_sm_ips_ports',
    'ct_state_ttl',
    'ct_flw_http_mthd',
    'is_ftp_login',
    'ct_ftp_cmd',
    'ct_srv_src',
    'ct_srv_dst',
    'ct_dst_ltm',
    'ct_src_ltm',
    'ct_src_dport_ltm',
    'ct_dst_sport_ltm',
    'ct_dst_src_ltm',
    ]
    MAPPING: final = {
    "Normal": 0,
    "Generic": 1,
    "Exploits": 2,
    "Fuzzers": 3,
    "DoS": 4,
    "Reconnaissance": 5,
    "Analysis": 6,
    "Backdoors": 7,
    "Shellcode": 8,
    "Worms": 9
}
elif args.dataset_name == '':
    CSV_FILES: final = ['UNSW-NB15_1.csv','UNSW-NB15_2.csv','UNSW-NB15_3.csv','UNSW-NB15_4.csv']
    FEATURES: final = [
    'flow_duration', 'Header_Length', 'Protocol Type', 'Duration',
       'Rate', 'Srate', 'Drate', 'fin_flag_number', 'syn_flag_number',
       'rst_flag_number', 'psh_flag_number', 'ack_flag_number',
       'ece_flag_number', 'cwr_flag_number', 'ack_count',
       'syn_count', 'fin_count', 'urg_count', 'rst_count', 
    'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC', 'TCP',
       'UDP', 'DHCP', 'ARP', 'ICMP', 'IPv', 'LLC', 'Tot sum', 'Min',
       'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Number', 'Magnitue',
       'Radius', 'Covariance', 'Variance', 'Weight', 
    ]
    FEATURES_TO_STANDARDIZE: final = [

    ]
    MAPPING: final = {
    "Normal": 0,
    "Generic": 1,
    "Exploits": 2,
    "Fuzzers": 3,
    "DoS": 4,
    "Reconnaissance": 5,
    "Analysis": 6,
    "Backdoors": 7,
    "Shellcode": 8,
    "Worms": 9
}
# elif args.dataset_name == 'CICAndMal2020':
#     CSV_FILES: final = ['Adware.csv', 'Backdoor.csv','Banker.csv','Ben0.csv','Ben1.csv','Ben2.csv','Ben3.csv','Ben4.csv','Dropper.csv','FileInfector.csv','NoCategory.csv','PUA.csv','Ransomware.csv','Riskware.csv','Scareware.csv','SMS.csv','Spy.csv','Trojan.csv','Zeroday.csv']
#     FEATURES: final = [

#     ]
# elif args.dataset_name == 'CICMalMen2022':
#     CSV_FILES: final = ['Obfuscated-MalMem2022.csv']
#     FEATURES: final = [

#     ]

    # Create small sample set that contains all classes
def create_balanced_sample(df, n_per_class):
    sample_dfs = []
    remaining_to_sample = n_per_class * len(df['label'].unique())

    # Step 1: Sample as much as possible from each class
    for label in df['label'].unique():
        class_samples = df[df['label'] == label]
        num_samples = min(n_per_class, len(class_samples))
        sample_dfs.append(class_samples.sample(num_samples, random_state=seed))
        remaining_to_sample -= num_samples

    # Step 2: If there's a deficit, sample more from other classes
    if remaining_to_sample > 0:
        other_samples = df[~df.index.isin(pd.concat(sample_dfs).index)]
        extra_samples = other_samples.sample(remaining_to_sample, random_state=seed)
        sample_dfs.append(extra_samples)

    return pd.concat(sample_dfs)


def process(df,data_name):
        # Clean data
        if data_name == 'TONIoT':
            df['label'] = df['type'].str.strip() # Used to remove leading and trailing white space characters (including spaces, tabs, etc.) from a string.

            df = df.drop('type', axis='columns')
        
            mask = df[FEATURES_TO_STANDARDIZE]=='-'
            # str -> float
            for feat in FEATURES_TO_STANDARDIZE:
                df[feat] = LabelEncoder().fit_transform(df[feat].astype(str))
            df[mask] = 0
            # df[LABEL_TO_NUM] = MinMaxScaler().fit_transform(df[LABEL_TO_NUM])
        elif data_name == 'CICIDS':
            drop_col =  ['Flow ID', 'Bwd PSH Flags', 'Bwd URG Flags', 'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate']
            df = df.drop(drop_col, axis='columns')
        else:
            df['label'] = df['label'].str.strip() # Used to remove leading and trailing white space characters (including spaces, tabs, etc.) from a string.
            
        # Mapping
        df['label'] = df['label'].map(MAPPING)
        # Use Min Max Scaler
        # df[FEATURES_TO_STANDARDIZE] = MinMaxScaler().fit_transform(df[FEATURES_TO_STANDARDIZE]) # Specify the feature column (FEATURES_TO_STANDARDIZE) for standardization.
        
        # 同时删除所有列中的最大值和最小值的行
        max_values = df[FEATURES_TO_STANDARDIZE].max()
        min_values = df[FEATURES_TO_STANDARDIZE].min()
        df = df[~(df[FEATURES_TO_STANDARDIZE].isin(max_values) | df[FEATURES_TO_STANDARDIZE].isin(min_values)).any(axis=1)]
        df[FEATURES_TO_STANDARDIZE] = Z_Scaler().fit_transform(df[FEATURES_TO_STANDARDIZE])
        return df
    

def pre_processing(data_name: str, num_all, train_test_ratio, seed, binary, batch_num):
    # Init global dataframe
    dataset_path = os.path.join('datasets', data_name, 'raw')
    df = pd.DataFrame(columns=FEATURES)
    print("loading dataset...")

    for csv_file in CSV_FILES:
        if data_name == 'UNSW15':
            df_local = pd.read_csv(os.path.join(dataset_path, csv_file), header=None, low_memory=False).fillna(0)
        else:
            df_local = pd.read_csv(os.path.join(dataset_path, csv_file), header=0, low_memory=False).fillna(0)
        print(csv_file)
        df_local.columns = FEATURES
        try:
            df = pd.concat([df, df_local])
        except ValueError as e:
            print(f"Error concatenating DataFrames: {e}")
            
    df = process(df,data_name)
    
    if batch_num > 0:
        df = create_balanced_sample(df, batch_num)

    # Create train, validation and test set
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    num_all = min(num_all, len(df))
    df = df[:num_all]
    print(set(df['label']))
    train_indices, val_indices, test_indices = generate_random_partition_indices(num_all, train_ratio=0.1)
    if binary:
        # Create train, validation and test set for binary classification
        df['label'] = df['label'].apply(lambda x: 1 if x != 0 else 0)
        # train, val = train_test_split(df, test_size=train_test_ratio-0.05, shuffle=True, random_state=seed)
        # train, test = train_test_split(train, test_size=train_test_ratio, shuffle=True, random_state=seed)
        train = df[train_indices]
        val=df[val_indices]
        test=df[test_indices]
        train.to_csv(os.path.join(dataset_path, f"{data_name}-{num_all}-train-binary.csv"), index=False)
        val.to_csv(os.path.join(dataset_path, f"{data_name}-{num_all}-val-binary.csv"), index=False)
        test.to_csv(os.path.join(dataset_path, f"{data_name}-{num_all}-test-binary.csv"), index=False)
    else:
        train, val = train_test_split(df, test_size=train_test_ratio-0.05, shuffle=True, random_state=seed)
        train, test = train_test_split(train, test_size=train_test_ratio, shuffle=True, random_state=seed)
        train.to_csv(os.path.join(dataset_path, f"{data_name}-{num_all}-train.csv"), index=False)
        val.to_csv(os.path.join(dataset_path, f"{data_name}-{num_all}-val.csv"), index=False)
        test.to_csv(os.path.join(dataset_path, f"{data_name}-{num_all}-test.csv"), index=False)
    print("Sample counts per class:")
    print(df['label'].value_counts())

def generate_random_partition_indices(num_nodes, train_ratio=0.03, val_ratio=0.47, test_ratio=0.5):
    test_ratio = train_ratio*5
    val_ratio = 1-train_ratio-test_ratio
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1."

    all_indices = np.arange(num_nodes)
    np.random.seed(9977) # CIC-IDS 117
    np.random.shuffle(all_indices)

    train_size = int(num_nodes * train_ratio)
    val_size = int(num_nodes * val_ratio)
    
    train_indices = all_indices[:train_size]
    val_indices = all_indices[train_size:train_size + val_size]
    test_indices = all_indices[train_size + val_size:]
    
    return train_indices, val_indices, test_indices

if __name__ == '__main__':
    data_name = args.dataset_name
    num_all = args.num_all
    train_test_ratio = args.train_test_ratio
    seed = args.seed

    batch_num = args.n_small

    print(f"dataset: {data_name}")
    pre_processing(data_name, num_all, train_test_ratio, seed, args.binary, batch_num)
    print("Done!")
