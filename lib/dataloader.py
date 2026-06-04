import numpy as np
import torch
import torch.utils.data
from lib.add_window import Add_Window_Horizon
from lib.load_dataset import load_st_dataset
from lib.normalization import NScaler, MinMax01Scaler, MinMax11Scaler, StandardScaler, ColumnMinMaxScaler

def get_scaler(data, normalizer, column_wise=False):
    if normalizer == 'max01':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax01Scaler(minimum, maximum)
        print('Normalize the dataset by MinMax01 Normalization')
    elif normalizer == 'max11':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax11Scaler(minimum, maximum)
        print('Normalize the dataset by MinMax11 Normalization')
    elif normalizer == 'std':
        if column_wise:
            mean = data.mean(axis=0, keepdims=True)
            std = data.std(axis=0, keepdims=True)
        else:
            mean = data.mean()
            std = data.std()
        scaler = StandardScaler(mean, std)
        print('Normalize the dataset by Standard Normalization')
    elif normalizer == 'None':
        scaler = NScaler()
        data = scaler.transform(data)
        print('Does not normalize the dataset')
    elif normalizer == 'cmax':
        #column min max, to be depressed
        #note: axis must be the spatial dimension, please check !
        scaler = ColumnMinMaxScaler(data.min(axis=0), data.max(axis=0))
        print('Normalize the dataset by Column Min-Max Normalization')
    else:
        raise ValueError
    return scaler

def split_data_by_days(data, val_days, test_days, interval=60):
    '''
    :param data: [B, *]
    :param val_days:
    :param test_days:
    :param interval: interval (15, 30, 60) minutes
    :return:
    '''
    T = int((24*60)/interval)
    test_data = data[-T*test_days:]
    val_data = data[-T*(test_days + val_days): -T*test_days]
    train_data = data[:-T*(test_days + val_days)]
    return train_data, val_data, test_data

def split_data_by_ratio(data, val_ratio, test_ratio):
    data_len = data.shape[0]
    test_data = data[-int(data_len*test_ratio):]
    val_data = data[-int(data_len*(test_ratio+val_ratio)):-int(data_len*test_ratio)]
    train_data = data[:-int(data_len*(test_ratio+val_ratio))]
    return train_data, val_data, test_data

def data_loader(X, Y, batch_size, device, shuffle=True, drop_last=True):
    X = torch.tensor(X, dtype=torch.float32).to(device)
    Y = torch.tensor(Y, dtype=torch.float32).to(device)
    data = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)
    return dataloader

# def get_dataloader(args, normalizer = 'std', tod=False, dow=False, weather=False, single=True):
#     #load raw st dataset
#     data = load_st_dataset(args.dataset)        # L, N, D
#     #normalize st data
#     scaler = normalize_dataset(data, normalizer, args.column_wise)
#     #spilit dataset by days or by ratio
#     if args.test_ratio > 1:
#         data_train, data_val, data_test = split_data_by_days(data, args.val_ratio, args.test_ratio)
#     else:
#         data_train, data_val, data_test = split_data_by_ratio(data, args.val_ratio, args.test_ratio)
#     #add time window
#     x_tra, y_tra = Add_Window_Horizon(data_train, args.lag, args.horizon, single)
#     x_val, y_val = Add_Window_Horizon(data_val, args.lag, args.horizon, single)
#     x_test, y_test = Add_Window_Horizon(data_test, args.lag, args.horizon, single)
#     print('Train: ', x_tra.shape, y_tra.shape)
#     print('Val: ', x_val.shape, y_val.shape)
#     print('Test: ', x_test.shape, y_test.shape)
#     ##############get dataloader######################
#     train_dataloader = data_loader(x_tra, y_tra, args.batch_size, args.device, shuffle=True, drop_last=True)
#     if len(x_val) == 0:
#         val_dataloader = None
#     else:
#         val_dataloader = data_loader(x_val, y_val, args.batch_size, args.device, shuffle=False, drop_last=True)
#     test_dataloader = data_loader(x_test, y_test, args.batch_size, args.device, shuffle=False, drop_last=False)
#     print('Train_batch: ', len(train_dataloader))
#     print('Val_batch: ', len(val_dataloader))
#     print('Test_batch: ', len(test_dataloader))
#     return train_dataloader, val_dataloader, test_dataloader, scaler

def get_dataloader(args, flow, normalizer = 'std', single=True):
    #load raw st dataset
    # data = load_st_dataset(args.dataset, flow)        # L, N, D
    data, period, unique_periods = load_st_dataset(args.dataset, flow, args.k)
    #spilit dataset by days or by ratio
    if args.test_ratio > 1:
        data_train, data_val, data_test = split_data_by_days(data, args.val_ratio, args.test_ratio)
    else:
        data_train, data_val, data_test = split_data_by_ratio(data, args.val_ratio, args.test_ratio)
    #add time window
    x_tra, y_tra = Add_Window_Horizon(data_train, args.lag, args.horizon, single)
    x_val, y_val = Add_Window_Horizon(data_val, args.lag, args.horizon, single)
    x_test, y_test = Add_Window_Horizon(data_test, args.lag, args.horizon, single)

    print('Train: ', x_tra.shape, y_tra.shape)
    print('Val: ', x_val.shape, y_val.shape)
    print('Test: ', x_test.shape, y_test.shape)

    #normalize st data
    scaler = get_scaler(data_train[:,:,:args.input_dim], normalizer, column_wise=True)
    x_tra[...,:args.input_dim] = scaler.transform(x_tra[...,:args.input_dim])
    x_val[...,:args.input_dim] = scaler.transform(x_val[...,:args.input_dim])
    x_test[...,:args.input_dim] = scaler.transform(x_test[...,:args.input_dim])

    print('Load Dataset shaped: ', x_tra[...,:args.input_dim].shape, x_tra[...,:args.input_dim].max(), x_tra[...,:args.input_dim].min(), x_tra[...,:args.input_dim].mean(), np.median(x_tra[...,:args.input_dim]))

    ##############get dataloader######################
    train_dataloader = data_loader(x_tra, y_tra, args.batch_size, args.device, shuffle=True, drop_last=True)
    val_dataloader = data_loader(x_val, y_val, args.batch_size, args.device, shuffle=False, drop_last=True)
    test_dataloader = data_loader(x_test, y_test, args.batch_size, args.device, shuffle=False, drop_last=False)

    return train_dataloader, val_dataloader, test_dataloader, scaler, period, unique_periods


if __name__ == '__main__':
    import argparse
    #MetrLA 207; BikeNYC 128; SIGIR_solar 137; SIGIR_electric 321
    DATASET = 'NYCTaxi'
    if DATASET == 'NYCTaxi':
        NODE_NUM = 266
    elif DATASET == 'BJTaxi':
        NODE_NUM = 1024
    elif DATASET == 'CCGTaxi':
        NODE_NUM = 77
    elif DATASET == 'NYCRide':
        NODE_NUM = 77
    elif DATASET == 'PEMSD7':
        NODE_NUM = 307
    flow = 'inflow'
    parser = argparse.ArgumentParser(description='PyTorch dataloader')
    parser.add_argument('--dataset', default=DATASET, type=str)
    parser.add_argument('--flow', default='inflow', type=str)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--num_nodes', default=NODE_NUM, type=int)
    parser.add_argument('--val_ratio', default=0.2, type=float)
    parser.add_argument('--test_ratio', default=0.2, type=float)
    parser.add_argument('--input_dim', default=1, type=int)
    parser.add_argument('--lag', default=4, type=int)
    parser.add_argument('--horizon', default=4, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    args = parser.parse_args()
    train_dataloader, val_dataloader, test_dataloader, scaler = get_dataloader(args, flow, normalizer = 'std', single=True)