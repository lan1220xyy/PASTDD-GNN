import os
import numpy as np
import pandas as pd
import torch


def generate_time_features(timestamps):
    return np.arange(len(timestamps)).reshape(-1, 1)

def FFT_node_decompose(x, k=3, tolerance=0.1):
    """
    节点级周期识别 + 非周期成分提取

    Args:
        x: [T, N]
        k: 每个节点最多提取k个周期
        tolerance: 周期误差容忍

    Returns:
        periods: [N, k] 每个节点的Top-K周期
        unique_periods: 所有节点的周期集合（list）
        x_dyc: [T, N] 每个节点去除其周期后的非周期成分
    """

    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()

    T, N = x.shape

    # ===== 1. FFT =====
    xf = torch.fft.rfft(x, dim=0)   # [F, N]
    freq = torch.abs(xf)
    freq[0] = 0  # 去掉直流

    # ===== 2. 找Top-K频率 =====
    k_actual = min(k, freq.shape[0])
    sorted_magnitudes, sorted_indices = torch.sort(
        freq.T, dim=1, descending=True
    )

    periods = torch.zeros(N, k_actual, dtype=torch.int)
    unique_periods_set = set()

    # ===== 3. 周期筛选 =====
    for n in range(N):
        valid_periods = []
        for i in range(min(k_actual, len(sorted_indices[n]))):
            idx = sorted_indices[n, i].item()
            if idx == 0:  # 跳过直流分量
                continue
            calc_period = T / idx  # 计算出的周期（可能是浮点数）
            int_period = round(calc_period)

            # 检查误差是否在容忍范围内
            if abs(calc_period - int_period) <= tolerance and 16<= int_period <= 672:
                valid_periods.append(int_period)
                unique_periods_set.add(int_period)  # 添加到集合中
                if len(valid_periods) == k_actual:
                    break

        # 填充结果，不足的部分补0
        periods[n, :len(valid_periods)] = torch.tensor(valid_periods)

    unique_periods = sorted(list(unique_periods_set))

    return periods, unique_periods


def load_st_dataset(dataset, flow, k):
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if dataset in ['PEMSD4', 'PEMSD8', 'PEMSD7']:
        data_path = os.path.join(base_path, f'data/{dataset}/data.csv')
        period = [288, 288 * 7]
    elif dataset in ['NYCTaxi', 'BJTaxi', 'CCGTaxi', 'CCGRide']:
        data_path = os.path.join(base_path, f'data/{dataset}/{flow}.csv')
        # period = [48, 336]
    else:
        raise ValueError

    df = pd.read_csv(data_path)
    # L*N
    data = df.drop(columns='time').to_numpy(dtype=np.float64)
    # L*1
    time_features = generate_time_features(df['time'])

    if len(data.shape) == 2:
        period, unique_period = FFT_node_decompose(data, k, 0.1)
        print(unique_period)
        print(period.shape)
        data = np.expand_dims(data, axis=-1)  # L*N*1
        print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
    if time_features is not None:
        # L*N*1
        time_features = np.repeat(time_features[:, np.newaxis, :], data.shape[1], axis=1)
        # L*N*2
        data = np.concatenate((data, time_features), axis=-1)
        print('Load data with time:', data.shape)
    return data, period, unique_period

if __name__ == '__main__':

    load_st_dataset('NYCTaxi', 'inflow')
    # dataset = 'BJTaxi'
    # data_path = os.path.join(f'../data/{dataset}/inflow.csv')
    # df = pd.read_csv(data_path)
    # data = df.drop(columns='time').to_numpy(dtype=np.float64)
    #
    # period,unique_period,x_dyc = FFT_node_decompose(data, k=2)
    # print(x_dyc.shape)
    # print(unique_period)
    # print(period.shape)
    # for i in range(len(period)):
    #     if i+5 < len(period):
    #         print(period[i:i+5])
    #     else: print(period[i:])


