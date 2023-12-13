import csv
import datetime
import h5py
import numpy as np
import numpy.ma as ma
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from matplotlib import pyplot as plt
from model import Transformer, ExtremeValueLoss
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.autograd import Variable

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

date_split = [
    ['20110901', '20171231', '20210930'], # 영덕
    ['20051224', '20151231', '20210930'], # 남해강진
    ['20051129', '20151231', '20210930'], # 통영사량
    ['20110831', '20171231', '20210930'], # 여수신월
    ['20051025', '20151231', '20210930'], # 완도청산
    ['20060703', '20151231', '20210930'], # 목포
    ['20110904', '20171231', '20210930'], # 보령효자도
    ['19960706', '20151231', '20210930'], # 칠발도
    ['20081115', '20151231', '20210930'], # 포항
    ['19980108', '20151231', '20210930'], # 거문도
    ['20080101', '20151231', '20210930'], # 거제도
    ['20080101', '20151231', '20210930'], # 덕적도
    ['20070101', '20151231', '20210930'], # 동해
    ['20081114', '20151231', '20210930'], # 마라도
    ['20091021', '20151231', '20210930'], # 외연도
    ['20111228', '20171231', '20210930'], # 울릉도
]

name = [
    'YD',
    'NH',
    'TY',
    'YS',
    'WD',
    'MP',
    'BH',
    'CB',
    'PH',
    'GM',
    'GJ',
    'DJ',
    'DH',
    'MR',
    'YY',
    'UL',
]

path = [
    '../../../../../data/YD.dat',
    '../../../../../data/NH.dat',
    '../../../../../data/TY.dat',
    '../../../../../data/YS.dat',
    '../../../../../data/WD.dat',
    '../../../../../data/MP.dat',
    '../../../../../data/BH.dat',
    '../../../../../data/CB.dat',
    '../../../../../data/PH.dat',
    '../../../../../data/GM.dat',
    '../../../../../data/GJ.dat',
    '../../../../../data/DJ.dat',
    '../../../../../data/DH.dat',
    '../../../../../data/MR.dat',
    '../../../../../data/YY.dat',
    '../../../../../data/UL.dat',
]

if __name__ == '__main__' :
    batch_size  = 128
    iteration   = 50000
    memory_size = 1024
    time_step   = [14, 7]
    x_array     = []
    x_label     = []
    y_array     = []
    y_label     = []
    CORR        = [[], [], []]
    RMSE        = [[], [], []]
    SCORE       = [[], [], []]

    def Generate_Batch(array, label, batch_size, date_split, time_step):
        x_batch = []
        y_batch = []
        z_batch = []
        t_batch = []

        while len(x_batch) < batch_size:
            data_index = np.random.randint(len(array))
            time_index = np.random.randint(time_step[0], len(array[data_index]) - time_step[1] + 1)
            time       = []
            
            date = datetime.datetime(int(date_split[data_index][0][:4]), int(date_split[data_index][0][4:6]), int(date_split[data_index][0][6:]))
            date = date + datetime.timedelta(days=time_index - time_step[0])

            for t in range(time_step[0] + time_step[1]):
                month = np.ones(1,) * (date.date().month - 1)
                day   = np.ones(1,) * (date.date().day - 1)

                time.append(np.concatenate([month, day]))
                date = date + datetime.timedelta(days=1)

            location = np.zeros(16,)
            location[data_index] = 1

            data = array[data_index][time_index - time_step[0]:time_index + time_step[1]]

            if len(data[np.isnan(data)]) == 0:
                x_batch.append(data)
                y_batch.append(label[data_index][time_index:time_index + time_step[1]])
                t_batch.append(time)
                z_batch.append(location)

        x_batch = np.array(x_batch)
        x_batch = Variable(torch.from_numpy(x_batch)).type(torch.FloatTensor).cuda()

        y_batch = np.array(y_batch)
        y_batch = Variable(torch.from_numpy(y_batch)).type(torch.LongTensor).cuda()
    
        z_batch = np.array(z_batch)
        z_batch = Variable(torch.from_numpy(z_batch)).type(torch.FloatTensor).cuda()

        t_batch = np.array(t_batch)
        t_batch = Variable(torch.from_numpy(t_batch)).type(torch.FloatTensor).cuda()
    
        return x_batch, t_batch, y_batch, z_batch
        # sst, time, label, location

    if not os.path.exists('model'):
        os.makedirs('model')

    mean = []
    std  = []

    for h in range(len(path)):
        data = pd.read_csv(path[h], encoding='cp949')
        data.index = data['date']

        x_data = data.loc[pd.IndexSlice[date_split[h][0]:date_split[h][1]], :]
        x_data = np.stack([x_data.loc[:, 'date'], x_data.loc[:, 'SST']], axis=1)

        new_data = [x_data[0]]

        for i in range(1, len(x_data)):
            prev_date = datetime.datetime(int(x_data[i - 1][0]) // 10000, (int(x_data[i - 1][0]) // 100) % 100, int(x_data[i - 1][0]) % 100)
            next_date = datetime.datetime(int(x_data[i][0]) // 10000, (int(x_data[i][0]) // 100) % 100, int(x_data[i][0]) % 100)

            for j in range((next_date - prev_date).days - 1):
                new_data.append(np.ones_like(x_data[0]) * np.nan)

            new_data.append(x_data[i])

        x_data = np.array(new_data)[:, 1:]

        y_data = data.loc[pd.IndexSlice[date_split[h][1]:date_split[h][2]], :]
        y_data = np.stack([y_data.loc[:, 'date'], y_data.loc[:, 'SST']], axis=1)[1:]

        new_data = [y_data[0]]

        for i in range(1, len(y_data)):
            prev_date = datetime.datetime(int(y_data[i - 1][0]) // 10000, (int(y_data[i - 1][0]) // 100) % 100, int(y_data[i - 1][0]) % 100)
            next_date = datetime.datetime(int(y_data[i][0]) // 10000, (int(y_data[i][0]) // 100) % 100, int(y_data[i][0]) % 100)

            for j in range((next_date - prev_date).days - 1):
                new_data.append(np.ones_like(y_data[0]) * np.nan)

            new_data.append(y_data[i])

        y_data = np.array(new_data)[:, 1:]

        x_array.append(x_data)
        y_array.append(y_data)

        # label
        label = np.zeros((len(x_data),), dtype=np.float32)
        label[x_data[:, 0] <= 4] = -1
        label[x_data[:, 0] >= 28] = 1
        x_label.append(label)

        number_sample = [len(label[label == -1]), len(label[label == 0]), len(label[label == 1])]

        label = np.zeros((len(y_data),), dtype=np.float32)
        label[y_data[:, 0] <= 4] = -1
        label[y_data[:, 0] >= 28] = 1
        y_label.append(label)

        mean.append(np.nanmean(x_array[h], axis=0))
        std.append(np.nanstd(x_array[h], axis=0))

        x_array[h] = (x_array[h] - mean[h]) / std[h]
        y_array[h] = (y_array[h] - mean[h]) / std[h]

    model = Transformer(hidden_size=32, output_size=1, num_layers=2, memory_size=memory_size, time_step=time_step).cuda()
    # model.load_state_dict(torch.load('model/{}_weights.pt'.format(name[h])), strict=False)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)

    memory = None

    for i in range(iteration):
        if i == 0:
            input, time, label, location = Generate_Batch(x_array, x_label, memory_size, date_split, time_step)

            memory = (location, time[:, :time_step[0]], input[:, :time_step[0]], time[:, time_step[0]:], torch.zeros(memory_size, time_step[1], 1).cuda(), label)

        input, time, label, location = Generate_Batch(x_array, x_label, batch_size, date_split, time_step)

        model.construct_memory(memory[0], memory[1], memory[2], memory[3], memory[4], memory[5], True if i == iteration - 1 else False)
        infer, logit = model(location, time[:, :time_step[0]], input[:, :time_step[0]], time[:, time_step[0]:], torch.zeros(batch_size, time_step[1], 1).cuda())

        optimizer.zero_grad()
        loss = [nn.MSELoss()(infer, input[:, time_step[0]:, :1]), ExtremeValueLoss(logit, torch.eye(3)[label + 1].cuda(), (number_sample[0], number_sample[1], number_sample[2]))]
        (loss[0] + loss[1]).backward()
        optimizer.step()

        print(i + 1, '/', iteration, loss[0].item(), loss[1].item())

    torch.save(model, 'model.pt')
    torch.save(model.state_dict(), 'weights.pt')

    #########################################################################

    model = Transformer(hidden_size=32, output_size=1, num_layers=2, memory_size=memory_size, time_step=time_step).cuda()
    model.load_state_dict(torch.load('weights.pt'))
    model.eval()

    if not os.path.exists('output'):
        os.makedirs('output')

    for h in range(len(y_array)):
        index  = []
        infer  = []
        label  = []
        result = [[], [], []]

        for i in range(time_step[0], len(y_array[h]) - time_step[1] + 1):
            data = np.array([y_array[h][i - time_step[0]:i + time_step[1]]])

            if len(data[np.isnan(data)]) == 0:
                input = Variable(torch.from_numpy(data)).type(torch.FloatTensor).cuda()
                time  = []
            
                date = datetime.datetime(int(date_split[h][1][:4]), int(date_split[h][1][4:6]), int(date_split[h][1][6:]))
                date = date + datetime.timedelta(days=i - time_step[0])

                for t in range(time_step[0] + time_step[1]):
                    month = np.ones(1,) * (date.date().month - 1)
                    day   = np.ones(1,) * (date.date().day - 1)

                    time.append(np.concatenate([month, day]))
                    date = date + datetime.timedelta(days=1)

                time = Variable(torch.from_numpy(np.array([time]))).type(torch.FloatTensor).cuda()

                location = np.zeros(16,)
                location[h] = 1
                location = Variable(torch.from_numpy(np.array([location]))).type(torch.FloatTensor).cuda()

                index.append(np.array(y_label[h][i:i + time_step[1]]))
                infer.append(model(location, time[:, :time_step[0]], input[:, :time_step[0]], time[:, time_step[0]:], torch.zeros(1, time_step[1], 1).cuda())[0][0].cpu().data.numpy())
            else:
                index.append(np.ones_like(y_label[h][i:i + time_step[1]]) * np.nan)
                infer.append(np.ones_like(y_array[h][i:i + time_step[1], :1]) * np.nan)

            label.append(np.array(y_array[h][i:i + time_step[1], :1]))

        index = np.array(index)
        infer = np.array(infer) * std[h] + mean[h]
        label = np.array(label) * std[h] + mean[h]

        for i in range(time_step[1]):
            _index = index[:, i]
            _infer = infer[:, i, 0]
            _label = label[:, i, 0]

            a = ma.masked_invalid(_infer)
            b = ma.masked_invalid(_label)
            mask = (~a.mask & ~b.mask)

            result[0].append(np.corrcoef(a[mask], b[mask])[0][1])

            a = ma.masked_invalid(_infer[_index == 0])
            b = ma.masked_invalid(_label[_index == 0])
            mask = (~a.mask & ~b.mask)

            result[1].append(np.corrcoef(a[mask], b[mask])[0][1])

            a = ma.masked_invalid(_infer[_index != 0])
            b = ma.masked_invalid(_label[_index != 0])
            mask = (~a.mask & ~b.mask)

            result[2].append(np.corrcoef(a[mask], b[mask])[0][1])

        CORR[0].append(np.array(result[0]))
        CORR[1].append(np.array(result[1]))
        CORR[2].append(np.array(result[2]))

        result = [[], [], []]

        for i in range(time_step[1]):
            _index = index[:, i]
            _infer = infer[:, i, 0]
            _label = label[:, i, 0]

            a = ma.masked_invalid(_infer)
            b = ma.masked_invalid(_label)
            mask = (~a.mask & ~b.mask)

            result[0].append(np.sqrt(((a[mask] - b[mask]) ** 2).mean(axis=0)))

            a = ma.masked_invalid(_infer[_index == 0])
            b = ma.masked_invalid(_label[_index == 0])
            mask = (~a.mask & ~b.mask)

            result[1].append(np.sqrt(((a[mask] - b[mask]) ** 2).mean(axis=0)))

            a = ma.masked_invalid(_infer[_index != 0])
            b = ma.masked_invalid(_label[_index != 0])
            mask = (~a.mask & ~b.mask)

            result[2].append(np.sqrt(((a[mask] - b[mask]) ** 2).mean(axis=0)))

        RMSE[0].append(np.array(result[0]))
        RMSE[1].append(np.array(result[1]))
        RMSE[2].append(np.array(result[2]))

        with open('output/' + name[h] + '.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['date'] + list(np.arange(time_step[1]) + 1))

            date = datetime.datetime(int(date_split[h][1][:4]), int(date_split[h][1][4:6]), int(date_split[h][1][6:]))
            date = date + datetime.timedelta(days=time_step[0] + 1)

            for i in range(len(infer) + time_step[1] - 1):
                row = ['{}{:02d}{:02d}'.format(date.year, date.month, date.day)]
                idx = None

                for t in range(time_step[1]):
                    if 0 <= i - t < len(infer):
                        row += [infer[i - t][t][0]]
                        idx = index[i - t][t]
                    else:
                        row += ['']

                writer.writerow(row)
                date = date + datetime.timedelta(days=1)

    CORR[0] = np.swapaxes(CORR[0], 0, 1)
    CORR[1] = np.swapaxes(CORR[1], 0, 1)
    CORR[2] = np.swapaxes(CORR[2], 0, 1)
    RMSE[0] = np.swapaxes(RMSE[0], 0, 1)
    RMSE[1] = np.swapaxes(RMSE[1], 0, 1)
    RMSE[2] = np.swapaxes(RMSE[2], 0, 1)

    path = ['result.csv', 'normal_result.csv', 'extreme_result.csv']

    for h in range(3):
        with open(path[h], 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')

            writer.writerow(['CORR'] + name)

            for i in range(len(CORR[h])):
                row = [str(i + 1)]

                for j in range(len(CORR[h][i])):
                    row.append(CORR[h][i][j])
                
                writer.writerow(row)

            writer.writerow([''])
            writer.writerow(['RMSE'] + name)

            for i in range(len(RMSE[h])):
                row = [str(i + 1)]

                for j in range(len(RMSE[h][i])):
                    row.append(RMSE[h][i][j])
                
                writer.writerow(row)
