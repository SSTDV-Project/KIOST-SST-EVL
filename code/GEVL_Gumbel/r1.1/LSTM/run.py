import csv
import datetime
import numpy as np
import numpy.ma as ma
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from model import LSTM, GumbelGEVL
from torch.autograd import Variable

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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
    '../../../../../../data/YD.dat',
    '../../../../../../data/NH.dat',
    '../../../../../../data/TY.dat',
    '../../../../../../data/YS.dat',
    '../../../../../../data/WD.dat',
    '../../../../../../data/MP.dat',
    '../../../../../../data/BH.dat',
    '../../../../../../data/CB.dat',
    '../../../../../../data/PH.dat',
    '../../../../../../data/GM.dat',
    '../../../../../../data/GJ.dat',
    '../../../../../../data/DJ.dat',
    '../../../../../../data/DH.dat',
    '../../../../../../data/MR.dat',
    '../../../../../../data/YY.dat',
    '../../../../../../data/UL.dat',
]

if __name__ == '__main__' :
    batch_size = 128
    iteration  = 50000
    time_step  = [14, 7]
    x_array    = []
    y_array    = []
    CORR       = []
    RMSE       = []

    def Generate_Batch(array, batch_size, date_split, time_step):
        x_batch = []
        y_batch = []
        z_batch = []

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
                y_batch.append(time)
                z_batch.append(location)

        x_batch = np.array(x_batch)
        x_batch = Variable(torch.from_numpy(x_batch)).type(torch.FloatTensor).cuda()

        y_batch = np.array(y_batch)
        y_batch = Variable(torch.from_numpy(y_batch)).type(torch.FloatTensor).cuda()
    
        z_batch = np.array(z_batch)
        z_batch = Variable(torch.from_numpy(z_batch)).type(torch.FloatTensor).cuda()
    
        return x_batch, y_batch, z_batch

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
                current_date = prev_date + datetime.timedelta(days=j + 1)
                new_data.append(np.ones_like(x_data[0]) * np.nan)
                new_data[-1][0] = '{}{:02d}{:02d}'.format(current_date.year, current_date.month, current_date.day)

            new_data.append(x_data[i])

        '''
        with open('data/train/{}.csv'.format(name[h]), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['date', 'SST'])

            for i in range(len(new_data)):
                writer.writerow([np.int32(new_data[i][0]), new_data[i][1]])
        '''

        x_array.append(np.array(new_data)[:, 1:])

        y_data = data.loc[pd.IndexSlice[date_split[h][1]:date_split[h][2]], :]
        y_data = np.stack([y_data.loc[:, 'date'], y_data.loc[:, 'SST']], axis=1)[1:]

        new_data = [y_data[0]]

        for i in range(1, len(y_data)):
            prev_date = datetime.datetime(int(y_data[i - 1][0]) // 10000, (int(y_data[i - 1][0]) // 100) % 100, int(y_data[i - 1][0]) % 100)
            next_date = datetime.datetime(int(y_data[i][0]) // 10000, (int(y_data[i][0]) // 100) % 100, int(y_data[i][0]) % 100)

            for j in range((next_date - prev_date).days - 1):
                current_date = prev_date + datetime.timedelta(days=j + 1)
                new_data.append(np.ones_like(y_data[0]) * np.nan)
                new_data[-1][0] = '{}{:02d}{:02d}'.format(current_date.year, current_date.month, current_date.day)

            new_data.append(y_data[i])

        '''
        with open('data/test/{}.csv'.format(name[h]), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['date', 'SST'])

            for i in range(len(new_data)):
                writer.writerow([np.int32(new_data[i][0]), new_data[i][1]])
        '''

        y_array.append(np.array(new_data)[:, 1:])

        mean.append(np.nanmean(x_array[h], axis=0))
        std.append(np.nanstd(x_array[h], axis=0))

        x_array[h] = (x_array[h] - mean[h]) / std[h]
        y_array[h] = (y_array[h] - mean[h]) / std[h]

    '''
    with open('data/statistics.csv'.format(name[h]), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['location', 'SST_mean', 'SST_std'])

        for h in range(len(path)):
            writer.writerow([name[h], mean[h][0], std[h][0]])
    '''

    model = LSTM(hidden_size=32, output_size=1, num_layers=2).cuda()
    # model.load_state_dict(torch.load('weights.pt'))
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)

    for i in range(iteration):
        sst, time, location = Generate_Batch(x_array, batch_size, date_split, time_step)

        optimizer.zero_grad()
        loss = GumbelGEVL(model(location, time[:, :time_step[0]], sst[:, :time_step[0]], time[:, time_step[0]:], torch.zeros(batch_size, time_step[1], 1).cuda()), sst[:, time_step[0]:])
        loss.backward()
        optimizer.step()

        print(i + 1, '/', iteration, loss.item())

    torch.save(model, 'model.pt')
    torch.save(model.state_dict(), 'weights.pt')

    ############################################

    model = LSTM(hidden_size=32, output_size=1, num_layers=2).cuda()
    model.load_state_dict(torch.load('weights.pt'))
    model.eval()

    if not os.path.exists('output'):
        os.makedirs('output')

    for h in range(len(y_array)):
        infer  = []
        label  = []
        result = []

        for i in range(time_step[0], len(y_array[h]) - time_step[1] + 1):
            data = np.array([y_array[h][i - time_step[0]:i + time_step[1]]])

            if len(data[np.isnan(data)]) == 0:
                sst  = Variable(torch.from_numpy(data)).type(torch.FloatTensor).cuda()
                time = []
            
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

                infer.append(model(location, time[:, :time_step[0]], sst[:, :time_step[0]], time[:, time_step[0]:], torch.zeros(1, time_step[1], 1).cuda())[0].cpu().data.numpy())
            else:
                infer.append(np.ones_like(y_array[h][i:i + time_step[1]]) * np.nan)

            label.append(np.array(y_array[h][i:i + time_step[1]]))

        infer = np.array(infer) * std[h] + mean[h]
        label = np.array(label) * std[h] + mean[h]

        for i in range(time_step[1]):
            a = ma.masked_invalid(infer[:, i, 0])
            b = ma.masked_invalid(label[:, i, 0])
            mask = (~a.mask & ~b.mask)

            result.append(np.corrcoef(a[mask], b[mask])[0][1])

        CORR.append(np.array(result))

        result = []

        for i in range(time_step[1]):
            a = ma.masked_invalid(infer[:, i, 0])
            b = ma.masked_invalid(label[:, i, 0])
            mask = (~a.mask & ~b.mask)

            result.append(np.sqrt(((a[mask] - b[mask]) ** 2).mean(axis=0)))

        RMSE.append(np.array(result))

        with open('output/' + name[h] + '.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['date'] + list(np.arange(time_step[1]) + 1))

            date = datetime.datetime(int(date_split[h][1][:4]), int(date_split[h][1][4:6]), int(date_split[h][1][6:]))
            date = date + datetime.timedelta(days=time_step[0] + 1)

            for i in range(len(infer) + time_step[1] - 1):
                row = ['{}{:02d}{:02d}'.format(date.year, date.month, date.day)]

                for t in range(time_step[1]):
                    if 0 <= i - t < len(infer):
                        row += [infer[i - t][t][0]]
                    else:
                        row += ['']

                writer.writerow(row)
                date = date + datetime.timedelta(days=1)

    CORR = np.swapaxes(CORR, 0, 1)
    RMSE = np.swapaxes(RMSE, 0, 1)

    with open('result.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')

        writer.writerow(['CORR'] + name)

        for i in range(len(CORR)):
            row = [str(i + 1)]

            for j in range(len(CORR[i])):
                row.append(CORR[i][j])
                
            writer.writerow(row)

        writer.writerow([''])
        writer.writerow(['RMSE'] + name)

        for i in range(len(RMSE)):
            row = [str(i + 1)]

            for j in range(len(RMSE[i])):
                row.append(RMSE[i][j])
                
            writer.writerow(row)
