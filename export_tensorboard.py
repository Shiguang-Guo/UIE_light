"""
@author: Guo Shiguang
@software: PyCharm
@file: export_tensorboard.py
@time: 2022/4/4 15:15
"""

import pandas as pd
from tensorboard.backend.event_processing import event_accumulator


def main():
    fin = '/data/guoshiguang/logdir/bang/ace2005-full-nar-150/train'
    fout = '/data/guoshiguang/logdir/bang/ace2005-full-nar-150/train.csv'

    event_data = event_accumulator.EventAccumulator(fin)  # a python interface for loading Event data
    event_data.Reload()  # synchronously loads all of the data written so far b
    loss = list(map(list, event_data.scalars.Items('loss')))  # print all tags
    f1 = list(map(list, event_data.scalars.Items('f1')))  # print all tags

    loss_df = pd.DataFrame(loss).drop(columns=0, axis=1)
    f1_df = pd.DataFrame(f1).drop(columns=0, axis=1)
    loss_df.columns = ['step', 'loss']
    f1_df.columns = ['step', 'f1']
    loss_df.set_index('step', inplace=True)
    f1_df.set_index('step', inplace=True)
    print(loss_df, f1_df)

    df = pd.concat([loss_df, f1_df], axis=1)

    # keys = event_data.scalars.Keys()  # get all tags,save in a list
    # print(keys)
    # df = pd.DataFrame(columns=['loss', 'f1'])
    # for key in ['loss', 'f1']:
    #     df[key] = pd.DataFrame(event_data.Scalars(key)).value
    # df.dropna(inplace=True)
    # df.index = pd.DataFrame(event_data.Scalars(key))['step']
    #
    df.to_csv(fout)

    print("Tensorboard data exported successfully")


if __name__ == '__main__':
    main()
