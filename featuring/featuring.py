from collections import defaultdict
import random

import pandas as pd
import numpy as np
from tqdm import tqdm

import constant, conf
from utils import TrainValidTestSplit


class Featuring(object):
    def __init__(self, action_df, user_df, item_df, train_valid_split=False):
        self.action = action_df
        self.user = user_df
        self.item = item_df
        self.train_valid_split = train_valid_split
        self.total = \
            self.action.merge(self.user, on=constant.USER_ID, how='left').merge(self.item, on=constant.ITEM_ID, how='left')

    def postive_samples(self):
        return self.total[self.total[constant.CLICK] == 1].reset_index(drop=True)

    def neg_samples(self):
        """
        负样本生成方式：曝光未点击 随机采样
        :return:
        """
        pos = self.postive_samples()
        user2clickeditem = pos.groupby(constant.USER_ID).agg(
            {
                constant.ITEM_ID: lambda x: list(x)
            }
        ).reset_index(drop=False)

        neg_samples = self.total[self.total[constant.CLICK] == 0]
        user2unclickeditem = neg_samples.groupby(constant.USER_ID).agg(
            {
                constant.ITEM_ID: lambda x: list(x)
            }
        ).reset_index()
        user2unclickeditem_dict = defaultdict(list)
        for row in tqdm(user2unclickeditem.itertuples()):
            user_id = getattr(row, constant.USER_ID)
            item_list = getattr(row, constant.ITEM_ID)
            user2unclickeditem_dict[user_id] = item_list

        neg_dict = defaultdict(list)
        for row in tqdm(user2clickeditem.itertuples()):
            user_id = getattr(row, constant.USER_ID)
            item_list = getattr(row, constant.ITEM_ID)

            sampling_neg_num = conf.neg_pos_rate * len(set(item_list))
            cur_user_expose = user2unclickeditem_dict[user_id]
            if len(cur_user_expose) <= 0:
                pass
            elif len(cur_user_expose) <= sampling_neg_num:
                # neg = neg.append(cur_user_expose)
                neg_dict[user_id] = [cur_user_expose]
            else:
                # neg = neg.append(self.random_sampling(cur_user_expose, sampling_neg_num))
                neg_dict[user_id] = [self.random_sampling_from_list(cur_user_expose, sampling_neg_num)]

        user2negsamples_df = pd.DataFrame(data=neg_dict).stack().unstack(0).reset_index()
        user2negsamples_df.columns = [constant.USER_ID, constant.ITEM_ID]
        # 拆开item_id
        user2negsamples_df = user2negsamples_df.explode(constant.ITEM_ID)
        user2negsamples_df = user2negsamples_df.merge(neg_samples, on=[constant.USER_ID, constant.ITEM_ID], how='left')

        samples = user2negsamples_df.append(pos)

        return samples

    def samples(self):
        return self.neg_samples()

    def gbdt_lr_v1(self):
        """
        只做了采样， 没有划分数据集
        :return:
        """
        samples_df = self.samples()
        print(
            'neg sampling end. expect neg_pos rate:{}, sampled neg_pos rate:{}'.format(
                conf.neg_pos_rate,
                np.sum(samples_df[constant.CLICK] == 0) // np.sum(samples_df[constant.CLICK] == 1)
            )
        )

        return samples_df[conf.baseline_features_columns + [constant.CLICK]].reset_index(drop=True)

    def gbdt_lr_v2(self, samples_df=None):
        """
        采样之后，划分训练集和验证集
        :return:
        """
        if samples_df is None:
            samples_df = self.samples()

        print(
            'neg sampling end. expect neg_pos rate:{}, sampled neg_pos rate:{}'.format(
                conf.neg_pos_rate,
                np.sum(samples_df[constant.CLICK] == 0) // np.sum(samples_df[constant.CLICK] == 1)
            )
        )

        if not self.train_valid_split:
            return samples_df[conf.baseline_features_columns + [constant.CLICK]].reset_index(drop=True)
        else:
            train, valid = TrainValidTestSplit().train_valid_split_v0(samples_df)
            return train[conf.baseline_features_columns + [constant.CLICK]].reset_index(drop=True), \
                   valid[conf.baseline_features_columns + [constant.CLICK]].reset_index(drop=True)

    def random_sampling(self, df, random_num):
        df = df.reset_index(drop=True)
        return df.sample(n=random_num, random_state=1, axis=0)

    def random_sampling_from_list(self, data_list, random_num):
        return random.sample(data_list, random_num)
