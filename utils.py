import os

import pandas as pd
import numpy as np
from collections import defaultdict

import constant, conf
from data_precess.raw_format2dataframe import do_raw_format2dataframe_and_save_v1

class Reading_data(object):
    def __init__(self):
        pass

    def reading_action_data(self, data_path):
        '''
        ['UserID', 'ItemID', 'ComposeCateAndActions', 'ComposeShopAndActions', '
        ComposeBrandAndActions', 'ComposeIntentAndActions', 'SceneID', 'click', 'buy']
        读取用户行为数据都使用这个接口， 保证数据格式一致
        :param data_path:
        :return:
        '''
        action_df = pd.read_csv(data_path)

        action_df = action_df.fillna(0)

        action_df.loc[:, 'UserID'] = action_df['UserID'].astype(np.int)
        action_df.loc[:, 'ItemID'] = action_df['ItemID'].astype(np.int)
        action_df.loc[:, 'click'] = action_df['click'].astype(np.int)
        action_df.loc[:, 'buy'] = action_df['buy'].astype(np.int)
        action_df.loc[:, 'SceneID'] = action_df['SceneID'].astype(np.int)
        action_df.loc[:, 'ComposeCateAndActions'] = action_df['ComposeCateAndActions'].astype(np.int)
        action_df.loc[:, 'ComposeShopAndActions'] = action_df['ComposeShopAndActions'].astype(np.int)
        action_df.loc[:, 'ComposeBrandAndActions'] = action_df['ComposeBrandAndActions'].astype(np.int)
        action_df.loc[:, 'ComposeIntentAndActions'] = action_df['ComposeIntentAndActions'].astype(np.str)

        return action_df

    def reading_user_data(self, data_path):
        # ['UserID', 'UserCateActions', 'UserShopActions', 'UserBrandActions', 'UserIntentActions',
        # 'UserCate1', 'UserCate2', 'UserGender', 'UserAge', 'UserComsumptionLevel1',
        # 'UserComsumptionLevel2', 'UserWorking', 'UserGeo']
        user_df = pd.read_csv(data_path)

        user_df = user_df.fillna(value=0)

        user_df.loc[:, 'UserID'] = user_df['UserID'].astype(np.int)
        user_df.loc[:, 'UserCateActions'] = user_df['UserCateActions'].astype(np.str)
        user_df.loc[:, 'UserShopActions'] = user_df['UserShopActions'].astype(np.str)
        user_df.loc[:, 'UserBrandActions'] = user_df['UserBrandActions'].astype(np.str)
        user_df.loc[:, 'UserIntentActions'] = user_df['UserIntentActions'].astype(np.str)
        user_df.loc[:, 'UserCate1'] = user_df['UserCate1'].astype(np.int)
        user_df.loc[:, 'UserCate2'] = user_df['UserCate2'].astype(np.int)
        user_df.loc[:, 'UserGender'] = user_df['UserGender'].astype(np.int)
        user_df.loc[:, 'UserAge'] = user_df['UserAge'].astype(np.int)
        user_df.loc[:, 'UserComsumptionLevel1'] = user_df['UserComsumptionLevel1'].astype(np.int)
        user_df.loc[:, 'UserComsumptionLevel2'] = user_df['UserComsumptionLevel2'].astype(np.int)
        user_df.loc[:, 'UserWorking'] = user_df['UserWorking'].astype(np.int)
        user_df.loc[:, 'UserGeo'] = user_df['UserGeo'].astype(np.int)

        return user_df

    def reading_item_data(self, data_path):
        # ['ItemID', 'ItemCateID', 'ItemShopID', 'ItemIntentID', 'ItemBrandID']
        item_df = pd.read_csv(data_path)

        item_df = item_df.fillna(0)
        item_df.loc[:, 'ItemID'] = item_df['ItemID'].astype(np.int)
        item_df.loc[:, 'ItemCateID'] = item_df['ItemCateID'].astype(np.int)
        item_df.loc[:, 'ItemShopID'] = item_df['ItemShopID'].astype(np.int)
        item_df.loc[:, 'ItemIntentID'] = item_df['ItemIntentID'].astype(np.str)
        item_df.loc[:, 'ItemBrandID'] = item_df['ItemBrandID'].astype(np.int)

        return item_df


class LabelTransf(object):
    """
    各种形式的标签格式转换成如下字典格式，要求得分最高的排在最前面：
    user1  item1 item2 item3 ...
    user2   item1 item2 ...
    user3   item2 item8 ...
    ...
    """

    def __init__(self):
        pass

    def label_transf_v1(self, label_dict):
        """
        字典类型转换成标签字典
        字典格式：
        42100: [(7616994, 1.0),
              (8096406, 1.0),
              (7653132, 1.0),
              (5330839, 1.0),
              (7628478, 1.0),
              (7478257, 1.0),
              (7864763, 1.0),
              (7249379, 0.5),
              (7369414, 0.25)]
        :param label_dict:
        :return:
        """
        res = defaultdict(list)
        for k, v in label_dict.items():
            v = sorted(v, key=lambda x: x[1], reverse=True)
            item_list = [item[0] for item in v]
            res[k] = item_list

        return res

    def label_transf_v2(self, action_df):
        """
        用户测试集行为历史转换成标签字典
        :param action_df:
        :return:
        """
        # 过滤出有click历史的行为
        action_df = action_df[action_df[constant.CLICK] == 1][[constant.USER_ID, constant.ITEM_ID, constant.CLICK]]
        action_df = action_df.groupby(constant.USER_ID).agg(
            {constant.ITEM_ID: lambda x: list(x), constant.CLICK: lambda x: list(x)}
        ).reset_index()

        res = defaultdict(list)
        for row in action_df.itertuples():
            user_id = getattr(row, constant.USER_ID)
            item_list = getattr(row, constant.ITEM_ID)
            if len(item_list) > 0:
                res[user_id] = item_list

        return res


class TrainValidTestSplit(object):
    def __init__(self):
        pass

    def train_valid_split_timeseq(self):
        pass

    def valid_test_split_timeseq(self, data_df, frac):
        len = int(data_df.shape[0] * frac)
        valid = data_df.iloc[: len, :]
        test = data_df.iloc[len:, :]

        return valid, test


def data_prepare(train_action_path, test_action_path):
    if not os.path.exists(train_action_path):
        do_raw_format2dataframe_and_save_v1(
            conf.raw_train_common_features_path,
            conf.raw_train_sample_skeleton_path,
            conf.raw_train_sampled_common_features_path,
            conf.raw_train_sampled_skeleton_path,
            conf.train_item_path, conf.train_user_path, conf.train_action_path
        )
    train_action_df = Reading_data().reading_action_data(conf.train_action_path)

    if not os.path.exists(test_action_path):
        do_raw_format2dataframe_and_save_v1(
            conf.raw_test_common_features_path,
            conf.raw_test_sample_skeleton_path,
            conf.raw_test_sampled_common_features_path,
            conf.raw_test_sampled_skeleton_path,
            conf.test_item_path, conf.test_user_path, conf.test_action_path
        )
    test_action_df = Reading_data().reading_action_data(conf.test_action_path)

    return train_action_df, test_action_df