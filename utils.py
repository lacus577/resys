import os

import pandas as pd
import numpy as np
from collections import defaultdict
import pickle

import constant, conf
from data_precess.raw_format2dataframe import do_raw_format2dataframe_and_save_v1, do_raw_format2dataframe_and_save_v2


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

        action_df.loc[:, 'SampleID'] = action_df['SampleID'].astype(np.int)
        action_df.loc[:, 'SampleID'] = action_df['SampleID'].astype(np.str)

        action_df.loc[:, 'UserID'] = action_df['UserID'].astype(np.int)
        action_df.loc[:, 'UserID'] = action_df['UserID'].astype(np.str)

        action_df.loc[:, 'ItemID'] = action_df['ItemID'].astype(np.int)
        action_df.loc[:, 'ItemID'] = action_df['ItemID'].astype(np.str)

        action_df.loc[:, 'click'] = action_df['click'].astype(np.int)
        action_df.loc[:, 'buy'] = action_df['buy'].astype(np.int)

        action_df.loc[:, 'SceneID'] = action_df['SceneID'].astype(np.int)
        action_df.loc[:, 'SceneID'] = action_df['SceneID'].astype(np.str)

        action_df.loc[:, 'ComposeCateAndActions'] = action_df['ComposeCateAndActions'].astype(np.int)
        action_df.loc[:, 'ComposeCateAndActions'] = action_df['ComposeCateAndActions'].astype(np.str)

        action_df.loc[:, 'ComposeShopAndActions'] = action_df['ComposeShopAndActions'].astype(np.int)
        action_df.loc[:, 'ComposeShopAndActions'] = action_df['ComposeShopAndActions'].astype(np.str)

        action_df.loc[:, 'ComposeBrandAndActions'] = action_df['ComposeBrandAndActions'].astype(np.int)
        action_df.loc[:, 'ComposeBrandAndActions'] = action_df['ComposeBrandAndActions'].astype(np.str)

        action_df.loc[:, 'ComposeIntentAndActions'] = action_df['ComposeIntentAndActions'].astype(np.str)

        return action_df

    def reading_user_data(self, data_path):
        # ['UserID', 'UserCateActions', 'UserShopActions', 'UserBrandActions', 'UserIntentActions',
        # 'UserCate1', 'UserCate2', 'UserGender', 'UserAge', 'UserComsumptionLevel1',
        # 'UserComsumptionLevel2', 'UserWorking', 'UserGeo']
        user_df = pd.read_csv(data_path)

        user_df = user_df.fillna(value=0)

        user_df.loc[:, 'UserID'] = user_df['UserID'].astype(np.int)
        user_df.loc[:, 'UserID'] = user_df['UserID'].astype(np.str)

        user_df.loc[:, 'UserCateActions'] = user_df['UserCateActions'].astype(np.str)
        user_df.loc[:, 'UserShopActions'] = user_df['UserShopActions'].astype(np.str)
        user_df.loc[:, 'UserBrandActions'] = user_df['UserBrandActions'].astype(np.str)
        user_df.loc[:, 'UserIntentActions'] = user_df['UserIntentActions'].astype(np.str)

        user_df.loc[:, 'UserCate1'] = user_df['UserCate1'].astype(np.int)
        user_df.loc[:, 'UserCate1'] = user_df['UserCate1'].astype(np.str)

        user_df.loc[:, 'UserCate2'] = user_df['UserCate2'].astype(np.int)
        user_df.loc[:, 'UserCate2'] = user_df['UserCate2'].astype(np.str)

        user_df.loc[:, 'UserGender'] = user_df['UserGender'].astype(np.int)
        user_df.loc[:, 'UserGender'] = user_df['UserGender'].astype(np.str)

        user_df.loc[:, 'UserAge'] = user_df['UserAge'].astype(np.int)
        user_df.loc[:, 'UserAge'] = user_df['UserAge'].astype(np.str)

        user_df.loc[:, 'UserComsumptionLevel1'] = user_df['UserComsumptionLevel1'].astype(np.int)
        user_df.loc[:, 'UserComsumptionLevel1'] = user_df['UserComsumptionLevel1'].astype(np.str)

        user_df.loc[:, 'UserComsumptionLevel2'] = user_df['UserComsumptionLevel2'].astype(np.int)
        user_df.loc[:, 'UserComsumptionLevel2'] = user_df['UserComsumptionLevel2'].astype(np.str)

        user_df.loc[:, 'UserWorking'] = user_df['UserWorking'].astype(np.int)
        user_df.loc[:, 'UserWorking'] = user_df['UserWorking'].astype(np.str)

        user_df.loc[:, 'UserGeo'] = user_df['UserGeo'].astype(np.int)
        user_df.loc[:, 'UserGeo'] = user_df['UserGeo'].astype(np.str)

        return user_df

    def reading_item_data(self, data_path):
        # ['ItemID', 'ItemCateID', 'ItemShopID', 'ItemIntentID', 'ItemBrandID']
        item_df = pd.read_csv(data_path)

        item_df = item_df.fillna(0)
        item_df.loc[:, 'ItemID'] = item_df['ItemID'].astype(np.int)
        item_df.loc[:, 'ItemID'] = item_df['ItemID'].astype(np.str)

        item_df.loc[:, 'ItemCateID'] = item_df['ItemCateID'].astype(np.int)
        item_df.loc[:, 'ItemCateID'] = item_df['ItemCateID'].astype(np.str)

        item_df.loc[:, 'ItemShopID'] = item_df['ItemShopID'].astype(np.int)
        item_df.loc[:, 'ItemShopID'] = item_df['ItemShopID'].astype(np.str)

        item_df.loc[:, 'ItemIntentID'] = item_df['ItemIntentID'].astype(np.str)

        item_df.loc[:, 'ItemBrandID'] = item_df['ItemBrandID'].astype(np.int)
        item_df.loc[:, 'ItemBrandID'] = item_df['ItemBrandID'].astype(np.str)

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
            res[k] = list(set(item_list))

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
                res[user_id] = list(set(item_list))

        return res

    def label_transf_v3(self, sample_df):
        """
        预测结果转换成标签字典
        :param action_df:
        :return:
        """
        # 过滤出有click历史的行为
        # action_df = action_df[action_df[constant.CLICK] == 1][[constant.USER_ID, constant.ITEM_ID, constant.CLICK]]
        sample_df = sample_df.sort_values(constant.CLICK_PRED, ascending=False).reset_index(drop=True)
        sample_df = sample_df.groupby(constant.USER_ID).agg(
            {constant.ITEM_ID: lambda x: list(x), constant.CLICK: lambda x: list(x)}
        ).reset_index()

        res = defaultdict(list)
        for row in sample_df.itertuples():
            user_id = getattr(row, constant.USER_ID)
            item_list = getattr(row, constant.ITEM_ID)
            if len(item_list) > 0:
                res[user_id] = list(set(item_list))

        return res


class TrainValidTestSplit(object):
    def __init__(self):
        pass

    def train_valid_split_v0(self, action_df):
        """
        将训练集中每个user的最后一次点击作为valid
        :return:
        """
        action_df = action_df.sort_values([constant.SAMPLE_ID], axis=0, ascending=True)
        valid_action_df = action_df.drop_duplicates([constant.USER_ID], keep='last').reset_index(drop=True)
        train_action_df = \
            action_df[~ action_df[constant.SAMPLE_ID].isin(valid_action_df[constant.SAMPLE_ID])].reset_index(drop=True)

        return train_action_df, valid_action_df

    def valid_test_split_v0(self, data_df, frac):
        """
        测试集中划分出valid和test两个集合
        :param data_df:
        :param frac:
        :return:
        """
        len = int(data_df.shape[0] * frac)
        valid = data_df.iloc[: len, :]
        test = data_df.iloc[len:, :]

        return valid, test


def data_prepare():
    if not os.path.exists(PathProcess().get_action_path(conf.train_root_path)):
        do_raw_format2dataframe_and_save_v1(
            PathProcess().path_combine(conf.train_root_path, conf.raw_train_common_features_file_name),
            PathProcess().path_combine(conf.train_root_path, conf.raw_train_sample_skeleton_file_name),
            PathProcess().path_combine(conf.train_root_path, conf.raw_train_sampled_common_features_file_name),
            PathProcess().path_combine(conf.train_root_path, conf.raw_train_sampled_skeleton_file_name),
            PathProcess().get_item_path(conf.train_root_path),
            PathProcess().get_user_path(conf.train_root_path),
            PathProcess().get_action_path(conf.train_root_path)
        )
    train_action_df = Reading_data().reading_action_data(PathProcess().get_action_path(conf.train_root_path))
    train_user_df = Reading_data().reading_user_data(PathProcess().get_user_path(conf.train_root_path))
    train_item_df = Reading_data().reading_item_data(PathProcess().get_item_path(conf.train_root_path))

    # 测试集采样训练集随机采样结果中的user， 防止出现冷启动， 本版本暂时不解决冷启动问题
    if not os.path.exists(PathProcess().get_action_path(conf.test_root_path)):
        do_raw_format2dataframe_and_save_v2(
            set(train_action_df[constant.USER_ID]),
            PathProcess().path_combine(conf.test_root_path, conf.raw_test_common_features_file_name),
            PathProcess().path_combine(conf.test_root_path, conf.raw_test_sample_skeleton_file_name),
            PathProcess().path_combine(conf.test_root_path, conf.raw_test_sampled_common_features_file_name),
            PathProcess().path_combine(conf.test_root_path, conf.raw_test_sampled_skeleton_file_name),
            PathProcess().get_item_path(conf.test_root_path),
            PathProcess().get_user_path(conf.test_root_path),
            PathProcess().get_action_path(conf.test_root_path)
        )
    test_action_df = Reading_data().reading_action_data(PathProcess().get_action_path(conf.test_root_path))
    test_user_df = Reading_data().reading_user_data(PathProcess().get_user_path(conf.test_root_path))
    test_item_df = Reading_data().reading_item_data(PathProcess().get_item_path(conf.test_root_path))

    return train_action_df, train_user_df, train_item_df, \
           test_action_df, test_user_df, test_item_df


class PathProcess(object):
    def __init__(self):
        pass

    def get_action_path(self, root_path):
        return self.path_combine(root_path, conf.action_file_name.format(self.my_str(conf.sampling_rate)))

    def get_user_path(self, root_path):
        return self.path_combine(root_path, conf.user_file_name.format(self.my_str(conf.sampling_rate)))

    def get_item_path(self, root_path):
        return self.path_combine(root_path, conf.item_file_name.format(self.my_str(conf.sampling_rate)))

    def my_str(self, float_str):
        return str(float_str).split('.')[1] + '_'

    def path_combine(self, a, b):
        return os.path.join(a, b)

    def model_save(self, model, model_path):
        pickle.dump(model, open(model_path, 'wb'))

    def model_load(self, model_path):
        return pickle.load(open(model_path, 'rb'))

    def is_file_exist(self, file_path):
        return os.path.exists(file_path)


def matching_res2action_df(matching_res, action_df, user_df, item_df):
    key1_list = []
    key2_list = []
    value_list = []
    for user_id, v in matching_res.items():
        for v_v in v:
            key1_list.append(user_id)
            key2_list.append(v_v[0])
            value_list.append(v_v[1])

    matching_user_item_df = pd.DataFrame()
    matching_user_item_df[constant.USER_ID] = key1_list
    matching_user_item_df[constant.ITEM_ID] = key2_list

    matching_action_df = matching_user_item_df.merge(action_df, on=[constant.USER_ID, constant.ITEM_ID], how='left')
    matching_action_df = matching_action_df.drop_duplicates([constant.SAMPLE_ID], keep='last').reset_index(drop=True)
    matching_user_df = matching_user_item_df[[constant.USER_ID]].merge(user_df, on=constant.USER_ID, how='left')
    matching_user_df = matching_user_df.drop_duplicates(constant.USER_COLUMS, keep='last').reset_index(drop=True)
    matching_item_df = matching_user_item_df[[constant.ITEM_ID]].merge(item_df, on=constant.ITEM_ID, how='left')
    matching_item_df = matching_item_df.drop_duplicates(constant.ITEM_COLUMNS, keep='last').reset_index(drop=True)

    return matching_action_df, matching_user_df, matching_item_df