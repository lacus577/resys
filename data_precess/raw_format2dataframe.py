import os

import numpy as np
import pandas as pd

import conf, constant
from data_precess.sample import data_sample, data_sample_v1

sample_skeleton_columns = ['SampleID', 'click', 'buy', 'md5', 'feature_num', 'feature_list']
common_features_columns = ['md5', 'feature_num', 'feature_list']

def raw_format2dataframe_and_save(
        sample_skeleton_train_path, common_features_train_path, skeleton_save_path, common_features_save_path):
    impression_sample_table = pd.read_table(sample_skeleton_train_path,
                                 sep=',', header=None, names=sample_skeleton_columns, engine='python')

    entire_fea_dict = {}
    for k, v in constant.skeletion_field_id2name.items():
        entire_fea_dict[v] = []
        entire_fea_dict[str(v) + 'Value'] = []
    for index, row in impression_sample_table.iterrows():
        feature_arr = row['feature_list'].split('\001')
        fea_dict = {}
        fea_value_dict = {}
        for k, v in constant.skeletion_field_id2name.items():
            fea_dict[k] = []
            fea_value_dict[k] = []
        for fea_kv in feature_arr:
            fea_field_id = fea_kv.split('\002')[0]
            fea_id_val = fea_kv.split('\002')[1]
            fea_id = fea_id_val.split('\003')[0]
            fea_val = fea_id_val.split('\003')[1]

            fea_dict[fea_field_id].append(fea_id)
            fea_value_dict[fea_field_id].append(fea_val)

        for k, v in fea_dict.items():
            if len(v) == 0:
                entire_fea_dict[constant.skeletion_field_id2name[k]].append('')
            else:
                entire_fea_dict[constant.skeletion_field_id2name[k]].append('|'.join(v))

        for k, v in fea_value_dict.items():
            if len(v) == 0:
                entire_fea_dict[str(constant.skeletion_field_id2name[k]) + 'Value'].append('')
            else:
                entire_fea_dict[str(constant.skeletion_field_id2name[k]) + 'Value'].append('|'.join(v))

        if index % 10000 == 0:
            print("current_index:", index)

    entire_fea_table = pd.DataFrame(data=entire_fea_dict, columns=list(entire_fea_dict.keys()))
    impression_sample_table = impression_sample_table.drop('feature_list', axis=1)
    sample_table = pd.concat([impression_sample_table, entire_fea_table], axis=1)

    sample_table.to_csv(skeleton_save_path, index=False)

def raw_format2dataframe_and_save_v1(
        raw_data_path, raw_columns, id2name_dict, data_save_path):
    impression_sample_table = pd.read_table(raw_data_path,
                                 sep=',', header=None, names=raw_columns, engine='python')
    entire_fea_dict = {}
    for k, v in id2name_dict.items():
        entire_fea_dict[v] = []
        entire_fea_dict[str(v) + 'Value'] = []
    for index, row in impression_sample_table.iterrows():
        feature_arr = row['feature_list'].split('\001')
        fea_dict = {}
        fea_value_dict = {}
        for k, v in id2name_dict.items():
            fea_dict[k] = []
            fea_value_dict[k] = []
        for fea_kv in feature_arr:
            fea_field_id = fea_kv.split('\002')[0]
            fea_id_val = fea_kv.split('\002')[1]
            fea_id = fea_id_val.split('\003')[0]
            fea_val = fea_id_val.split('\003')[1]

            fea_dict[fea_field_id].append(fea_id)
            fea_value_dict[fea_field_id].append(fea_val)

        for k, v in fea_dict.items():
            if len(v) == 0:
                entire_fea_dict[id2name_dict[k]].append('')
            else:
                entire_fea_dict[id2name_dict[k]].append('|'.join(sorted(v)))

        for k, v in fea_value_dict.items():
            if len(v) == 0:
                entire_fea_dict[str(id2name_dict[k]) + 'Value'].append('')
            else:
                entire_fea_dict[str(id2name_dict[k]) + 'Value'].append('|'.join(sorted(v)))

        if index % 10000 == 0:
            print("current_index:", index)

    entire_fea_table = pd.DataFrame(data=entire_fea_dict, columns=list(entire_fea_dict.keys()))
    impression_sample_table = impression_sample_table.drop('feature_list', axis=1)
    sample_table = pd.concat([impression_sample_table, entire_fea_table], axis=1)

    sample_table.to_csv(data_save_path, index=False)

def raw_format2dataframe_v1(
        raw_data_path, raw_columns, id2name_dict):
    impression_sample_table = pd.read_table(raw_data_path,
                                 sep=',', header=None, names=raw_columns, engine='python')
    entire_fea_dict = {}
    for k, v in id2name_dict.items():
        entire_fea_dict[v] = []
        entire_fea_dict[str(v) + 'Value'] = []
    for index, row in impression_sample_table.iterrows():
        feature_arr = row['feature_list'].split('\001')
        fea_dict = {}
        fea_value_dict = {}
        for k, v in id2name_dict.items():
            fea_dict[k] = []
            fea_value_dict[k] = []
        for fea_kv in feature_arr:
            fea_field_id = fea_kv.split('\002')[0]
            fea_id_val = fea_kv.split('\002')[1]
            fea_id = fea_id_val.split('\003')[0]
            fea_val = fea_id_val.split('\003')[1]

            fea_dict[fea_field_id].append(fea_id)
            fea_value_dict[fea_field_id].append(fea_val)

        for k, v in fea_dict.items():
            if len(v) == 0:
                entire_fea_dict[id2name_dict[k]].append('')
            else:
                entire_fea_dict[id2name_dict[k]].append('|'.join(sorted(v)))

        for k, v in fea_value_dict.items():
            if len(v) == 0:
                entire_fea_dict[str(id2name_dict[k]) + 'Value'].append('')
            else:
                entire_fea_dict[str(id2name_dict[k]) + 'Value'].append('|'.join(sorted(v)))

        if index % 10000 == 0:
            print("current_index:", index)

    entire_fea_table = pd.DataFrame(data=entire_fea_dict, columns=list(entire_fea_dict.keys()))
    impression_sample_table = impression_sample_table.drop('feature_list', axis=1)
    sample_table = pd.concat([impression_sample_table, entire_fea_table], axis=1)

    return sample_table
    # sample_table.to_csv(data_save_path, index=False)

# def do_raw_format2dataframe_and_save():
#     sample_skeleton_train_path = conf.raw_train_sampled_skeleton_path.format(str(conf.sampling_rate).split('.')[1])
#     common_features_train_path = conf.raw_train_sampled_common_features_path.format(str(conf.sampling_rate).split('.')[1])
#     if not os.path.exists(sample_skeleton_train_path) or not os.path.exists(common_features_train_path):
#         data_sample(
#             conf.raw_train_common_features_path, conf.raw_train_sample_skeleton_path,
#             conf.sampling_rate,
#             conf.raw_train_sampled_common_features_path, conf.raw_train_sampled_skeleton_path
#         )
#
#     raw_format2dataframe_and_save_v1(
#         sample_skeleton_train_path, sample_skeleton_columns, constant.skeletion_field_id2name,
#         conf.train_sample_skeleton_path)
#     raw_format2dataframe_and_save_v1(
#         common_features_train_path, common_features_columns, constant.common_features_field_id2name,
#         conf.train_common_features_path)
#
#     # 拆成user、item、action
#     sample_skeleton_df = pd.read_csv(conf.train_sample_skeleton_path)
#     common_features_df = pd.read_csv(conf.train_common_features_path)
#     total_df = sample_skeleton_df.merge(common_features_df, how='inner', on='md5')
#     item_df = total_df[constant.ITEM_COLUMNS].drop_duplicates(constant.ITEM_COLUMNS, keep='last')
#     user_df = total_df[constant.USER_COLUMS].drop_duplicates(constant.USER_COLUMS, keep='last')
#     action_df = total_df[constant.ACTION_COLUMNS].drop_duplicates(constant.ACTION_COLUMNS, keep='last')
#
#     # clean
#     action_df = action_df[~action_df[constant.USER_ID].isna()]
#     action_df = action_df[~action_df[constant.ITEM_ID].isna()]
#     item_df = item_df[~item_df[constant.ITEM_ID].isna()]
#     user_df = user_df[~user_df[constant.USER_ID].isna()]
#
#     item_df.to_csv(conf.train_item_path, index=False)
#     user_df.to_csv(conf.train_user_path, index=False)
#     action_df.to_csv(conf.train_action_path, index=False)

def do_raw_format2dataframe_and_save_v1(
        raw_common_features_path,
        raw_sample_skeleton_path,
        raw_sampled_common_features_path,
        raw_sampled_skeleton_path,
        item_path, user_path, action_path
):
    """
    随机采样
    :param raw_common_features_path: 原始下载的common_features csv路径
    :param raw_sample_skeleton_path: 原始下载的sample_skeleton csv路径
    :param raw_sampled_common_features_path: 采样后的common_feature csv路径
    :param raw_sampled_skeleton_path: 采样后的sample_skeleton csv路径
    :param item_path:
    :param user_path:
    :param action_path:
    :return:
    """
    sample_skeleton_path = raw_sampled_skeleton_path.format(str(conf.sampling_rate).split('.')[1])
    common_features_path = raw_sampled_common_features_path.format(str(conf.sampling_rate).split('.')[1])
    if not os.path.exists(sample_skeleton_path) or not os.path.exists(common_features_path):
        data_sample(
            raw_common_features_path, raw_sample_skeleton_path,
            conf.sampling_rate,
            raw_sampled_common_features_path, raw_sampled_skeleton_path
        )

    sample_skeleton_df = raw_format2dataframe_v1(
        sample_skeleton_path, sample_skeleton_columns, constant.skeletion_field_id2name)
    common_features_df = raw_format2dataframe_v1(
        common_features_path, common_features_columns, constant.common_features_field_id2name)

    # 拆成user、item、action
    # sample_skeleton_df = pd.read_csv(conf.train_sample_skeleton_path)
    # common_features_df = pd.read_csv(conf.train_common_features_path)

    total_df = sample_skeleton_df.merge(common_features_df, how='inner', on='md5')
    item_df = total_df[constant.ITEM_COLUMNS].drop_duplicates(constant.ITEM_COLUMNS, keep='last')
    user_df = total_df[constant.USER_COLUMS].drop_duplicates(constant.USER_COLUMS, keep='last')
    action_df = total_df[constant.ACTION_COLUMNS].drop_duplicates(constant.ACTION_COLUMNS, keep='last')

    # clean
    action_df = action_df[~action_df[constant.USER_ID].isna()]
    action_df = action_df[~action_df[constant.ITEM_ID].isna()]
    item_df = item_df[~item_df[constant.ITEM_ID].isna()]
    user_df = user_df[~user_df[constant.USER_ID].isna()]

    item_df.to_csv(item_path, index=False)
    user_df.to_csv(user_path, index=False)
    action_df.to_csv(action_path, index=False)

def do_raw_format2dataframe_and_save_v2(
        train_sampled_user_set,
        raw_common_features_path,
        raw_sample_skeleton_path,
        raw_sampled_common_features_path,
        raw_sampled_skeleton_path,
        item_path, user_path, action_path
):
    """
    用于测试集， 根据训练集随机采样的user来采样
    :param raw_common_features_path: 原始下载的common_features csv路径
    :param raw_sample_skeleton_path: 原始下载的sample_skeleton csv路径
    :param raw_sampled_common_features_path: 采样后的common_feature csv路径
    :param raw_sampled_skeleton_path: 采样后的sample_skeleton csv路径
    :param item_path:
    :param user_path:
    :param action_path:
    :return:
    """
    sample_skeleton_path = raw_sampled_skeleton_path.format(str(conf.sampling_rate).split('.')[1])
    common_features_path = raw_sampled_common_features_path.format(str(conf.sampling_rate).split('.')[1])
    if not os.path.exists(sample_skeleton_path) or not os.path.exists(common_features_path):
        data_sample_v1(
            train_sampled_user_set,
            raw_common_features_path, raw_sample_skeleton_path,
            conf.sampling_rate,
            raw_sampled_common_features_path, raw_sampled_skeleton_path
        )

    sample_skeleton_df = raw_format2dataframe_v1(
        sample_skeleton_path, sample_skeleton_columns, constant.skeletion_field_id2name)
    common_features_df = raw_format2dataframe_v1(
        common_features_path, common_features_columns, constant.common_features_field_id2name)

    # 拆成user、item、action
    # sample_skeleton_df = pd.read_csv(conf.train_sample_skeleton_path)
    # common_features_df = pd.read_csv(conf.train_common_features_path)
    total_df = sample_skeleton_df.merge(common_features_df, how='inner', on='md5')
    item_df = total_df[constant.ITEM_COLUMNS].drop_duplicates(constant.ITEM_COLUMNS, keep='last')
    user_df = total_df[constant.USER_COLUMS].drop_duplicates(constant.USER_COLUMS, keep='last')
    action_df = total_df[constant.ACTION_COLUMNS].drop_duplicates(constant.ACTION_COLUMNS, keep='last')

    # clean
    action_df = action_df[~action_df[constant.USER_ID].isna()]
    action_df = action_df[~action_df[constant.ITEM_ID].isna()]
    item_df = item_df[~item_df[constant.ITEM_ID].isna()]
    user_df = user_df[~user_df[constant.USER_ID].isna()]

    item_df.to_csv(item_path, index=False)
    user_df.to_csv(user_path, index=False)
    action_df.to_csv(action_path, index=False)

if __name__ == '__main__':
    pass