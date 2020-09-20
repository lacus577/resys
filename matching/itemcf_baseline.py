import os
from collections import defaultdict

import pickle
import numpy as np

import conf, constant
from utils import LabelTransf, Reading_data, PathProcess, data_prepare
from evaluation.matching_evaluation import MatchingEvaluation

def itemcf(action_df, matching_num):
    """
    itemcf 召回
    :param matching_num:  召回数量
    :return:
    """
    # 只保留点击行为
    click_action_df = \
        action_df[action_df[constant.CLICK] == 1][[constant.USER_ID, constant.ITEM_ID, constant.CLICK]].reset_index(drop=True)
    print('click aciton num:{}'.format(click_action_df.shape[0]))

    if PathProcess().is_file_exist(os.path.join(conf.root_caching_path, conf.itemcf_sim_mat)):
        sim_mat = PathProcess().model_load(
            os.path.join(conf.root_caching_path, conf.itemcf_sim_mat)
        )
    else:
        sim_mat = itemcf_training(training_data=click_action_df)
        # save
        PathProcess().model_save(sim_mat, os.path.join(conf.root_caching_path, conf.itemcf_sim_mat))

    return itemcf_inference(action_df, sim_mat, matching_num)

def itemcf_training(training_data):
    # user-item倒排表
    user2item_df = training_data.groupby(constant.USER_ID).agg(
        {constant.ITEM_ID: lambda x: list(x)}
    ).reset_index()

    sim_mat = {}
    for row in user2item_df.itertuples():
        # TODO user点击历史是否要去重
        item_id_list = getattr(row, constant.ITEM_ID)
        for i in range(len(item_id_list)):
            if not sim_mat.get(item_id_list[i]):
                sim_mat[item_id_list[i]] = {}

            for j in range(i + 1, len(item_id_list)):
                if not sim_mat.get(item_id_list[j]):
                    sim_mat[item_id_list[j]] = {}

                if not sim_mat[item_id_list[i]].get(item_id_list[j]):
                    sim_mat[item_id_list[i]][item_id_list[j]] = 1
                    sim_mat[item_id_list[j]][item_id_list[i]] = 1
                else:
                    sim_mat[item_id_list[i]][item_id_list[j]] += 1
                    sim_mat[item_id_list[j]][item_id_list[i]] += 1

    # 计算每个item的uv
    item2user_df = training_data.groupby(constant.ITEM_ID).agg(
        {constant.USER_ID: lambda x: list(x)}
    ).reset_index()
    item2user_dict = defaultdict(set)
    for row in item2user_df.itertuples():
        item_id = getattr(row, constant.ITEM_ID)
        user_id_list = getattr(row, constant.USER_ID)
        item2user_dict[item_id].update(user_id_list)

    for k, v in sim_mat.items():
        for k_inner, v_inner in v.items():
            sim_mat[k][k_inner] = \
                1.0 * sim_mat[k][k_inner] / max(1, len(item2user_dict[k])) / max(1, len(item2user_dict[k_inner]))

    return sim_mat


def itemcf_inference(action_df, sim_mat, matching_num, is_matching_clicked=False):
    """
    is_matching_clicked=True允许召回历史点击过的item
    :param click_action_df:
    :param sim_mat:
    :param matching_num:
    :return:
    """
    matching_res = defaultdict(list)

    click_action_df = action_df[action_df[constant.CLICK] == 1]
    click_action_df = click_action_df.groupby(constant.USER_ID).agg(
        {constant.ITEM_ID: lambda x: list(x)}
    ).reset_index(drop=False)
    for row in click_action_df.itertuples():
        user_id = getattr(row, constant.USER_ID)
        item_id_list = getattr(row, constant.ITEM_ID)

        near_item_score_dict = defaultdict(np.float32)
        for item in item_id_list:
            # user历史点击item的k近邻
            if not sim_mat.get(item):
                continue

            # top item相似集合
            for k, v in sorted(sim_mat[item].items(), key=lambda x: x[1], reverse=True)[: matching_num]:
                # 不允许召回历史点击 并且 k不在历史点击里
                if sim_mat[item].get(k) and (is_matching_clicked or k not in item_id_list):
                    # todo 非对称
                    near_item_score_dict[k] += sim_mat[item][k]

        matching_res[user_id] = sorted(near_item_score_dict.items(), key=lambda x: x[1], reverse=True)[: matching_num]

    return matching_res


def itemcf_train(action_df):
    # 只保留点击行为
    click_action_df = \
        action_df[action_df[constant.CLICK] == 1][[constant.USER_ID, constant.ITEM_ID, constant.CLICK]].reset_index(
            drop=True)
    print('click aciton num:{}'.format(click_action_df.shape[0]))

    if PathProcess().is_file_exist(os.path.join(conf.root_caching_path, conf.itemcf_sim_mat)):
        sim_mat = PathProcess().model_load(
            os.path.join(conf.root_caching_path, conf.itemcf_sim_mat)
        )
    else:
        sim_mat = itemcf_training(training_data=click_action_df)
        # save
        PathProcess().model_save(sim_mat, os.path.join(conf.root_caching_path, conf.itemcf_sim_mat))

    return sim_mat
    # return itemcf_inference(action_df, sim_mat, matching_num)

def do_itemcf():
    train_action_df, _, _, \
    test_action_df, _, _ = data_prepare()

    pred = itemcf(train_action_df, conf.itemcf_matching_num)

    pred = LabelTransf().label_transf_v1(pred)
    truth = LabelTransf().label_transf_v2(test_action_df)
    print('itemcf recall@k:', MatchingEvaluation(truth, pred).topk_recall())


# todo baseline recall:
# itemcf recall@k: 0.007987164427367018
if __name__ == '__main__':
    do_itemcf()
