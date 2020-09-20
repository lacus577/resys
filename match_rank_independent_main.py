import os

import conf, constant
from utils import data_prepare, PathProcess, LabelTransf, matching_res2action_df
from matching.itemcf_baseline import itemcf_train, itemcf_inference
from featuring.featuring import Featuring
from ranking.gbdt_lr import GbdtLR
from evaluation.matching_evaluation import MatchingEvaluation

def main():
    # 训练阶段
    train()

    # 推理阶段
    inference()


def train():
    # 数据准备
    train_action_df, train_user_df, train_item_df, _, _, _ = data_prepare()

    # 召回训练
    print('matching train start')
    itemcf_train(train_action_df)
    print('matching train end\n')

    # 排序特征工程
    print('ranking featuring start')
    train, valid = Featuring(train_action_df, train_user_df, train_item_df, train_valid_split=True).gbdt_lr_v2()
    print('ranking featuring end\n')

    # 排序模型训练
    print('ranking train start')
    gbdt_lr = GbdtLR(train, valid, evl='auc')
    gbdt_lr.fit()
    gbdt_lr.save_model(
        os.path.join(conf.root_caching_path, conf.gbdt_model_save_file_name),
        os.path.join(conf.root_caching_path, conf.lr_model_save_file_name)
    )
    print('ranking train end\n')


def inference():
    # test_df 数据准备
    train_action_df, train_user_df, train_item_df, \
    test_action_df, test_user_df, test_item_df = data_prepare()

    # 测试集召回
    print('matching inference start')
    matching_res = itemcf_inference(
        train_action_df,
        PathProcess().model_load(PathProcess().path_combine(conf.root_caching_path, conf.itemcf_sim_mat)),
        conf.itemcf_matching_num
    )
    pred = LabelTransf().label_transf_v1(matching_res)
    truth = LabelTransf().label_transf_v2(test_action_df)
    print('itemcf recall@k:', MatchingEvaluation(truth, pred).topk_recall())
    print('matching inference end\n')

    check(matching_res, test_action_df)

    # 召回结果处理成排序要求格式
    matching_action_df, matching_user_df, matching_item_df = \
        matching_res2action_df(matching_res, test_action_df, test_user_df, test_item_df)

    # 排序特征工程
    print('ranking featuring start')
    matching_action_df.fillna(0, inplace=True)
    matching_user_df.fillna(0, inplace=True)
    matching_item_df.fillna(0, inplace=True)
    feature_obj = Featuring(matching_action_df, matching_user_df, matching_item_df, train_valid_split=False)
    test_sample_df = feature_obj.samples()
    test_feature_df = feature_obj.gbdt_lr_v2(test_sample_df)
    print('ranking featuring end\n')

    # 排序
    print('ranking inference start')
    gbdt_lr = GbdtLR(test=test_feature_df, evl='auc')
    gbdt_lr.load_model(
        PathProcess().path_combine(conf.root_caching_path, conf.gbdt_model_save_file_name),
        PathProcess().path_combine(conf.root_caching_path, conf.lr_model_save_file_name)
    )
    test_pred_y = gbdt_lr.predict()
    print('ranking inference end\n')

    # 端到端评估
    test_sample_df[constant.CLICK_PRED] = test_pred_y
    pred = LabelTransf().label_transf_v3(test_sample_df)
    print('end2end NDCG@k:', MatchingEvaluation(truth, pred).my_ndcg(100))


def check(matching_res, action_df):
    res = 0
    action_df = action_df[action_df[constant.CLICK] == 1]

    for user_id, v in matching_res.items():
        item_list = list(action_df[action_df[constant.USER_ID] == user_id][constant.ITEM_ID])
        for v_v in v:
            item_id = v_v[0]
            if item_id in item_list:
                res += 1
                print(user_id, item_id)
        if res > 10:
            break
    print('********', res)



if __name__ == '__main__':
    main()
