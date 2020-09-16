import os

import conf
from utils import data_prepare
from matching.itemcf_baseline import do_itemcf_v1
from featuring.featuring import Featuring
from ranking.gbdt_lr import GbdtLR

def main():
    # 训练阶段
    train(conf.train_action_path, conf.test_action_path)

    # 推理阶段
    infernece()


def train(train_data_path, test_action_path):
    # 数据准备
    train_action_df, test_action_df = data_prepare(train_data_path, test_action_path)

    # 召回
    matching_res = do_itemcf_v1(train_action_df, test_action_df)

    # 召回结果整理
    # action_df, user_df, item_df = data_compose_from_matching(
    #     [matching_res],
    #     conf.train_action_path,
    #     conf.test_action_path
    # )

    # 排序
    train_action_df = Reading_data().reading_action_data(conf.train_action_path)
    train_user_df = Reading_data().reading_user_data(conf.train_user_path)
    train_item_df = Reading_data().reading_item_data(conf.train_item_path)
    test_action_df = Reading_data().reading_action_data(conf.test_action_path)
    test_user_df = Reading_data().reading_user_data(conf.test_user_path)
    test_item_df = Reading_data().reading_item_data(conf.test_item_path)

    # featuring
    train = Featuring(train_action_df, train_user_df, train_item_df).gbdt_lr_v1()
    test = Featuring(test_action_df, test_user_df, test_item_df).gbdt_lr_v1()
    valid, test = TrainValidTestSplit().valid_test_split_timeseq(test.sample(frac=1).reset_index(drop=True), 0.2)

    # gbdt_lr = GbdtLR(train, valid, test)
    # gbdt_lr.fit()
    # gbdt_lr.save_model(conf.gbdt_model_save_path, conf.lr_model_save_path)
    # gbdt_lr.predict()
    # 排序特征工程
    train = Featuring(action_df, user_df, item_df).gbdt_lr_v1()
    test = Featuring(test_action_df, test_user_df, test_item_df).gbdt_lr_v1()
    valid, test = TrainValidTestSplit().valid_test_split_timeseq(test.sample(frac=1).reset_index(drop=True), 0.2)
    # 排序模型
    GbdtLR()

    # 评估

if __name__ == '__main__':
    main()
