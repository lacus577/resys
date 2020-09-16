import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
import warnings
import pickle

from tqdm import tqdm

warnings.filterwarnings('ignore')

import constant, conf
from utils import Reading_data
from featuring.featuring import Featuring
from utils import TrainValidTestSplit

class GbdtLR(object):
    def __init__(self, train, valid=None, test=None):
        self.train = train
        self.valid = valid
        self.test = test
        self.gbdt_model = None
        self.lr_model = None

    def fit(self):
        # 数据集
        train_x = self.train[set(self.train.columns).difference([constant.CLICK])]
        train_y = self.train[constant.CLICK]
        valid_x = self.valid[set(self.valid.columns).difference([constant.CLICK])]
        valid_y = self.valid[constant.CLICK]

        self.gbdt_model = self.gbdt_fit(train_x, train_y, valid_x, valid_y)
        self.lr_model = self.lr_fit(train_x, train_y)

    def gbdt_fit(self, train_x, train_y, valid_x, valid_y):
        print('开始训练gbdt..')
        gbm = lgb.LGBMClassifier(objective='binary',
                                # subsample=0.8,
                                # min_child_weight=0.5,
                                # colsample_bytree=0.7,
                                num_leaves=100,
                                # max_depth=12,
                                # learning_rate=0.05,
                                n_estimators=10,
                                )

        gbm.fit(train_x.values, train_y.values,
                eval_set=[(train_x.values, train_y.values), (valid_x.values, valid_y.values)],
                eval_names=['train', 'val'],
                eval_metric='auc'
                )

        return gbm

    def lr_fit(self, train_x, train_y):
        train_x = self.features_from_gbdt(train_x)

        print('开始训练lr...')
        lr = LogisticRegression()
        lr.fit(train_x.values, train_y.values)
        print('lr训练完毕...')

        return lr

    def features_from_gbdt(self, train_x):
        # print('训练得到叶子数')
        gbdt_feats_train = self.gbdt_model.booster_.predict(train_x, pred_leaf=True)
        gbdt_feats_name = ['gbdt_tree_' + str(i) for i in range(gbdt_feats_train.shape[1])]
        df_train_gbdt_feats = pd.DataFrame(gbdt_feats_train, columns=gbdt_feats_name)

        # 叶子数one-hot
        # print('开始one-hot...')
        num_leaves = self.gbdt_model.num_leaves
        for col in tqdm(gbdt_feats_name):
            # print('this is feature:', col)
            one_hot_feature_nd = np.zeros([df_train_gbdt_feats.shape[0], num_leaves], dtype=np.int)

            one_col_list = list(df_train_gbdt_feats[col])
            for i in range(len(one_col_list)):
                one_hot_feature_nd[i][one_col_list[i]] = 1

            one_hot_feature_df = \
                pd.DataFrame(data=one_hot_feature_nd, columns=[str(col) + '_' + str(i) for i in range(num_leaves)])

            # onehot_feats = pd.get_dummies(df_train_gbdt_feats[col], prefix=col)
            df_train_gbdt_feats.drop([col], axis=1, inplace=True)
            df_train_gbdt_feats = pd.concat([df_train_gbdt_feats, one_hot_feature_df], axis=1)
        # print('one-hot结束')

        return df_train_gbdt_feats

    def predict(self):
        train_x = self.train[set(self.train.columns).difference([constant.CLICK])]
        train_x = self.features_from_gbdt(train_x)
        train_y = self.train[constant.CLICK]

        valid_x = self.valid[set(self.valid.columns).difference([constant.CLICK])]
        valid_x = self.features_from_gbdt(valid_x)
        valid_y = self.valid[constant.CLICK]

        test_x = self.test[set(self.test.columns).difference([constant.CLICK])]
        test_x = self.features_from_gbdt(test_x)
        test_y = self.test[constant.CLICK]

        train_pred_y = self.lr_model.predict_proba(train_x)[:, 1]
        print('train set AUC:{}'.format(roc_auc_score(train_y, train_pred_y)))

        valid_pred_y = self.lr_model.predict_proba(valid_x)[:, 1]
        print('valid set AUC:{}'.format(roc_auc_score(valid_y, valid_pred_y)))

        test_pred_y = self.lr_model.predict_proba(test_x)[:, 1]
        print('test set AUC:{}'.format(roc_auc_score(test_y, test_pred_y)))

    def save_model(self, gbdt_path, lr_path):
        pickle.dump(self.gbdt_model, open(gbdt_path, 'wb'))
        pickle.dump(self.lr_model, open(lr_path, 'wb'))

if __name__ == '__main__':
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

    gbdt_lr = GbdtLR(train, valid, test)
    gbdt_lr.fit()
    gbdt_lr.save_model(conf.gbdt_model_save_path, conf.lr_model_save_path)
    gbdt_lr.predict()
