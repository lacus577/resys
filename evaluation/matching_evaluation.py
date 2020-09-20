import numpy as np


class MatchingEvaluation(object):
    def __init__(self, truth, pred):
        self.truth = truth
        self.pred = pred

    def topk_recall(self):
        """
        Embedding-based Retrieval in Facebook Search中使用的召回评估指标
        :return:
        """
        real_pred_users = len(set(self.truth.keys()).intersection(set(self.pred.keys())))
        avg_recall = 0
        for k, v in self.truth.items():
            if self.pred.get(k) is not None:
                truth_click_list = v
                pred_click_list = self.pred[k]
                avg_recall += 1.0 * len(set(truth_click_list).intersection(set(pred_click_list))) / len(
                    truth_click_list)

        avg_recall = avg_recall * 1.0 / real_pred_users

        return avg_recall

    def my_ndcg(self, k):
        # 1.test truth click 2.设定k值 3.pred，包括user及对应的itemlist，要求有顺序
        # 计算公式reg/log(i+1) / truth-DCG
        avg_ndcg = 0
        total_user_num = 0

        for user_id, v in self.pred.items():
            if self.truth.get(user_id) is not None:
                total_user_num += 1
                truth_click_item_list = self.truth[user_id][0:k]  # topk 截断
                idcg_topk = self.get_idcg(truth_click_item_list, k)
                one_user_dcg_topk = 0
                for i in range(0, min(k, len(v))):
                    if v[i] in truth_click_item_list:
                        one_user_dcg_topk += 1.0 / np.log2(i + 2.0)

                assert one_user_dcg_topk <= idcg_topk
                avg_ndcg += 1.0 * one_user_dcg_topk / idcg_topk

        avg_ndcg /= total_user_num

        return avg_ndcg

    def get_idcg(self, item_list, k):
        idcg = 0
        # item_list = sorted(item_list, reverse=False)
        for i in range(0, min(k, len(item_list))):
            idcg += 1.0 / np.log2(i + 2.0)

        return idcg
