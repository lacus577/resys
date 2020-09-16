

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
                avg_recall += 1.0 * len(set(truth_click_list).intersection(set(pred_click_list))) / len(truth_click_list)

        avg_recall = avg_recall * 1.0 / real_pred_users

        return avg_recall