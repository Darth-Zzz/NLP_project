#coding=utf8

class Evaluator():

    def acc(self, predictions, labels):
        metric_dicts = {}
        metric_dicts['acc'] = self.accuracy(predictions, labels)
        metric_dicts['fscore'] = self.fscore(predictions, labels)
        return metric_dicts

    @staticmethod
    def accuracy(predictions, labels):
        # correct_values, total_values = 0, 0
        # correct_slots, total_slots = 0, 0
        corr, total = 0, 0
        for i, pred in enumerate(predictions):
            total += 1
            # if (set(pred) != set(labels[i])):
            #     print(pred, labels[i])
            values = [l.split("-")[-1] for l in pred]
            gt_values = [l.split("-")[-1] for l in labels[i]]
            # slots = [l.split("-")[1] for l in pred]
            # gt_slots = [l.split("-")[1] for l in labels[i]]
            # correct_values += set(values) == set(gt_values)
            # total_values += 1
            # correct_slots += set(slots) == set(gt_slots)
            # total_slots += 1
            
            corr += set(pred) == set(labels[i])
        # print("correct_values", correct_values, "total_values", total_values)
        # print("correct_slots", correct_slots, "total_slots", total_slots)
        return 100 * corr / total

    @staticmethod
    def fscore(predictions, labels):
        TP, TP_FP, TP_FN = 0, 0, 0
        for i in range(len(predictions)):
            pred = set(predictions[i])
            label = set(labels[i])
            TP += len(pred & label)
            TP_FP += len(pred)
            TP_FN += len(label)
        if TP_FP == 0:
            precision = 0
        else:
            precision = TP / TP_FP
        recall = TP / TP_FN
        if precision + recall == 0:
            fscore = 0
        else:
            fscore = 2 * precision * recall / (precision + recall)
        return {'precision': 100 * precision, 'recall': 100 * recall, 'fscore': 100 * fscore}
