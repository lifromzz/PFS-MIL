import torch
class stop_early_lg():
    def __init__(self,
                 patient=50,
                 metric="acc"):
        self.patient=patient
        self.count=0
        self.metric=metric

        if metric=="acc":
            self.best_metric=0
        elif metric=="loss":
            self.best_metric=torch.inf

    def stop(self,cur_metric):
        if "acc" in self.metric:
            if cur_metric>self.best_metric:
                self.best_metric=cur_metric
                self.count = 0
            else:
                self.count += 1
        elif "loss" in self.metric:
            if cur_metric < self.best_metric:
                self.best_metric=cur_metric
                self.count = 0
            else:
                self.count +=1

        if self.count>self.patient:
            return True
        else:
            return False