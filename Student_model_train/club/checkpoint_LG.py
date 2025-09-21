import torch
import os

class checkpoint_lg():
    def __init__(self,metric=None,checkpoint_dir=None):
        if metric == "acc":
            self.best_metric = 0
        elif metric == "loss":
            self.best_metric = torch.inf
        self.metirc=metric
        self.checkpoint_dir=checkpoint_dir
    def init_checkpoint_dir(self):
        if self.checkpoint_dir is not None:
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)

    def save(self,epoch_metric,model,optimizer,epoch,path):
        self.init_checkpoint_dir()
        tsave_dict = {
            'state_dict': model.state_dict(),
            "optimizer": optimizer.state_dict(),
            self.metirc: self.best_metric,
            "epoch": epoch
        }
        torch.save(tsave_dict, path)
        print("saving epoch-{} best={} cur-{}={}".format(epoch,self.best_metric,
                                                         self.metirc, epoch_metric))
        print("saving  checkpoint:{}".format(path))


    def save_best_checkpoint(self,epoch_metric,model,optimizer,epoch,path):
        if "acc" in self.metirc:
            if epoch_metric>self.best_metric:
                print("{}>{}".format(epoch_metric,self.best_metric))

                self.save(epoch_metric,model,optimizer,epoch,path)
                self.best_metric = epoch_metric

        elif "loss" in self.metirc:
            if epoch_metric<self.best_metric:
                print("{}<{}".format(epoch_metric,self.best_metric))

                self.save(epoch_metric,model,optimizer,epoch,path)
                self.best_metric = epoch_metric

    def save_epoch_checkpoint(self,
                              epoch_metric,
                              model,
                              optimizer,
                              epoch,
                              ckp_dir,
                              name,
                              epoch_frq,
                              seed):

        if epoch%epoch_frq==0:
            epoch_name = "{}_epoch{}_seed{}.pth".format(name,epoch,seed)
            path = os.path.join(ckp_dir, epoch_name)
            self.save(epoch_metric,model,optimizer,epoch,path)
        best_name = "{}_{}_seed{}.pth".format(name, "best", seed)
        best_path = os.path.join(ckp_dir, best_name)
        self.save_best_checkpoint(epoch_metric, model, optimizer, epoch, best_path)



    def load_checkpoint(self,model=None,optimizer=None,path=None):
        checkpoint=torch.load(path,map_location="cpu")

        print("loading  checkpoint:{}".format(path))

        # print("loading epoch:{}--{}:{}".format(checkpoint["epoch"],self.metirc,checkpoint[self.metirc]))

        if model is not None and optimizer is None:
            model.load_state_dict(checkpoint['state_dict'])
            return model
        elif model is not None and optimizer is not None:
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            return model,optimizer





