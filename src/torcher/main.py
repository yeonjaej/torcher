import torch,yaml,time
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import torcher.util
from torcher.util import configure_model, configure_train, configure_inference
from torcher.util import verify_config_keys, VALID_CONFIG_KEYS
import numpy as np


class trainer(torch.nn.Module):

    def __init__(self,cfg=None):
        super().__init__()
        self.to(torcher.util.DEFAULT_DEVICE)
        if not cfg is None:
            result = self.configure(cfg)
        self.log = dict(time_epoch=[],time_iter=[],time_pre=[],time_post=[],time_train=[],loss=[])

    def to(self,device):
        super().to(device)
        self.device=device
        if hasattr(self,'model'):
            self.model = self.model.to(device)

    def configure(self, cfg):
        if not 'Train' in cfg and not 'Inference' in cfg:
            print(yaml.dump(cfg,default_flow_style=False))
            raise KeyError('Needs either Train and/or Inference configuration')

        required_keys=['Model']
        optional_keys=['Train','Inference'] + [key for key in VALID_CONFIG_KEYS if not key in required_keys]
        verify_config_keys(cfg,required_keys,optional_keys)
        if 'Train' in cfg:
            train_cfg = cfg.pop('Train')
            configure_train(train_cfg,self)
        if 'Inference' in cfg:
            inference_cfg = cfg.pop('Inference')
            configure_inference(inference_cfg,self)
        configure_model(cfg,self)
        self.model = self.model.to(self.device)

    def save_state(self):
        fname = self.weight_file % self.trained_epochs
        data = dict(state_dict = self.model.state_dict(),
                    optimizer  = self.optimizer.state_dict(),
                    trained_epochs = self.trained_epochs)
        torch.save(data, fname)
     
    def train_step(self,input_data,target_data):
        t0=time.time()
        self.optimizer.zero_grad()
        output = self.model(input_data)
        result = self.criterion(output, target_data)
        result['loss'].backward()
        self.optimizer.step()
        #self.time['train']=time.time()-t0

        return output,result

    def inference(self):
        print(f'Running inference for {self.inference_epochs} epochs on the device {self.device}')

        out_data = None
        loss_record = []
        epoch = 0.
        with torch.inference_mode():
            self.model.eval()
            iter2epoch = 1./int(len(self.inference_data) / self.inference_loader.batch_size)

            for data,label in self.inference_loader:

                self.move_dict_device(data,self.device)
                self.move_dict_device(label,self.device)
                prediction = self.model(data)

                epoch += iter2epoch

                if hasattr(self,'criterion'):
                    loss = self.criterion(prediction, label)
                    loss_record.append(loss['loss'].detach().item())

                if self.out_file:
                    data.update(prediction)
                    data.update(label)
                    self.dict_tensor_to_numpy(data)

                    if out_data is None:
                        out_data = {key:[val] for key,val in data.items()}
                    else:
                        for key,val in data.items():
                            out_data[key].append(val)

                if epoch >= self.inference_epochs:
                    break

        if len(loss_record):
            print(f'Inference loss average: {np.mean(loss_record)}')

        if out_data is not None:
            file_name = self.out_file % self.trained_epochs
            print(f'Saving the inference output {file_name}')
            if len(loss_record):
                out_data['loss']=np.array(loss_record,dtype=float)
            np.savez(file_name,**out_data)

    
    def move_dict_device(self,data,device=None):
        if device is None: device = self.device 
        for key,val in data.items():
            if not isinstance(val,torch.Tensor):
                continue
            data[key] = val.to(device)

    def dict_tensor_to_numpy(self,data):
        for key,val in data.items():
            if not isinstance(val,torch.Tensor):
                continue
            data[key] = val.to('cpu').detach().numpy()

    def train(self):
        print(f'Running training for {self.train_epochs} epochs on the device {self.device}')
        iter2epoch = 1./int(len(self.train_data) / self.train_loader.batch_size)

        save_counter   = self.trained_epochs / self.save_frequency + 1
        report_counter = self.trained_epochs / self.train_report_frequency + 1

        if self.trained_epochs == 0.:
            self.save_state()

        max_epoch = self.trained_epochs + self.train_epochs
        while self.trained_epochs < max_epoch:

            self.model.train()
            t0 = time.time()
            for data,label in self.train_loader:

                # Move data torch.Tensor type data attributes to the right device
                self.move_dict_device(data,self.device)
                self.move_dict_device(label,self.device)
                self.log['time_pre'].append(time.time() - t0)
                t0=time.time()

                # Train step                
                output,loss = self.train_step(data,label)
                self.log['time_train'].append(time.time() - t0)
                self.log['loss'].append(loss['loss'].item())
                t0=time.time()

                # Update epoch
                self.trained_epochs += iter2epoch
                
                # Post processing

                # save snapshot
                if self.trained_epochs >= (save_counter * self.save_frequency):
                    save_counter += 1
                    self.save_state()

                self.log['time_post'].append(time.time() - t0)
                t0 = time.time()

                # Report
                if self.trained_epochs >= (report_counter * self.train_report_frequency):
                    report_counter += 1

                    total_time = self.log["time_pre"][-1] + self.log["time_train"][-1] + self.log["time_post"][-1]
                    time_pre   = 100.*self.log["time_pre"][-1]/total_time
                    time_train = 100.*self.log["time_train"][-1]/total_time
                    time_post  = 100.*self.log["time_post"][-1]/total_time
                    msg  = 'Epoch %06.2f ... Loss %05.5f ... Time %02.2f' % (self.trained_epochs,loss['loss'].item(),total_time)
                    msg += ' (Preproc %02.2f%%  Train %02.2f%%  Postproc %02.2f%%)' % (time_pre,time_train,time_post)
                    print(msg)

                if self.trained_epochs > max_epoch:
                    return

            print(f'Train loss average over last epoch: {np.mean(self.log["loss"][-1*len(self.train_loader):])}')

            if hasattr(self,'inference_data'):
                self.inference()
