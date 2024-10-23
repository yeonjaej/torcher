import torch,yaml,time
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import torcher.util
from torcher.util import configure_model, configure_train, configure_inference
from torcher.util import verify_config_keys, VALID_CONFIG_KEYS
import numpy as np
import h5py


class trainer(torch.nn.Module):

    def __init__(self,cfg=None):
        super().__init__()
        self.to(torcher.util.DEFAULT_DEVICE)
        if not cfg is None:
            result = self.configure(cfg)
        self.log = dict(epoch=[],
            time_epoch=[],time_iter=[],time_pre=[],time_post=[],time_train=[],
            loss=[],loss_epoch=[],
            )

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
     
    def save_output(self, out_data):

        if not self.out_file or not out_data:
            return

        file_name = self.out_file % self.trained_epochs
        print(f'Saving the inference output {file_name}')
        with h5py.File(file_name,'w') as f:

            for key,val in out_data.items():
                grp = f.create_group(key)
                for k,v in val.items():
                    if not hasattr(v[0],'__len__'):
                        grp.create_dataset(k,data=np.array(v))
                    elif len(v[0].shape) < 1:
                        grp.create_dataset(k,data=np.stack(v))
                    else:
                        grp.create_dataset(k,data=np.concatenate(v))
        print('Finished saving the output')


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
        log = dict()
        epoch = 0.
        log=dict(time_iter=[],time_pre=[],time_post=[],time_forward=[])
        t0=time.time()
        with torch.inference_mode():
            self.model.eval()
            iter2epoch = 1./int(len(self.inference_data) / self.inference_loader.batch_size)

            for data,label in self.inference_loader:

                self.move_dict_device(data,self.device)
                self.move_dict_device(label,self.device)
                log['time_pre'].append(time.time()-t0)
                t1=time.time()

                prediction = self.model(data)
                log['time_forward'].append(time.time()-t1)
                t1=time.time()
                epoch += iter2epoch

                if hasattr(self,'criterion'):
                    loss = self.criterion(prediction, label)
                    loss_record.append(loss['loss'].detach().item())

                if self.out_file:

                    for dic in [data,label,prediction,loss]:
                        self.dict_tensor_to_numpy(dic)
                    dict_data = dict(input=data, label=label, output=prediction, loss=loss)

                    if out_data is None:
                        out_data = dict()
                        for key,val in dict_data.items():
                            out_data[key]={k:[v] for k,v in val.items()}
                    else:
                        for key,val in dict_data.items():
                            for k,v in val.items():
                                out_data[key][k].append(v)

                if self.inference_log_file is not None:
                    self.append_scalar_elements(loss,log)

                log['time_post'].append(time.time()-t1)
                log['time_iter'].append(time.time()-t0)
                t0=time.time()

                if epoch >= self.inference_epochs:
                    break

        if len(loss_record):
            print(f'Inference loss average: {np.mean(loss_record)}')

        if out_data is not None:
            self.save_output(out_data)

        if self.inference_log_file is not None:
            if len(loss_record):
                log['loss']=loss_record
            np.savez(self.inference_log_file % self.trained_epochs, **log)



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

    def append_scalar_elements(self,data,log_dict):
        for k,v in data.items():
            if hasattr(v,'__len__'):
                continue
            if not k in log_dict:
                log_dict[k]=[]
            log_dict[k].append(v)

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
            tepoch=time.time()
            for data,label in self.train_loader:

                # Move data torch.Tensor type data attributes to the right device
                self.move_dict_device(data,self.device)
                self.move_dict_device(label,self.device)
                self.log['time_pre'].append(time.time() - t0)
                t1=time.time()

                # Train step                
                output,loss = self.train_step(data,label)
                self.log['time_train'].append(time.time() - t1)
                t1=time.time()

                # Log any scalar values from the loss dictionary
                self.log['loss'].append(loss['loss'].item())
                self.append_scalar_elements(loss,self.log)

                # Update epoch
                self.log['epoch'].append(self.trained_epochs)
                self.trained_epochs += iter2epoch

                # Post processing

                # save snapshot and log
                if self.trained_epochs >= (save_counter * self.save_frequency):
                    save_counter += 1
                    self.save_state()
                    if self.train_log_file:
                        np.savez(self.train_log_file,**self.log)

                self.log['time_post'].append(time.time() - t1)

                # Report
                if self.trained_epochs >= (report_counter * self.train_report_frequency):
                    report_counter += 1

                    total_time = self.log["time_pre"][-1] + self.log["time_train"][-1] + self.log["time_post"][-1]
                    time_pre   = 100.*self.log[ "time_pre"   ][-1]/total_time
                    time_train = 100.*self.log[ "time_train" ][-1]/total_time
                    time_post  = 100.*self.log[ "time_post"  ][-1]/total_time
                    msg  = 'Epoch %06.2f ... Loss %05.5f ... Time %02.2f' % (self.trained_epochs,loss['loss'].item(),total_time)
                    msg += ' (Preproc %02.2f%%  Train %02.2f%%  Postproc %02.2f%%)' % (time_pre,time_train,time_post)
                    print(msg)

                self.log['time_iter'].append(time.time()-t0)
                t0=time.time()

                if self.trained_epochs > max_epoch:
                    return

            print(f'Train loss average over last epoch: {np.mean(self.log["loss"][-1*len(self.train_loader):])}')
            self.log['time_epoch'].append(time.time()-tepoch)
            self.log['loss_epoch'].append(self.log['loss'][-1])
            if hasattr(self,'inference_data'):
                self.inference()
