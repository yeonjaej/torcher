import os, torch, yaml, inspect, pathlib, importlib
VALID_CONFIG_KEYS=['Data','DataLoader','Sampler','Collate','Optimizer','Criterion','Model','ModelWeights']
DEFAULT_DEVICE = 'cpu' if not torch.cuda.is_available() else 'cuda'

def check_prefix(prefix,create=False):
    if not '/' in prefix:
        return True
    path = pathlib.Path(prefix)
    if not os.path.isdir(path.parents[0]):
        try:
            os.makedirs(path.parents[0])
        except OSError:
            return False
    return True
    
def verify_config_keys(cfg,param_keys,optional_keys=[]):
    for needed in param_keys:
        if not needed in cfg:
            print(cfg)
            print(yaml.dump(cfg, default_flow_style=False))
            raise KeyError(f'The configuration is missing a required parameter: {needed}')
    for param in cfg:
        if not param in param_keys+optional_keys:
            print(f'[WARNING] ignoring an irrelevant parameter: {param}') 

def instantiate(cfg):
    needed_params = ['Module','Name']
    optional_params = ['Parameters']
    verify_config_keys(cfg,needed_params,optional_params)

    try:
        module = importlib.import_module(cfg['Module'])
    except ModuleNotFoundError:
        print(yaml.dump(cfg,default_flow_style=False))
        raise ModuleNotFoundError(f'Module name {cfg["Module"]} not found')

    class_name = cfg['Name']
    if not hasattr(module,class_name):
        raise KeyError(f'Invalid name: {class_name}')
    
    try:
        class_type = getattr(module,cfg['Name'])
        if not 'Parameters' in cfg or not cfg['Parameters']:
            instance = class_type()
        elif hasattr(class_type,"configure"):
            instance = class_type()
            instance.configure(**cfg['Parameters'])
        else:
            instance = class_type(**cfg['Parameters'])
    except TypeError:
        args = inspect.getfullargspec(class_type).args[1:]
        print(yaml.dump(cfg,default_flow_style=False))
        raise TypeError(f'Unexpected parameter found. Supported key words are: {args}')
    return instance    

def list_args(module,function_name,class_name=None):
    try:
        module = importlib.import_module(module)
    except ModuleNotFoundError:
        raise ModuleNotFoundError(f'Module {module} not found')

    if class_name is None:
        if not hasattr(module,function_name):
            raise KeyError(f'Invalid function name {function_name}')
        return inspect.getfullargspec(getattr(module,function_name)).args
    else:
        if not hasattr(module,class_name):
            raise KeyError(f'Invalid class name {class_name}')
        class_instance = getattr(module,class_name)
        if not hasattr(class_instance,function_name):
            raise KeyError(f'Invalid function name {function_name}')
        return inspect.getfullargspec(getattr(class_instance,function_name)).args[1:]


def configure_sampler(sampler_cfg,dataset=None):

    args = list_args(sampler_cfg['Module'],'__init__',sampler_cfg['Name'])
    if 'dataset' in args:
        if 'Parameters' in sampler_cfg:
            sampler_cfg['Parameters']['dataset'] = dataset
        else:
            sampler_cfg['Parameters']=dict(dataset=dataset)
    return instantiate(sampler_cfg)


def configure_collate(collate_cfg,dataset=None):
    args = list_args(collate_cfg['Module'],'__init__',collate_cfg['Name'])
    if 'dataset' in args:
        if 'Parameters' in collate_cfg:
            collate_cfg['Parameters']['dataset'] = dataset
        else:
            collate_cfg['Parameters']=dict(dataset=dataset)
    return instantiate(collate_cfg)


def configure_dataloader(loader_cfg,dataset,sampler=None,collate=None):
    loader_cfg['Parameters'].update(dict(dataset=dataset))
    if sampler:
        loader_cfg['Parameters'].update(dict(sampler=sampler))
    if collate:
        loader_cfg['Parameters'].update(dict(collate_fn=collate))
    return instantiate(loader_cfg)


def configure_model(cfg,obj=None):
    print('[configure_model] running')
    if obj and not isinstance(obj,object):
        raise TypeError(f'[configure_model] obj argument must be object instance {type(obj)}')

    required_keys = ['Model']
    optional_keys = [key for key in VALID_CONFIG_KEYS if not key in required_keys]
    for key in optional_keys:
        cfg[key]=cfg.get(key,None)
    verify_config_keys(cfg,required_keys,optional_keys)

    result=dict()
    
    result['model'] = instantiate(cfg['Model'])
    #
    # Optional elements (with some conditions)
    #
    if cfg['Data']:
        result['data' ] = instantiate(cfg['Data'])

    if cfg['Optimizer']:
        optim_cfg = dict(cfg['Optimizer'])
        optim_cfg['Parameters'].update(dict(params=result['model'].parameters()))
        result['optimizer'] = instantiate(optim_cfg)
        
    if cfg['Criterion']:
        result['criterion'] = instantiate(cfg['Criterion'])

    if cfg['Sampler']:
        result['sampler'] = configure_sampler(cfg['Sampler'],result['data'])

    if cfg['Collate']:
        result['collate'] = configure_collate(cfg['Collate'],result['data'])

    if cfg['DataLoader']:
        if not 'data' in result:
            print(yaml.dump(cfg,default_flow_style=False))
            raise KeyError('DataLoader requires Data configuration block')
        sampler = None if not 'sampler' in result else result['sampler']
        collate = None if not 'collate' in result else result['collate']
        result['dataloader'] = configure_dataloader(cfg['DataLoader'],result['data'],sampler,collate)

    if cfg['ModelWeights']:
        with torch.load(cfg['ModelWeights'],weights_only=True) as f:
            result['model'].load_state_dict(f["state_dict"])
            if 'optimizer' in result:
                resut['optimizer'].load_state_dict(f['optimizer'])
            result['trained_epochs'] = int(f['trained_epochs'])
    else:
        result['trained_epochs'] = 0

    if obj is None:
        return result
    else:
        for key in result.keys():
            if not hasattr(obj,str(key)):
                continue
            raise AttributeError(f'[configure_model] the given object instance already has an attribute with the name {key}')
        for key,value in result.items():
            setattr(obj,str(key),value)     
    print('[configure_model] done')


def configure_train(cfg, obj=None):
    '''
    TrainEpochs, LogPrefix, WeightPrefix, SaveFrequency, Data, DataLoader
    '''
    print('[configure_train] running')

    if obj and not isinstance(obj,object):
        raise TypeError(f'[configure_train] obj argument must be object instance {type(obj)}')
        
    required_keys = ['RunEpochs','WeightPrefix','SaveFrequency','Data','DataLoader']
    optional_keys = ['LogPrefix','Sampler','Collate','ReportFrequency']
    for key in optional_keys:
        cfg[key]=cfg.get(key,None)
    verify_config_keys(cfg,required_keys,optional_keys)

    train_epochs = int(cfg['RunEpochs'])

    train_data   = instantiate(cfg['Data'])

    train_sampler = None
    if cfg['Sampler']:
        train_sampler = configure_sampler(cfg['Sampler'],train_data)

    train_collate = None
    if cfg['Collate']:
        train_collate = configure_collate(cfg['Collate'],train_data)

    train_loader = configure_dataloader(cfg['DataLoader'],train_data,train_sampler,train_collate)

    save_frequency = float(cfg['SaveFrequency'])

    if cfg['ReportFrequency']:
        train_report_frequency = float(cfg['ReportFrequency'])
    else:
        train_report_frequency = 1e20

    if check_prefix(cfg["WeightPrefix"]):
        weight_file = cfg["WeightPrefix"] + '_epochs_%010.2f.ckpt'
    else:
        raise ValueError(f'Could not find/create the out file path {cfg["Output"]}')

    if cfg['LogPrefix']:
        if check_prefix(cfg['LogPrefix']):            
            train_log_file = cfg['LogPrefix'] + '_train.npz'
        else:
            raise ValueError(f'Could not find/create the log file path {cfg["LogPrefix"]}')
    else:
        train_log_file = None

    result = dict(train_epochs = train_epochs,
                  train_data   = train_data,
                  train_sampler= train_sampler,
                  train_collate= train_collate,
                  train_loader = train_loader,
                  save_frequency = save_frequency,
                  weight_file    = weight_file,
                  train_log_file = train_log_file,
                  train_report_frequency = train_report_frequency,
                 )
        
    if obj is None:
        return result
    else:
        for key in result.keys():
            if not hasattr(obj,str(key)):
                continue
            raise AttributeError(f'[configure_train] the given object instance already has an attribute with the name {key}')
        for key,value in result.items():
            setattr(obj,str(key),value)        
    print('[configure_train] done')


def configure_inference(cfg, obj=None):
    '''
    RunEpochs, Output, Data, DatLoader, LogPrefix
    '''
    print(['[configure_inference] running'])

    if obj and not isinstance(obj,object):
        raise TypeError(f'[configure_inference] obj argument must be object instance {type(obj)}')

    required_keys = ['Data','DataLoader']
    optional_keys = ['RunEpochs','Output','Sampler','Collate','LogPrefix']
    for key in optional_keys:
        cfg[key]=cfg.get(key,None)
    verify_config_keys(cfg,required_keys,optional_keys)

    inference_epochs = int(cfg['RunEpochs']) if cfg['RunEpochs'] else 1.0
    if inference_epochs > 1.0:
        raise ValueError(f'[configure_inference] Inference.RunEpochs cannot be larger than 1.0 (set to {inference_epochs})')
    inference_data   = instantiate(cfg['Data'])

    inference_sampler = None
    if cfg['Sampler']:
        inference_sampler = configure_sampler(cfg['Sampler'],inference_data)

    inference_collate = None
    if cfg['Collate']:
        inference_collate = configure_collate(cfg['Collate'],inference_data)

    inference_loader = configure_dataloader(cfg['DataLoader'],inference_data,inference_sampler,inference_collate)

    if check_prefix(cfg["Output"]):
        out_file = cfg["Output"] + '_epochs_%010.2f.h5'
    else:
        raise ValueError(f'Could not find/create the out file path {cfg["Output"]}')
    if cfg['LogPrefix']:
        if check_prefix(cfg['LogPrefix']):            
            inference_log_file = cfg['LogPrefix'] + '_inference_ecpochs_%010.2f.npz'
        else:
            raise ValueError(f'Could not find/create the log file path {cfg["LogPrefix"]}')
    else:
        inference_log_file = None

    result = dict(inference_epochs   = inference_epochs,
                  inference_data     = inference_data,
                  inference_sampler  = inference_sampler,
                  inference_collate  = inference_collate,
                  inference_loader   = inference_loader,
                  inference_log_file = inference_log_file,
                  out_file = out_file,
                 )
        
    if obj is None:
        return result
    else:
        for key in result.keys():
            if not hasattr(obj,str(key)):
                continue
            raise AttributeError(f'[configure_inference] the given object instance already has an attribute with the name {key}')
        for key,value in result.items():
            setattr(obj,str(key),value)
    print('[configure_inference] done')
