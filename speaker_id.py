# speaker_id.py
# Mirco Ravanelli 
# Mila - University of Montreal 

# July 2018

# Description: 
# This code performs a speaker_id experiments with SincNet.
 
# How to run it:
# python speaker_id.py --cfg=cfg/SincNet_TIMIT.cfg

import os
#import scipy.io.wavfile
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import sys
import numpy as np
from dnn_models import MLP,flip, SpeakerIDNet
from dnn_models import SincNet as CNN 
from data_io import ReadList,read_conf,str_to_bool
from argparse import Namespace

def create_batches_rnd(batch_size,data_folder,wav_lst,N_snt,wlen,lab_dict,fact_amp):
    
 # Initialization of the minibatch (batch_size,[0=>x_t,1=>x_t+N,1=>random_samp])
 sig_batch=np.zeros([batch_size,wlen])
 lab_batch=np.zeros(batch_size)
  
 snt_id_arr=np.random.randint(N_snt, size=batch_size)
 
 rand_amp_arr = np.random.uniform(1.0-fact_amp,1+fact_amp,batch_size)

 for i in range(batch_size):
     
  # select a random sentence from the list 
  #[fs,signal]=scipy.io.wavfile.read(data_folder+wav_lst[snt_id_arr[i]])
  #signal=signal.astype(float)/32768

  [signal, fs] = sf.read(data_folder+wav_lst[snt_id_arr[i]])

  # accesing to a random chunk
  snt_len=signal.shape[0]
  snt_beg=np.random.randint(snt_len-wlen-1) #randint(0, snt_len-2*wlen-1)
  snt_end=snt_beg+wlen

  channels = len(signal.shape)
  if channels == 2:
    print('WARNING: stereo to mono: '+data_folder+wav_lst[snt_id_arr[i]])
    signal = signal[:,0]
  
  sig_batch[i,:]=signal[snt_beg:snt_end]*rand_amp_arr[i]
  lab_batch[i]=lab_dict[wav_lst[snt_id_arr[i]]]
  
 inp=Variable(torch.from_numpy(sig_batch).float().cuda().contiguous())
 lab=Variable(torch.from_numpy(lab_batch).float().cuda().contiguous())
  
 return inp,lab  

@torch.no_grad()
def evaluate(model:SpeakerIDNet, lab_dict, wlen, wshift, Batch_dev, cost):
    model.eval()
    test_flag=1 
    loss_sum=0
    err_sum=0
    err_sum_snt=0
    
    with torch.no_grad():  
        for i in range(args.snt_te):
        
            [signal, fs] = sf.read(args.data_folder+args.wav_lst_te[i])
        
            signal=torch.from_numpy(signal).float().cuda().contiguous()
            lab_batch=lab_dict[args.wav_lst_te[i]]
            
            # split signals into chunks
            beg_samp=0
            end_samp=wlen
            
            N_fr=int((signal.shape[0]-wlen)/(wshift))
            
            sig_arr=torch.zeros([Batch_dev,wlen]).float().cuda().contiguous()
            lab= ((torch.zeros(N_fr+1)+lab_batch).cuda().contiguous().long())
            pout=(torch.zeros(N_fr+1,args.class_lay[-1]).float().cuda().contiguous())
            count_fr=0
            count_fr_tot=0
            while end_samp<signal.shape[0]:
                sig_arr[count_fr,:]=signal[beg_samp:end_samp]
                beg_samp=beg_samp+wshift
                end_samp=beg_samp+wlen
                count_fr=count_fr+1
                count_fr_tot=count_fr_tot+1
                if count_fr==Batch_dev:
                    inp=(sig_arr)
                    pout[count_fr_tot-Batch_dev:count_fr_tot,:]=model(inp)
                    count_fr=0
                    sig_arr=torch.zeros([Batch_dev,wlen]).float().cuda().contiguous()
          
            if count_fr>0:
                inp=(sig_arr[0:count_fr])
                pout[count_fr_tot-count_fr:count_fr_tot,:]=model(inp)
        
            pred=torch.max(pout,dim=1)[1]
            loss = cost(pout, lab.long())
            err = torch.mean((pred!=lab.long()).float())
            
            [val,best_class]=torch.max(torch.sum(pout,dim=0),0)
            err_sum_snt=err_sum_snt+(best_class!=lab[0]).float()
            
            loss_sum=loss_sum+loss.detach()
            err_sum=err_sum+err.detach()
      
        err_tot_dev_snt=err_sum_snt/args.snt_te
        loss_tot_dev=loss_sum/args.snt_te
        err_tot_dev=err_sum/args.snt_te

    print("loss_te=%f err_te=%f err_te_snt=%f" % (loss_tot_dev,err_tot_dev,err_tot_dev_snt))
    

def get_cfg():
    # Reading cfg file
    options=read_conf()
    #[data]
    args = Namespace()
    args.tr_lst=options.tr_lst
    args.te_lst=options.te_lst
    args.pt_file=options.pt_file if options.pt_file_reset=='' else options.pt_file_reset
    args.class_dict_file=options.lab_dict
    args.data_folder=(options.data_folder if options.data_folder_reset=='' else options.data_folder_reset)+'/'
    args.output_folder=options.output_folder
    #[windowing]
    args.fs=int(options.fs)
    args.cw_len=int(options.cw_len)
    args.cw_shift=int(options.cw_shift)
    #[cnn]
    args.cnn_N_filt=list(map(int, options.cnn_N_filt.split(',')))
    args.cnn_len_filt=list(map(int, options.cnn_len_filt.split(',')))
    args.cnn_max_pool_len=list(map(int, options.cnn_max_pool_len.split(',')))
    args.cnn_use_laynorm_inp=str_to_bool(options.cnn_use_laynorm_inp)
    args.cnn_use_batchnorm_inp=str_to_bool(options.cnn_use_batchnorm_inp)
    args.cnn_use_laynorm=list(map(str_to_bool, options.cnn_use_laynorm.split(',')))
    args.cnn_use_batchnorm=list(map(str_to_bool, options.cnn_use_batchnorm.split(',')))
    args.cnn_act=list(map(str, options.cnn_act.split(',')))
    args.cnn_drop=list(map(float, options.cnn_drop.split(',')))
    #[dnn]
    args.fc_lay=list(map(int, options.fc_lay.split(',')))
    args.fc_drop=list(map(float, options.fc_drop.split(',')))
    args.fc_use_laynorm_inp=str_to_bool(options.fc_use_laynorm_inp)
    args.fc_use_batchnorm_inp=str_to_bool(options.fc_use_batchnorm_inp)
    args.fc_use_batchnorm=list(map(str_to_bool, options.fc_use_batchnorm.split(',')))
    args.fc_use_laynorm=list(map(str_to_bool, options.fc_use_laynorm.split(',')))
    args.fc_act=list(map(str, options.fc_act.split(',')))
    #[class]
    args.class_lay=list(map(int, options.class_lay.split(',')))
    args.class_drop=list(map(float, options.class_drop.split(',')))
    args.class_use_laynorm_inp=str_to_bool(options.class_use_laynorm_inp)
    args.class_use_batchnorm_inp=str_to_bool(options.class_use_batchnorm_inp)
    args.class_use_batchnorm=list(map(str_to_bool, options.class_use_batchnorm.split(',')))
    args.class_use_laynorm=list(map(str_to_bool, options.class_use_laynorm.split(',')))
    args.class_act=list(map(str, options.class_act.split(',')))
    #[optimization]
    args.lr=float(options.lr)
    args.batch_size=int(options.batch_size)
    args.N_epochs=int(options.N_epochs)
    args.N_batches=int(options.N_batches)
    args.N_eval_epoch=int(options.N_eval_epoch)
    args.seed=int(options.seed)
    # training list
    args.wav_lst_tr=ReadList(args.tr_lst)
    args.snt_tr=len(args.wav_lst_tr)
    # test list
    args.wav_lst_te=ReadList(args.te_lst)
    args.snt_te=len(args.wav_lst_te)
    args.eval = options.eval
    # Folder creation
    try:
        os.stat(args.output_folder)
    except:
        os.mkdir(args.output_folder) 
    return args
    

def main(args):
    # setting seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # loss function
    cost = nn.NLLLoss()
    
    # Converting context and shift in samples
    wlen=int(args.fs*args.cw_len/1000.00)
    wshift=int(args.fs*args.cw_shift/1000.00)
    
    # Batch_dev
    Batch_dev=128

    # Feature extractor CNN
    CNN_arch = {'input_dim': wlen,
              'fs': args.fs,
              'cnn_N_filt': args.cnn_N_filt,
              'cnn_len_filt': args.cnn_len_filt,
              'cnn_max_pool_len':args.cnn_max_pool_len,
              'cnn_use_laynorm_inp': args.cnn_use_laynorm_inp,
              'cnn_use_batchnorm_inp': args.cnn_use_batchnorm_inp,
              'cnn_use_laynorm':args.cnn_use_laynorm,
              'cnn_use_batchnorm':args.cnn_use_batchnorm,
              'cnn_act': args.cnn_act,
              'cnn_drop':args.cnn_drop,          
              }

    DNN1_arch = {'fc_lay': args.fc_lay,
              'fc_drop': args.fc_drop, 
              'fc_use_batchnorm': args.fc_use_batchnorm,
              'fc_use_laynorm': args.fc_use_laynorm,
              'fc_use_laynorm_inp': args.fc_use_laynorm_inp,
              'fc_use_batchnorm_inp':args.fc_use_batchnorm_inp,
              'fc_act': args.fc_act,
              }

    DNN2_arch = {'input_dim':args.fc_lay[-1] ,
              'fc_lay': args.class_lay,
              'fc_drop': args.class_drop, 
              'fc_use_batchnorm': args.class_use_batchnorm,
              'fc_use_laynorm': args.class_use_laynorm,
              'fc_use_laynorm_inp': args.class_use_laynorm_inp,
              'fc_use_batchnorm_inp':args.class_use_batchnorm_inp,
              'fc_act': args.class_act,
              }

    model = SpeakerIDNet(CNN_arch, DNN1_arch, DNN2_arch)
    # Loading label dictionary
    lab_dict=np.load(args.class_dict_file).item()

    if args.pt_file!='none':
        print("load model from:", args.pt_file)
        checkpoint_load = torch.load(args.pt_file)
        if os.path.splitext(args.pt_file)[1] == '.pkl':
            model.load_raw_state_dict(checkpoint_load)
        else:
            model.load_state_dict(checkpoint_load)
    model = model.cuda()
    if args.eval:
        print('only eval the model')
        evaluate(model, lab_dict, wlen, wshift, Batch_dev, cost)
        return
    else:
        print("train the model")
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr,alpha=0.95, eps=1e-8) 

    for epoch in range(args.N_epochs):
        test_flag=0
        model.train()
       
        loss_sum=0
        err_sum=0
      
        for i in range(args.N_batches):
     
            [inp,lab]=create_batches_rnd(args.batch_size,args.data_folder,args.wav_lst_tr,args.snt_tr,wlen,lab_dict,0.2)
            pout=model(inp)
            
            pred=torch.max(pout,dim=1)[1]
            loss = cost(pout, lab.long())
            err = torch.mean((pred!=lab.long()).float())
            
            model.zero_grad() 
            
            loss.backward()
            optimizer.step()
            
            loss_sum=loss_sum+loss.detach()
            err_sum=err_sum+err.detach()
      
        loss_tot=loss_sum/args.N_batches
        err_tot=err_sum/args.N_batches
      
        # Full Validation  new  
        if epoch%args.N_eval_epoch==0:
           
            model.eval()
            test_flag=1 
            loss_sum=0
            err_sum=0
            err_sum_snt=0
            
            with torch.no_grad():  
                for i in range(args.snt_te):
                 
                    #[fs,signal]=scipy.io.wavfile.read(data_folder+wav_lst_te[i])
                    #signal=signal.astype(float)/32768
                
                    [signal, fs] = sf.read(args.data_folder+args.wav_lst_te[i])
                
                    signal=torch.from_numpy(signal).float().cuda().contiguous()
                    lab_batch=lab_dict[args.wav_lst_te[i]]
                    
                    # split signals into chunks
                    beg_samp=0
                    end_samp=wlen
                    
                    N_fr=int((signal.shape[0]-wlen)/(wshift))
                    
                    sig_arr=torch.zeros([Batch_dev,wlen]).float().cuda().contiguous()
                    lab= Variable((torch.zeros(N_fr+1)+lab_batch).cuda().contiguous().long())
                    pout=Variable(torch.zeros(N_fr+1,args.class_lay[-1]).float().cuda().contiguous())
                    count_fr=0
                    count_fr_tot=0
                    while end_samp<signal.shape[0]:
                        sig_arr[count_fr,:]=signal[beg_samp:end_samp]
                        beg_samp=beg_samp+wshift
                        end_samp=beg_samp+wlen
                        count_fr=count_fr+1
                        count_fr_tot=count_fr_tot+1
                        if count_fr==Batch_dev:
                            inp=Variable(sig_arr)
                            pout[count_fr_tot-Batch_dev:count_fr_tot,:]=model(inp)
                            count_fr=0
                            sig_arr=torch.zeros([Batch_dev,wlen]).float().cuda().contiguous()
                  
                    if count_fr>0:
                        inp=Variable(sig_arr[0:count_fr])
                        pout[count_fr_tot-count_fr:count_fr_tot,:]=model(inp)
                
                    
                    pred=torch.max(pout,dim=1)[1]
                    loss = cost(pout, lab.long())
                    err = torch.mean((pred!=lab.long()).float())
                    
                    [val,best_class]=torch.max(torch.sum(pout,dim=0),0)
                    err_sum_snt=err_sum_snt+(best_class!=lab[0]).float()
                    
                    
                    loss_sum=loss_sum+loss.detach()
                    err_sum=err_sum+err.detach()
              
                err_tot_dev_snt=err_sum_snt/args.snt_te
                loss_tot_dev=loss_sum/args.snt_te
                err_tot_dev=err_sum/args.snt_te
     
       
            print("epoch %i, loss_tr=%f err_tr=%f loss_te=%f err_te=%f err_te_snt=%f" % (epoch, loss_tot,err_tot,loss_tot_dev,err_tot_dev,err_tot_dev_snt))
           
            with open(args.output_folder+"/res.res", "a") as res_file:
                res_file.write("epoch %i, loss_tr=%f err_tr=%f loss_te=%f err_te=%f err_te_snt=%f\n" % (epoch, loss_tot,err_tot,loss_tot_dev,err_tot_dev,err_tot_dev_snt))   
         
            checkpoint=model.state_dict()
            torch.save(checkpoint,args.output_folder+'/model_raw_{}.pickle'.format(epoch))
       
        else:
            print("epoch %i, loss_tr=%f err_tr=%f" % (epoch, loss_tot,err_tot))
     




if __name__=='__main__':
    args = get_cfg()
    main(args)
    