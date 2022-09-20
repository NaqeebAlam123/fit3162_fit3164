import os
from string import Template
import time
import pickle
import torch
import torch.nn as nn # neural network modules 
from torch.nn import init
import torch.nn.functional as F # activation function like ReLU etc
from torch.autograd import Variable 
from torch.utils.data import DataLoader # for easier dataset management (creation of batches to train and validate data on) 
import torch.utils
import torchvision
import numpy as np
# from ..utils.dataload import SER_MFCC, GET_MFCC, SMED_1D_lstm_landmark_pca
from dataload import SER_MFCC, GET_MFCC, SMED_1D_lstm_landmark_pca
from models import EmotionNet, AutoEncoder2x, AT_emotion

#import tensorboardX
from torch.utils.tensorboard import SummaryWriter
from config import config
from constants import *


def _initialize_weights(net, init_type='xavier', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                init.normal_(m.weight.data, 1.0, gain)
                init.constant_(m.bias.data, 0.0)

        print('initialize network with %s' % init_type)
        net.apply(init_func)

def train_test_batches_single_epoch(cross_entropy_loss, config, epoch, model, optimizer, start_train_time,
                            iteration_val, data_loader, writer,mode,template):
    
    all_acc = 0.0     # calculating the accuracy for tha particular set of batches for training/validation
    for batch_id_x, (data, target_label) in enumerate(data_loader):
        iteration_val += 1
        if config.cuda:
            data, target_label = Variable(data.float()).cuda(), Variable(target_label.long()).cuda()
        else:
            data, target_label = Variable(data.float()), Variable(target_label.long())

        # removes all the dimension of the input size 1 whilst returning tensor
        target_label = torch.squeeze(target_label)

        # predicting label 
        predicted_label = model(data)
        
        # difference between predicted and target output
        loss = cross_entropy_loss(predicted_label, target_label)  
        
        # calculate the accuracy b/w predicted and target label value
        acc = accuracy(target_label, predicted_label)

        if (mode == "Train"):

          # clearing the gradient of all the Variables
          optimizer.zero_grad()
          
          # for back propagation of errors (computing all gradients)
          loss.backward()
          
          # func called after the computation of all gradients
          optimizer.step()
        
        # adding data to the summary writer comprising of the loss, iter val, etc
        writer.add_scalar(mode, loss, iteration_val)

        # cummulative accuracy for that particular set of batches
        all_acc += acc.item()

        if (iteration_val % 1000 == 0):
            time_taken=time.time()-start_train_time
            # display loss, epochs, batch id, data size, etc on the terminal
            print(template.substitute(epoch=epoch+1,batch_id=batch_id_x,data_Size=len(data_loader),mode=mode,loss=loss.item(),time=time_taken))
    
    writer.add_scalar(mode+'_acc', float(all_acc) / (batch_id_x + 1), epoch + 1)

    if mode=="Train":
     return (model,iteration_val)
    else:
     return iteration_val







def accuracy(target_label, predicted_label):
    '''
    input: target_label, predicted_label
    output: acc
    '''
    # returns 1 largest element along dimension 1
    _, pred = predicted_label.topk(1, 1)
    # joins 1 dimensional elements by removing dimension 1 from the size
    pred0 = pred.squeeze().data
    # compares predicted and target labels
    acc = 100 * torch.sum(pred0 == target_label.data) / target_label.size(0)
    return acc

# emotion_pretrain
def train_EmotionNet(config):
    '''
    input: mfcc data
    output: SER_99.pkl
    '''
    if not os.path.exists(EMOTION_NET_DATASET_DIR):
        os.makedirs(EMOTION_NET_MODEL_DIR)

    # ------- 1. Load data -------#
    print('load data begin')
    train_set = SER_MFCC(EMOTION_NET_DATASET_DIR, 'train')
    val_set = SER_MFCC(EMOTION_NET_DATASET_DIR, 'val')


    train_loader = DataLoader(train_set, batch_size=config.batch_size,
                              num_workers=config.num_thread,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=config.batch_size,
                            num_workers=config.num_thread,
                            shuffle=True, drop_last=True)
    print('load data end')

    # ------- 2. Initialize network and set up the environment(gpu or cpu) -------#
    model = EmotionNet()

    # loss between the predicted and target label
    cross_entropy_loss = nn.CrossEntropyLoss()

    if config.cuda:
        device_ids = [int(i) for i in config.device_ids.split(',')]
        model = nn.DataParallel(model, device_ids=device_ids).cuda()
        cross_entropy_loss = cross_entropy_loss.cuda()
        #tripletloss = tripletloss.cuda()

    # initialize weight of model
    _initialize_weights(model)

    # optimizating algorithm - Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))

    #------- 3. Train and validate the model-------#
    writer = SummaryWriter(comment='M030')
    train_iteration_val, val_iteration_val = 0, 0

    # template for displaying the loss and other info on terminal
    t=Template('[$epoch,$batch_id / $data_Size] $mode loss :$loss timespent : $time')
    
    start_train_time = time.time()
    for epoch in range(config.max_epochs_val):
        # training of the particular set of batches
        print('start training of epoch', epoch+1)
        (model,train_iteration_val) = train_test_batches_single_epoch(cross_entropy_loss, config, epoch, model, optimizer, start_train_time,
                                             train_iteration_val, train_loader, writer,"Train",t)
       
        model.eval()

        path_val = os.path.join(EMOTION_NET_MODEL_DIR, 'SER_' + str(epoch) + '.pkl')  ## NOTE: where SER_99.pkl comes from
        with open(path_val, 'wb') as f:
            pickle.dump(model.state_dict(), f)  ## NOTE: save model parameters
           
       
        # called to tell the testing process is being initiated
        model.eval()

        print("start to validate, epoch %d" % (epoch + 1))
        with torch.no_grad():
            # testing/validation of the particular set of batches
            val_iteration_val = train_test_batches_single_epoch(cross_entropy_loss, config, epoch, model, optimizer,start_train_time,val_iteration_val, val_loader, writer, "Test", t)
            


    
    


# dtw
def train_AutoEncoder2x(config):
    os.makedirs(AUTOENCODER_2X_MODEL_DIR,exist_ok = True)

    #------- 1. Load model -------#
    torch.backends.cudnn.benchmark = True
    model = AutoEncoder2x(config)
    #CroEn_loss =  nn.CrossEntropyLoss()
    if config.cuda:
        device_ids = [int(i) for i in config.device_ids.split(',')]
        model = model.cuda()

    _initialize_weights(model)

    if config.pretrain:
        # ATnet resume
        pretrain = torch.load(ATPRETRAINED_DIR, map_location=torch.device('cpu'))
        tgt_state = model.state_dict()
        strip = 'module.'
        for name, param in pretrain.items():
            if strip is not None and name.startswith(strip):
                name = name[len(strip):]
                name = 'con_encoder.'+name
            if name not in tgt_state:
                continue
            if isinstance(param, nn.Parameter):
                param = param.data
            tgt_state[name].copy_(param)

        # SER resume
        pretrain = torch.load(SERPRETRAINED_DIR)
        # tgt_state = model.state_dict()

        strip = 'module.'
        for name, param in pretrain.items():
            if strip is not None and name.startswith(strip):
                name = name[len(strip):]
                name = 'emo_encoder.'+name
            if name not in tgt_state:
                continue
            if isinstance(param, nn.Parameter):
                param = param.data
            if name in tgt_state:
                tgt_state[name].copy_(param)

        strip = 'module.'
        for name, param in pretrain.items():
            if strip is not None and name.startswith(strip):
                name = name[len(strip):]
                name = 'classify.'+name
            if name not in tgt_state:
                continue
            if isinstance(param, nn.Parameter):
                param = param.data
            if name in tgt_state:
                tgt_state[name].copy_(param)

    #------- 2. Load training data -------#
    print('start split')
    train_set = GET_MFCC(AUTOENCODER_2X_DATASET_DIR,'train')
    test_set = GET_MFCC(AUTOENCODER_2X_DATASET_DIR,'test')
    train_loader = DataLoader(train_set,batch_size=config.batch_size,
                                        num_workers=config.num_thread,
                                        shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set,batch_size=config.batch_size,
                                        num_workers=config.num_thread,
                                        shuffle=True, drop_last=True)
    print('end split')

    #------- 3. Train the model-------#
    writer = SummaryWriter(comment='M030')

    total_steps, train_iter, val_iter = 0, 0, 0
    start_epoch = config.start_epoch

    # if config.resume :
    #     # ATnet resume
    #     resume = torch.load(AUTOENCODER_2X_RESUME_DIR)
    #     tgt_state = model.state_dict()
    #     train_iter = resume['train_step']
    #     val_iter = resume['test_step']
    #     start_epoch = resume['epoch']
    #     resume_state = resume['model']
    #     model.load_state_dict(resume_state)
    #     print('load resume model')

    a = time.time()

    for epoch in range(start_epoch, config.max_epochs):
        epoch_start_time = time.time()
        acc_1 = 0.0
        acc_2 = 0.0

        for i, data in enumerate(train_loader):
            # data = Variable(data.float().cuda())
            iter_start_time = time.time()
            outputs, losses, acces = model.train_func(data)
            losses_values = {k:v.item() for k, v in losses.items()}
            acces_values = {k:v.item() for k, v in acces.items()}
            
            for k, v in losses_values.items():
                writer.add_scalar(k, v, train_iter)

            acc_1 += acces_values['acc_1']
            acc_2 += acces_values['acc_2']

            loss = sum(losses.values())
            writer.add_scalar('train_loss', loss, train_iter)
            
            # opt_m.zero_grad()
            # loss.backward()
            # opt_m.step()

            # if (train_iter % 10 == 0):
            #     print('[%d,%5d / %d] con_feature loss :%.10f cla_1 loss :%.10f train loss :%.10f time spent: %f s' %(epoch + 1, i+1, len(train_loader), losses['con_feature'].item(),losses['cla_1'].item(),loss.item(),time.time()-a))

            if (train_iter % 100 ==0): #500
                with open(AUTOENCODER_2X_LOG_DIR + 'train.txt','a') as file_handle:
                    file_handle.write('[%d,%5d / %d] con_feature loss :%.10f cla_1 loss :%.10f train loss :%.10f time spent: %f s' %(epoch + 1, i+1, len(train_loader), losses['con_feature'].item(),losses['cla_1'].item(),loss.item(),time.time()-a))
                    file_handle.write('\n')

            if (train_iter % 500 == 0): #2000
                save_path = os.path.join(AUTOENCODER_2X_IMAGE_DIR+'train/'+str(epoch+1),str(train_iter))
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                model.save_fig(data,outputs,save_path)

            train_iter += 1
            # string = os.path.join(AUTOENCODER_2X_MODEL_DIR,'SER_'+str(epoch) + '.pkl')
            # torch.save(model.state_dict(), string)
            writer.add_scalar('acc_1',float(acc_1)/(i+1),epoch+1)
            writer.add_scalar('acc_2',float(acc_2)/(i+1),epoch+1)
        torch.save({
                    'train_step': train_iter,
                    'test_step': val_iter,
                    'epoch': epoch,
                    'model': model.state_dict(),
                    }, os.path.join(AUTOENCODER_2X_MODEL_DIR, str(epoch) + "_"  + 'pretrain.pth'))

        # model.eval()
        print("start to validate, epoch %d" %(epoch+1))

        acc_1_v = 0.0
        acc_2_v = 0.0

        with torch.no_grad():
            for i, data in enumerate(test_loader):
                outputs, losses, acces = model.val_func(data)

                losses_values = {k:v.item() for k, v in losses.items()}
                acces_values = {k:v.item() for k, v in acces.items()}

                for k, v in losses_values.items():
                    writer.add_scalar(k+'_v', v, val_iter)

                acc_1_v += acces_values['acc_1']
                acc_2_v += acces_values['acc_2']

                loss = sum(losses.values())
                writer.add_scalar('test_loss', loss, val_iter)

                if (val_iter % 10 == 0):
                    print('[%d,%5d / %d] con_feature loss :%.10f cla_1 loss :%.10f val loss :%.10f time spent: %f s' %(epoch + 1, i+1, len(test_loader), losses['con_feature'].item(),losses['cla_1'].item(),loss.item(),time.time()-a))

                if (val_iter % 500 == 0): #2000
                    save_path = os.path.join(AUTOENCODER_2X_IMAGE_DIR+'val/'+str(epoch+1),str(val_iter))
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    model.save_fig(data,outputs,save_path)

                    with open(AUTOENCODER_2X_LOG_DIR + 'val.txt','a') as file_handle:
                        file_handle.write('[%d,%5d / %d] con_feature loss :%.10f cla_1 loss :%.10f val loss :%.10f time spent: %f s' %(epoch + 1, i+1, len(test_loader), losses['con_feature'].item(),losses['cla_1'].item(),loss.item(),time.time()-a))
                        file_handle.write('\n')

                val_iter += 1
                writer.add_scalar('acc_1_v',float(acc_1_v)/ (i+1), epoch+1)
                writer.add_scalar('acc_2_v',float(acc_2_v)/ (i+1), epoch+1)
    return


# landmark
def train_AT_Emotion(config):

    """ 1. load the data """
    print('start split')
    train_set=SMED_1D_lstm_landmark_pca(LM_ENCODER_DATASET_LANDMARK_DIR,'train')
    test_set=SMED_1D_lstm_landmark_pca(LM_ENCODER_DATASET_LANDMARK_DIR,'test')
    #train_set,test_set = train_test_split(dataset,test_size=0.2,random_state=1)
    train_loader = DataLoader(train_set,
                            batch_size=config.batch_size,
                            num_workers=config.num_thread,
                            shuffle=True, drop_last=True)

    test_loader = DataLoader(test_set,
                            batch_size=config.batch_size,
                            num_workers=config.num_thread,
                            shuffle=True, drop_last=True)

    print('end split')

    #pca = torch.FloatTensor(np.load('/home/jixinya/first-2D frontalize/U_2Df.npy')[:,:20] )#20
    #mean = torch.FloatTensor(np.load('/home/jixinya/first-2D frontalize/mean_2Df.npy'))
    num_steps_per_epoch = len(train_loader)
    num_val = len(test_loader)

    writer=SummaryWriter(comment='M019')

    """ 2. load the model """
    generator = AT_emotion(config)
    if config.cuda:
        device_ids = [int(i) for i in config.device_ids.split(',')]
        generator = generator.cuda()

    #initialize_weights(generator)

    if config.pretrain :
        # ATnet resume
        pretrain = torch.load(LM_ENCODER_PRETRAINED_DIR)
        pretrain = pretrain['model']
        tgt_state = generator.state_dict()
        strip = 'con_encoder.'
        for name, param in pretrain.items():
            if isinstance(param, nn.Parameter):
                param = param.data
            if strip is not None and name.startswith(strip):
                tgt_state[name].copy_(param)
            if name not in tgt_state:
                continue
        
        # SER resume
        strip = 'emo_encoder.'
        for name, param in pretrain.items():
            if isinstance(param, nn.Parameter):
                param = param.data
            if name not in tgt_state:
                continue
            if strip is not None and name.startswith(strip):
                tgt_state[name].copy_(param)

    if config.pretrain_sep :
        # ATnet resume
        pretrain = torch.load(ATPRETRAINED_DIR)
        tgt_state = generator.state_dict()
        strip = 'module.'
        for name, param in pretrain.items():
            if strip is not None and name.startswith(strip):
                name = name[len(strip):]
                name = 'con_encoder.'+name
            if name not in tgt_state:
                continue
            if isinstance(param, nn.Parameter):
                param = param.data
            tgt_state[name].copy_(param) # load pretrain param

        #SER resume
        pretrain = torch.load(config.serpretrained_dir)
    #  tgt_state = model.state_dict()

        strip = 'module.'
        for name, param in pretrain.items():
            if strip is not None and name.startswith(strip):
                name = name[len(strip):]
                name = 'emo_encoder.'+name
            if name not in tgt_state:
                continue
            if isinstance(param, nn.Parameter):
                param = param.data
            if name in tgt_state:
                tgt_state[name].copy_(param)


    """ 3. loss function """
    #l1_loss_fn   = nn.L1Loss()
    #mse_loss_fn = nn.MSELoss()
    ## NOTE: uncomment 2 lines below

    pca =np.load(f'{LANDMARK_BASICS}U_68.npy')[:, :16]
    mean =np.load(f'{LANDMARK_BASICS}mean_68.npy')

    """ 4. train & validation"""
    start_epoch = 0
    t0 = time.time()
    train_itr=0
    test_itr=0
    for epoch in range(start_epoch, config.max_epochs):
    #           t0 = time.time()
    #           train_loss_meter.reset()
        all_loss = 0.0
        for step, (example_landmark, example_audio, lmark, mfccs) in enumerate(train_loader):
            t1 = time.time()

            if config.cuda:
                lmark    = Variable(lmark.float()).cuda()
                mfccs = Variable(mfccs.float()).cuda()
                example_landmark = Variable(example_landmark.float()).cuda()

            fake_lmark, loss_pca, loss_lm= generator.train_func(example_landmark, lmark, mfccs)

            loss_pca = 1000*loss_pca
            loss_lm = 1000*loss_lm
            loss = loss_pca + loss_lm
    #     fake = fake_lmark.data.cpu().numpy()
    #     for i in range()
    #     result=np.dot(lmark,pca.T)+mean

            all_loss += loss.item()
            t2 = time.time()
            train_itr += 1
        #  writer.add_scalars('Train',{'loss_pca':loss_pca,'loss_lm':loss_lm,'loss':loss},train_itr)
            writer.add_scalar('Train',loss,train_itr)
            writer.add_scalar('Train_lm',loss_lm,train_itr)
            writer.add_scalar('Train_pca',loss_pca,train_itr)

            print("[{}/{}][{}/{}]    loss: {:.8f},data time: {:.4f},  model time: {} second, loss time: {} second"
                            .format(epoch+1, config.max_epochs,
                                    step+1, num_steps_per_epoch,  loss, t1-t0,  t2 - t1, time.time()-t1))
            print("all time: {} second"
                            .format(time.time() - t1))

        torch.save(generator.state_dict(),
                                "{}/atnet_emotion_{}.pth"
                                .format(LM_ENCODER_MODEL_DIR,epoch))

        t0 = time.time()
        print("final average train loss = ", float(all_loss)/(step+1))

        print("start to validate, epoch %d" %(epoch+1))
        # ------- Validation ------ #
        all_val_loss = 0.0
        for step, (example_landmark, example_audio, lmark, mfccs) in enumerate(test_loader):
            with torch.no_grad():
                if config.cuda:
                    lmark    = Variable(lmark.float().cuda())
                    mfccs = Variable(mfccs.float().cuda())
                    example_landmark = Variable(example_landmark.float().cuda())

                fake_lmark,loss_pca, loss_lm= generator.val_func( example_landmark, lmark, mfccs)

                loss_pca = 1000*loss_pca
                loss_lm = 1000*loss_lm
                test_loss = loss_pca + loss_lm
            #  test_loss = 1000*test_loss

                all_val_loss += test_loss.item()
                test_itr+=1
                writer.add_scalar('Test',test_loss,test_itr)
                writer.add_scalar('Test_lm',loss_lm,test_itr)
                writer.add_scalar('Test_pca',loss_pca,test_itr)

        #     writer.add_scalars('Val',{'loss_pca':loss_pca,'loss_lm':loss_lm,'loss':test_loss},test_itr)
                print("[{}/{}][{}/{}]   loss: {:.8f},all time: {} second"
                            .format(epoch+1, config.max_epochs,
                                    step+1, num_val, test_loss,   time.time()-t1))

        print("final average test loss = ", float(all_val_loss)/(step+1))
    return


if __name__ == '__main__':
    if config.emotion_pretrain:
        train_EmotionNet(config)
    elif config.dtw:
        train_AutoEncoder2x(config)
    elif config.landmark:
        train_AT_Emotion(config)
