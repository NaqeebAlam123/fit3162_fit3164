import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from ops import conv2d
import torchvision.models as models
import numpy as np
import seaborn as sns
from constants import LANDMARK_BASICS

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Lm_encoder(nn.Module):
    def __init__(self):
        super(Lm_encoder, self).__init__()
        self.lmark_encoder = nn.Sequential(
            nn.Linear(16,256),
            nn.ReLU(True),
            nn.Linear(256,512),
            nn.ReLU(True),
        ).cuda()

    def forward(self, example_landmark):
        # print('lm_encoder', example_landmark.get_device())
        example_landmark_f = self.lmark_encoder(example_landmark)
        return example_landmark_f


class Ct_encoder(nn.Module):
    def __init__(self):
        super(Ct_encoder, self).__init__()
        self.audio_encoder = nn.Sequential(
            conv2d(1,64,3,1,1),
            conv2d(64,128,3,1,1),
            nn.MaxPool2d(3, stride=(1,2)),
            conv2d(128,256,3,1,1),
            conv2d(256,256,3,1,1),
            conv2d(256,512,3,1,1),
            nn.MaxPool2d(3, stride=(2,2))
        ).cuda()
        self.audio_encoder_fc = nn.Sequential(
            nn.Linear(1024 *12,2048),
            nn.ReLU(True),
            nn.Linear(2048,256),
            nn.ReLU(True),
        ).cuda()

    def forward(self, audio):
        # print('ct_encoder', audio.get_device())
        feature = self.audio_encoder(audio.cuda())
        feature = feature.view(feature.size(0),-1)
        x = self.audio_encoder_fc(feature)
        return x


class EmotionNet(nn.Module):
    def __init__(self):
        super(EmotionNet, self).__init__()
        self.emotion_encoder = nn.Sequential(
            conv2d(1,64,3,1,1),
            nn.MaxPool2d((1,3), stride=(1,2)), #[1, 64, 12, 12]
            conv2d(64,128,3,1,1),
            conv2d(128,256,3,1,1),
            nn.MaxPool2d((12,1), stride=(12,1)), #[1, 256, 1, 12]
            conv2d(256,512,3,1,1),
            nn.MaxPool2d((1,2), stride=(1,2)) #[1, 512, 1, 6]
        ).cuda()
        self.emotion_encoder_fc = nn.Sequential(
            nn.Linear(512*6,2048),
            nn.ReLU(True),
            nn.Linear(2048,128),
            nn.ReLU(True),
        ).cuda()
        self.last_fc = nn.Linear(128,8).cuda()

    def forward(self, mfcc):
        mfcc=torch.transpose(mfcc,2,3)
        feature = self.emotion_encoder(mfcc.cuda())
        feature = feature.view(feature.size(0),-1)
        x = self.emotion_encoder_fc(feature)
        if self.training:
            # print('Emotion net is training', self.training)
            x = self.last_fc(x)
        return x


class Ct_Decoder(nn.Module):
    def __init__(self):
        super(Ct_Decoder, self).__init__()
        self.decon = nn.Sequential(
                # nn.ConvTranspose2d(131328, 384, kernel_size=6),
                nn.ConvTranspose2d(384, 256, kernel_size=6, stride=2, padding=1),#4,4
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.ConvTranspose2d(256, 128, kernel_size=(4,2), stride=2, padding=1),#8,6
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), #16,12
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.ConvTranspose2d(64, 32, kernel_size=(4,3), stride=(2,1), padding=(3,1)),#28,12
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1),#28,12
                nn.Tanh(),
                )

    def forward(self, content, emotion):
        features = torch.cat([content,  emotion], 1) #connect tensors inputs and dimension
        features = torch.unsqueeze(features,2)
        features = torch.unsqueeze(features,3)
        x = 90*self.decon(features) #[1, 1,28, 12]
        return x


class Classify(nn.Module):
    def __init__(self):
        super(Classify, self).__init__()
        self.last_fc = nn.Linear(128,8)

    def forward(self, feature):
       # mfcc= torch.unsqueeze(mfcc, 1)
        x = self.last_fc(feature)
        return x


class AutoEncoder2x(nn.Module):
    def __init__(self,config):
        super(AutoEncoder2x, self).__init__()

        self.con_encoder = Ct_encoder()
        self.emo_encoder = EmotionNet()
        self.decoder = Ct_Decoder()
        self.classify = Classify()

        self.CroEn_loss =  nn.CrossEntropyLoss()
        self.l1loss = nn.L1Loss()
        self.tripletloss = nn.TripletMarginLoss(margin=config.triplet_margin)
        self.triplet_weight = config.triplet_weight

        self.use_triplet = config.use_triplet

        self.labels_name = ['label1','label2']
        self.inputs_name = ['input11',  'input12', 'input21', 'input32']
        self.targets_name = ['target11', 'target22',"target12","target21"]

        self.optimizer = torch.optim.Adam(list(self.con_encoder.parameters())
                                            +list(self.emo_encoder.parameters())
                                            +list(self.decoder.parameters())
                                            +list(self.classify.parameters()), config.lr,betas=(config.beta1, config.beta2))

    def cross(self, x1, x2, x3, x4): # called in process()
        self.emo_encoder.eval()
        c1 = self.con_encoder(x1.cuda())
        c2 = self.con_encoder(x4.cuda())

        e1 = self.emo_encoder(x2.cuda())
        e2 = self.emo_encoder(x3.cuda())

        out1 = self.decoder(c1.cuda(),e1.cuda())
        out2 = self.decoder(c1.cuda(),e2.cuda())
        out3 = self.decoder(c2.cuda(),e1.cuda())
        out4 = self.decoder(c2.cuda(),e2.cuda())

        return out1, out2, out3, out4

    def transfer(self, x1, x2): # NOT called
        m1 = self.mot_encoder(x1)
        b2 = self.static_encoder(x2[:, :-2, :]).repeat(1, 1, m1.shape[-1])

        out12 = self.decoder(torch.cat([m1, b2], dim=1))

        return out12

    def cross_with_triplet(self, x1, x2, x12, x21): # called
        c1 = self.con_encoder(x1)
        e1 = self.emo_encoder(x1)
        c2 = self.con_encoder(x2)
        e2 = self.emo_encoder(x2)

        out1 = self.decoder(c1,e1)
        out2 = self.decoder(c2,e2)
        out12 = self.decoder(c1,e2)
        out21 = self.decoder(c2,e1)

        c12 = self.con_encoder(x12)
        e12 = self.emo_encoder(x12)
        c21 = self.con_encoder(x21)
        e21 = self.emo_encoder(x21)

        outputs = [out1, out2, out12, out21]
        contentvecs = [c1.reshape(c1.shape[0], -1),
                      c2.reshape(c2.shape[0], -1),
                      c12.reshape(c12.shape[0], -1),
                      c21.reshape(c21.shape[0], -1)]
        emotionvecs = [e1.reshape(e1.shape[0], -1),
                      e2.reshape(e2.shape[0], -1),
                      e21.reshape(e21.shape[0], -1),
                      e12.reshape(e12.shape[0], -1)]

        return outputs, contentvecs, emotionvecs

    def compute_acc(self,input_label, out):
        _, pred = out.topk(1, 1)
        pred0 = pred.squeeze().data
        acc = 100 * torch.sum(pred0 == input_label.data) / input_label.size(0)
        return acc

    def process(self, data): # called in train_func
        labels = [data[name] for name in self.labels_name]
        inputs = [data[name] for name in self.inputs_name]
        targets = [data[name] for name in self.targets_name]

        losses = {}
        acces = {}

        if self.use_triplet:
            outputs, contentvecs, emotionvecs = self.cross_with_triplet(*inputs)
            losses['c_tpl1'] = self.triplet_weight * self.tripletloss(contentvecs[2], contentvecs[0], contentvecs[1])
            losses['c_tpl2'] = self.triplet_weight * self.tripletloss(contentvecs[3], contentvecs[1], contentvecs[0])
            losses['e_tpl1'] = self.triplet_weight * self.tripletloss(emotionvecs[2], emotionvecs[0], emotionvecs[1])
            losses['e_tpl2'] = self.triplet_weight * self.tripletloss(emotionvecs[3], emotionvecs[1], emotionvecs[0])
        else:
            outputs = self.cross(inputs[0], inputs[1], inputs[2], inputs[3])

        for i, target in enumerate(targets):
            try:
                losses['rec' + self.targets_name[i][6:]] = self.l1loss(outputs[i], target.cuda())
            except Exception as e:
                import ipdb
                ipdb.set_trace()

        c1 = self.con_encoder(inputs[0])
        c2 = self.con_encoder(inputs[3])

        e1 = self.emo_encoder(inputs[1])
        e2 = self.emo_encoder(inputs[2])
        losses['con_feature'] = self.l1loss(c1, c2)

        label1 = labels[0]
        label1=torch.squeeze(label1).cuda()
        label2 = labels[1]
        label2=torch.squeeze(label2).cuda()

        fake1 = self.classify(e1).cuda()
        fake2 = self.classify(e2).cuda()

        losses['cla_1'] = self.CroEn_loss(fake1,label1)
        losses['cla_2'] = self.CroEn_loss(fake2,label2)

        acces['acc_1'] = self.compute_acc(label1,fake1)
        acces['acc_2'] = self.compute_acc(label2,fake2)

        outputs_dict = {
            "output1": outputs[0],
            "output2": outputs[1],
            "output3": outputs[2],
            "output4": outputs[3],
        }
        return outputs_dict, losses , acces

    def forward(self, x):
        c = self.con_encoder(x)
        e = self.emo_encoder(x[:, :-2, :])

        d = torch.cat([c, e], dim=1)
        d = self.decoder(d)
        return d

    def update_network(self, loss_dcit): # called in train_func
        loss = sum(loss_dcit.values())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_learning_rate(self):
        self.scheduler.step(self.clock.epoch)

    def train_func(self, data): # called
        self.classify.train()
        self.decoder.train()
        self.con_encoder.train()
        self.emo_encoder.train()

        outputs, losses, acces = self.process(data)

        self.update_network(losses)

        return outputs, losses, acces

    def val_func(self, data): # called
        self.classify.eval()
        self.decoder.eval()
        self.con_encoder.eval()
        self.emo_encoder.eval()

        with torch.no_grad():
            outputs, losses, acces = self.process(data)

        return outputs, losses, acces

    def save_fig(self,data,outputs,save_path): # called

    #    output1 = outputs['output1']
    #    output2 = outputs['output2']
    #    output12 = outputs['output12']
    #    output21 = outputs['output21']

    #    target1 = data['target11']
    #    target2 = data['target22']
    #    target12 = data['target12']
    #    target21 = data['target21']

        a=['output1','output2','output3','output4']
        b=['target11','target22','target12','target21']

        for j in range(len(a)):
            output = outputs[a[j]]
            target = data[b[j]]

            for i in range(output.size(0)):
                g = target[i,:,:,:].squeeze()
                g = g.cpu().numpy()
           # plt.figure()
                # ax = sns.heatmap(g, vmin=-100, vmax=100,cmap='rainbow')      #frames

                filepath = os.path.join(save_path,a[j])
                if not os.path.exists(filepath):
                    os.makedirs(filepath)
                # plt.savefig(os.path.join(filepath,'real_'+str(i)+'.png'))
                # plt.close()
      #      plt.show()
                o = output[i,:,:,:].squeeze()
                o = o.cpu().detach().numpy()
      #      plt.figure()
                # ax = sns.heatmap(o, vmin=-100, vmax=100,cmap='rainbow')      #frames

                # plt.savefig(os.path.join(filepath,'fake_'+str(i)+'.png'))
                # plt.close()
      #      plt.show()


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(128*7,256,3,batch_first = True).cuda()
        self.lstm_fc = nn.Sequential(
            nn.Linear(256,16),#20
        ).cuda()

    def forward(self, lstm_input):
        print(lstm_input.shape)
        hidden = ( torch.autograd.Variable(torch.zeros(3, lstm_input.size(0), 256).cuda()),# torch.Size([3, 16, 256])
                      torch.autograd.Variable(torch.zeros(3, lstm_input.size(0), 256).cuda()))# torch.Size([3, 16, 256])
        print(hidden[0].shape, hidden[1].shape)

        lstm_out, _ = self.lstm(lstm_input, hidden) #torch.Size([16, 16, 256])
        print(lstm_out.shape)
        fc_out   = []
        for step_t in range(lstm_out.size(1)):
            fc_in = lstm_out[:,step_t,:]
            fc_out.append(self.lstm_fc(fc_in))
            # print(step_t, fc_out[-1].shape)
        return torch.stack(fc_out, dim = 1)


class AT_emotion(nn.Module):
    def __init__(self,config):
        super(AT_emotion, self).__init__()

        self.con_encoder = Ct_encoder()
        self.emo_encoder = EmotionNet()
        self.decoder = Decoder()
        self.lm_encoder = Lm_encoder()

        self.CroEn_loss =  nn.CrossEntropyLoss()
        self.mse_loss_fn = nn.MSELoss()
        self.l1loss = nn.L1Loss()

        # self.pca = torch.FloatTensor(np.load(f'{LANDMARK_BASICS}U_68.npy')[:, :16])
        # self.mean = torch.FloatTensor(np.load(f'{LANDMARK_BASICS}mean_68.npy'))

        self.pca = torch.FloatTensor(np.load(f'{LANDMARK_BASICS}U_68.npy')[:, :16]).cuda()
        self.mean = torch.FloatTensor(np.load(f'{LANDMARK_BASICS}mean_68.npy')).cuda()


        self.optimizer = torch.optim.Adam(list(self.con_encoder.parameters())
                                            +list(self.emo_encoder.parameters())
                                            +list(self.decoder.parameters())
                                            +list(self.lm_encoder.parameters()), config.lr,betas=(config.beta1, config.beta2))

    def compute_acc(self,input_label, out):
        _, pred = out.topk(1, 1)
        pred0 = pred.squeeze().data
        acc = 100 * torch.sum(pred0 == input_label.data) / input_label.size(0)
        return acc

    def process(self,example_landmark, landmark, mfccs):
        l = self.lm_encoder(example_landmark)

        lstm_input = []

        for step_t in range(mfccs.size(1)): #16 torch.Size([16, 16, 28, 12])
            mfcc = mfccs[:,step_t,:,:].unsqueeze(1)
            c_feature = self.con_encoder(mfcc)
            e_feature = self.emo_encoder(mfcc)

            current_feature = torch.cat([c_feature,e_feature],1)
            features = torch.cat([l,  current_feature], 1) #torch.Size([16, 768])
            lstm_input.append(features)

        lstm_input = torch.stack(lstm_input, dim = 1)

        fake = self.decoder(lstm_input)

     #   real = landmark - example_landmark.expand_as(landmark)

        loss_pca = self.mse_loss_fn(fake, landmark)

        fake_result = torch.mm(fake[0]+example_landmark,self.pca.transpose(0,1))+self.mean.expand(fake.shape[1],136).unsqueeze(0)
        for i in range(1,len(fake)):
            fake_result = torch.cat((fake_result,torch.mm(fake[i]+example_landmark,self.pca.transpose(0,1))+self.mean.expand(fake.shape[1],136).unsqueeze(0)),0)
        result = torch.mm(landmark[0]+example_landmark,self.pca.transpose(0,1))+self.mean.expand(landmark.shape[1],136).unsqueeze(0)
        for i in range(1,len(landmark)):
            result = torch.cat((result,torch.mm(landmark[i]+example_landmark,self.pca.transpose(0,1))+self.mean.expand(landmark.shape[1],136).unsqueeze(0)),0)

      #  result = torch.mm(landmark,self.pca.transpose(0,1))+self.mean.expand(len(fake),16,136)
        loss_lm = self.mse_loss_fn(fake_result, result)
       # loss = self.l1loss(fake, landmark)
        return fake, loss_pca,10*loss_lm

    def forward(self, example_landmark, mfccs,emo_mfcc):
        self.emo_encoder.eval()
        l = self.lm_encoder(example_landmark.cuda())
        lstm_input = []

        for step_t in range(mfccs.size(1)): #16 torch.Size([16, 16, 28, 12])
   #         current_audio = audio[ : ,step_t , :, :].unsqueeze(1) #unsqueeze(arg) -add argth dimension as 1 torch.Size([16, 1, 28, 12])
   #         current_feature = self.audio_encoder(current_audio) #torch.Size([16, 512, 12, 2])
   #         current_feature = current_feature.view(current_feature.size(0), -1) # torch.Size([16, 12288])
   #         current_feature = self.audio_encoder_fc(current_feature) # torch.Size([16, 256])
            mfcc = mfccs[:,step_t,:,:].unsqueeze(1)
            emo = emo_mfcc[:,step_t,:,:].unsqueeze(1)
            c_feature = self.con_encoder(mfcc)
            e_feature = self.emo_encoder(emo)

            current_feature = torch.cat([c_feature,e_feature],1)
            features = torch.cat([l,  current_feature], 1) #torch.Size([16, 768])
            lstm_input.append(features)

        lstm_input = torch.stack(lstm_input, dim = 1)

        fake = self.decoder(lstm_input)


        return fake

    def feature_input(self, example_landmark, mfccs,emo_feature):

        l = self.lm_encoder(example_landmark)

        lstm_input = []

        for step_t in range(mfccs.size(1)): #16 torch.Size([16, 16, 28, 12])
   #         current_audio = audio[ : ,step_t , :, :].unsqueeze(1) #unsqueeze(arg) -add argth dimension as 1 torch.Size([16, 1, 28, 12])
   #         current_feature = self.audio_encoder(current_audio) #torch.Size([16, 512, 12, 2])
   #         current_feature = current_feature.view(current_feature.size(0), -1) # torch.Size([16, 12288])
   #         current_feature = self.audio_encoder_fc(current_feature) # torch.Size([16, 256])
            mfcc = mfccs[:,step_t,:,:].unsqueeze(1)
            e_feature = emo_feature[:,step_t,:]
            c_feature = self.con_encoder(mfcc)
     #       e_feature = self.emo_encoder(emo)

            current_feature = torch.cat([c_feature,e_feature],1)
            features = torch.cat([l,  current_feature], 1) #torch.Size([16, 768])
            lstm_input.append(features)

        lstm_input = torch.stack(lstm_input, dim = 1)

        fake = self.decoder(lstm_input)


        return fake

    def update_network(self, loss_pca, loss_lm):

        loss = loss_pca + loss_lm
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_learning_rate(self):
        self.scheduler.step(self.clock.epoch)

    def train_func(self, example_landmark, landmark, mfccs):

        self.lm_encoder.train()
        self.decoder.train()
        self.con_encoder.train()
        self.emo_encoder.train()

        output, loss_pca, loss_lm = self.process(example_landmark, landmark, mfccs)

        self.update_network(loss_pca, loss_lm )

        return output, loss_pca, loss_lm

    def val_func(self, example_landmark, landmark, mfccs):
        self.lm_encoder.eval()
        self.decoder.eval()
        self.con_encoder.eval()
        self.emo_encoder.eval()

        with torch.no_grad():
            output, loss_pca, loss_lm  = self.process(example_landmark, landmark, mfccs)

        return output, loss_pca, loss_lm

    def save_fig(self,data,output,save_path):

    #    output1 = outputs['output1']
    #    output2 = outputs['output2']
    #    output12 = outputs['output12']
    #    output21 = outputs['output21']

    #    target1 = data['target11']
    #    target2 = data['target22']
    #    target12 = data['target12']
    #    target21 = data['target21']


        return 0
