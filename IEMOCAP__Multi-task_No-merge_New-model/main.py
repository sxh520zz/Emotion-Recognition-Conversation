import os
import time
import random
import argparse
import pickle
import copy
import torch
import numpy as np
import torch.utils.data as Data
import torch.nn.utils.rnn as rmm_utils
import torch.utils.data.dataset as Dataset
import torch.optim as optim
from utils import Get_data
from torch.autograd import Variable
from models import Utterance_net,Dialogue_net,Output_net,Output_net_1,Output_net_2,Utterance_net_Input
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import recall_score



os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.backends.cudnn.enabled = False

with open('Train_data_NEW_Model.pickle', 'rb') as file:
    data = pickle.load(file)

parser = argparse.ArgumentParser(description="RNN_Model")
parser.add_argument('--cuda', action='store_false')
parser.add_argument('--bid_flag', action='store_false')
parser.add_argument('--batch_first', action='store_false')
parser.add_argument('--batch_size', type=int, default=1, metavar='N')
parser.add_argument('--log_interval', type=int, default=1000, metavar='N')
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--optim', type=str, default='Adam')
parser.add_argument('--seed', type=int, default=1111)
parser.add_argument('--dia_layers', type=int, default=2)
parser.add_argument('--hidden_layer', type=int, default=256)
parser.add_argument('--out_class', type=int, default=4)
parser.add_argument('--out_class_1', type=int, default=1)
parser.add_argument('--utt_insize', type=int, default=88)
parser.add_argument('--Dia_insize', type=int, default=600)
args = parser.parse_args()

torch.manual_seed(args.seed)


def concordance_correlation_coefficient(y_true, Y_pred,
                                        sample_weight=None,
                                        multioutput='uniform_average'):
    """Concordance correlation coefficient.
    The concordance correlation coefficient is a measure of inter-rater agreement.
    It measures the deviation of the relationship between predicted and true values
    from the 45 degree angle.
    Read more: https://en.wikipedia.org/wiki/Concordance_correlation_coefficient
    Original paper: Lawrence, I., and Kuei Lin. "A concordance correlation coefficient to evaluate reproducibility." Biometrics (1989): 255-268.
    Parameters
    ----------
    y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Estimated target values.
    Returns
    -------
    loss : A float in the range [-1,1]. A value of 1 indicates perfect agreement
    between the true and the predicted values.
    Examples
    --------
    0.97678916827853024
    """
    y_pred = []
    for i in range(len(Y_pred)):
        y_pred.append(Y_pred[i][0].tolist())

    cor = np.corrcoef(y_true, y_pred)[0][1]

    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)

    var_true = np.var(y_true)
    var_pred = np.var(y_pred)

    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)

    numerator = 2 * cor * sd_true * sd_pred

    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2

    return numerator / denominator

def Train(epoch):
    train_loss = 0
    dia_net_a.train()
    dia_net_b.train()
    dia_net_all.train()
    output_net.train()
    output_net_1.train()
    #output_net_2.train()
    # data_1: input_train_data_trad,torch.Size([1, 6, 1, 88])
    # data_2: input_train_data_trad/2,torch.Size([1, 3, 1, 88])
    # data_3: input_train_data_trad/2,torch.Size([1, 3, 1, 88])
    # data_1_1: input_train_data_tran,torch.Size([1, 6, 1, 768])
    # data_2_1: input_train_data_tran/2,torch.Size([1, 3, 1, 768])
    # data_3_1: input_train_data_tran/2,torch.Size([1, 3, 1, 768])
    # label: emotion_label
    for batch_idx, (data_1,data_2,data_3,data_1_1,data_2_1,data_3_1,target,target_1,ID_1,ID_2) in enumerate(train_loader):
        if args.cuda:
            data_1, data_2, data_3, data_1_1, data_2_1, data_3_1, target, target_1, ID_1, ID_2 = \
                data_1.cuda(), data_2.cuda(), data_3.cuda(), data_1_1.cuda(), data_2_1.cuda(), data_3_1.cuda(), target.cuda(), target_1.cuda(), ID_1.cuda(), ID_2.cuda()
        # data (batch_size, step, 88)
        # target (batch_size, 1)
        data_1, data_2, data_3, data_1_1, data_2_1, data_3_1, target, target_1,  ID_1, ID_2 = \
            Variable(data_1), Variable(data_2),Variable(data_3),Variable(data_1_1),Variable(data_2_1),Variable(data_3_1),Variable(target),Variable(target_1),Variable(ID_1),Variable(ID_2)

        target = target.squeeze(0)
        target_1 = target_1.squeeze(0)
        ID_1 = ID_1.squeeze(0)
        ID_2 = ID_2.squeeze(0)

        data_1 = data_1.squeeze(2)
        data_2 = data_2.squeeze(2)
        data_3 = data_3.squeeze(2)

        data_1_1 = data_1_1.squeeze(2)
        data_2_1 = data_2_1.squeeze(2)
        data_3_1 = data_3_1.squeeze(2)

        Gru_input_1 = torch.cat((data_2,data_2_1), 2)
        Gru_input_2 = torch.cat((data_3,data_3_1), 2)
        Gru_input_3 = torch.cat((data_1,data_1_1), 2)

        # print(dia_out_a.size())
        # print(dia_hid_a.size())

        dia_out_all_concate, dia_out_all = dia_net_all(data_3)
        speaker_1 = dia_out_all.index_select(1, ID_1)
        speaker_2 = dia_out_all.index_select(1, ID_2)

        Gru_1 = torch.cat((data_1,speaker_1), 2)
        Gru_2 = torch.cat((data_2,speaker_2), 2)

        dia_out_a, dia_hid_a = dia_net_a(Gru_1)
        dia_out_b, dia_hid_b = dia_net_b(Gru_2)

        line_input = torch.cat((dia_out_a,dia_out_b,dia_out_all_concate), 1)
        line_out = output_net(line_input)
        line_out_1 = output_net_1(line_input)


        dia_net_a_optimizer.zero_grad()
        dia_net_b_optimizer.zero_grad()
        dia_net_all_optimizer.zero_grad()
        output_net_optimizer.zero_grad()
        output_net_1_optimizer.zero_grad()
        #output_net_2_optimizer.zero_grad()

        #loss = torch.nn.CrossEntropyLoss()(line_out, target.long())
        loss_1 = torch.nn.CrossEntropyLoss()(line_out, target.long())
        loss_2 = torch.nn.MSELoss()(line_out_1, target_1)
        #loss_2 = torch.nn.CrossEntropyLoss()(line_out_1, target_1.long())
        #loss_3 = torch.nn.CrossEntropyLoss()(line_out_2, target_2.long())

        loss = loss_1 + loss_2
        loss.backward()

        dia_net_a_optimizer.step()
        dia_net_b_optimizer.step()
        dia_net_all_optimizer.step()
        output_net_optimizer.step()
        output_net_1_optimizer.step()
        #output_net_2_optimizer.step()

        train_loss += loss

        if batch_idx > 0 and batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * args.batch_size, len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), train_loss.item() / args.log_interval
            ))
            train_loss = 0
def Test():
    dia_net_a.eval()
    dia_net_b.eval()
    dia_net_all.eval()
    output_net.eval()
    output_net_1.eval()
    #output_net_2.eval()
    label_pre = []
    label_true = []


    with torch.no_grad():
        for batch_idx, (data_1,data_2,data_3,data_1_1,data_2_1,data_3_1,target,target_1,ID_1,ID_2) in enumerate(
                test_loader):

            if args.cuda:
                data_1, data_2, data_3, data_1_1, data_2_1, data_3_1, target, target_1, ID_1, ID_2 = \
                    data_1.cuda(), data_2.cuda(), data_3.cuda(), data_1_1.cuda(), data_2_1.cuda(), data_3_1.cuda(), target.cuda(), target_1.cuda(), ID_1.cuda(), ID_2.cuda()
            # data (batch_size, step, 88)
            # target (batch_size, 1)
            data_1, data_2, data_3, data_1_1, data_2_1, data_3_1, target, target_1, ID_1, ID_2 = \
                Variable(data_1), Variable(data_2), Variable(data_3), Variable(data_1_1), Variable(data_2_1), Variable(
                    data_3_1), Variable(target), Variable(target_1), Variable(ID_1), Variable(ID_2)

            target = target.squeeze(0)
            target_1 = target_1.squeeze(0)
            ID_1 = ID_1.squeeze(0)
            ID_2 = ID_2.squeeze(0)

            data_1 = data_1.squeeze(2)
            data_2 = data_2.squeeze(2)
            data_3 = data_3.squeeze(2)

            data_1_1 = data_1_1.squeeze(2)
            data_2_1 = data_2_1.squeeze(2)
            data_3_1 = data_3_1.squeeze(2)

            Gru_input_1 = torch.cat((data_2, data_2_1), 2)
            Gru_input_2 = torch.cat((data_3, data_3_1), 2)
            Gru_input_3 = torch.cat((data_1, data_1_1), 2)

            # print(dia_out_a.size())
            # print(dia_hid_a.size())

            dia_out_all_concate, dia_out_all = dia_net_all(data_3)
            speaker_1 = dia_out_all.index_select(1, ID_1)
            speaker_2 = dia_out_all.index_select(1, ID_2)

            Gru_1 = torch.cat((data_1, speaker_1), 2)
            Gru_2 = torch.cat((data_2, speaker_2), 2)

            dia_out_a, dia_hid_a = dia_net_a(Gru_1)
            dia_out_b, dia_hid_b = dia_net_b(Gru_2)

            line_input = torch.cat((dia_out_a, dia_out_b, dia_out_all_concate), 1)
            line_out = output_net(line_input)
            output = torch.argmax(line_out, dim=1)

            label_true.extend(target.cpu().data.numpy())
            label_pre.extend(output.cpu().data.numpy())
        '''
        accuracy_recall = concordance_correlation_coefficient(label_true, label_pre)
        accuracy_f1 = mean_squared_error(label_true, label_pre)

        print("########################################")
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print(accuracy_recall)
        print(accuracy_f1)
        print("########################################")

        '''
        accuracy_recall = recall_score(label_true, label_pre, average='macro')
        accuracy_f1 = metrics.f1_score(label_true, label_pre, average='macro')
        CM_test = confusion_matrix(label_true, label_pre)


        print("########################################")
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print(accuracy_recall)
        print(accuracy_f1)
        print(CM_test)
        print("########################################")


    return accuracy_f1, accuracy_recall, label_pre, label_true


Final_result = []
Final_f1 = []
kf = KFold(n_splits=5)
for index, (train, test) in enumerate(kf.split(data)):
    print(index)

    train_loader, test_loader,input_test_data_id, input_test_label_org, test_len = Get_data(data, train, test, args)

    dia_net_a = Utterance_net(args.Dia_insize, args)
    dia_net_b = Utterance_net(args.Dia_insize, args)
    dia_net_all = Utterance_net_Input(args.utt_insize, args)
    output_net = Output_net(1536 , args)
    output_net_1 = Output_net_1(1536, args)
    output_net_2 = Output_net_2(1536, args)


    if args.cuda:
        dia_net_a = dia_net_a.cuda()
        dia_net_b = dia_net_b.cuda()
        dia_net_all = dia_net_all.cuda()
        output_net = output_net.cuda()
        output_net_1 = output_net_1.cuda()
        output_net_2 = output_net_2.cuda()


    lr = args.lr
    dia_net_a_optimizer = getattr(optim, args.optim)(dia_net_a.parameters(), lr=lr)
    dia_net_b_optimizer = getattr(optim, args.optim)(dia_net_b.parameters(), lr=lr)
    dia_net_all_optimizer = getattr(optim, args.optim)(dia_net_all.parameters(), lr=lr)
    output_net_optimizer = getattr(optim, args.optim)(output_net.parameters(), lr=lr)
    output_net_1_optimizer = getattr(optim, args.optim)(output_net_1.parameters(), lr=lr)
    output_net_2_optimizer = getattr(optim, args.optim)(output_net_2.parameters(), lr=lr)


    dia_net_a_optim = optim.Adam(dia_net_a.parameters(), lr=lr)
    dia_net_b_optim = optim.Adam(dia_net_b.parameters(), lr=lr)
    dia_net_all_optim = optim.Adam(dia_net_all.parameters(), lr=lr)
    output_net_optim = optim.Adam(output_net.parameters(), lr=lr)
    output_net_1_optim = optim.Adam(output_net_1.parameters(), lr=lr)
    output_net_2_optim = optim.Adam(output_net_2.parameters(), lr=lr)

    f1 = 0
    recall = 0
    for epoch in range(1, args.epochs + 1):
        Train(epoch)
        accuracy_f1, accuracy_recall, pre_label, true_label = Test()
        if epoch % 10 == 0:
            lr /= 10
            for param_group in dia_net_a_optimizer.param_groups:
                param_group['lr'] = lr
            for param_group in dia_net_b_optimizer.param_groups:
                param_group['lr'] = lr
            for param_group in dia_net_all_optimizer.param_groups:
                param_group['lr'] = lr
            for param_group in output_net_optimizer.param_groups:
                param_group['lr'] = lr
            for param_group in output_net_1_optimizer.param_groups:
                param_group['lr'] = lr
            for param_group in output_net_2_optimizer.param_groups:
                param_group['lr'] = lr

        if (accuracy_f1 > f1 and accuracy_recall > recall):
            predict = copy.deepcopy(input_test_label_org)
            num = 0
            for x in range(len(predict)):
                predict[x] = pre_label[num]
                num = num + 1
            result_label = predict
            recall = accuracy_recall
            f1 = accuracy_f1
        print("Best Result until now: ", f1)
    onegroup_result = []

    for i in range(len(input_test_data_id)):
        a = {}
        a['id'] = input_test_data_id[i]
        a['Predict_label'] = pre_label[i]
        a['True_label'] = input_test_label_org[i]
        onegroup_result.append(a)
    Final_result.append(onegroup_result)
    Final_f1.append(f1)

file = open('Final_result.pickle', 'wb')
pickle.dump(Final_result, file)
file.close()
file = open('Final_f1.pickle', 'wb')
pickle.dump(Final_f1, file)
file.close()

true_label = []
predict_label = []
for i in range(len(Final_result)):
    for j in range(len(Final_result[i])):
        predict_label.append(Final_result[i][j]['Predict_label'])
        true_label.append(Final_result[i][j]['True_label'])

accuracy_recall = recall_score(true_label, predict_label, average='macro')
accuracy_acc = accuracy_score(true_label, predict_label)
CM_test = confusion_matrix(true_label, predict_label)

print(len(true_label))
print(accuracy_recall, accuracy_acc)
print(CM_test)