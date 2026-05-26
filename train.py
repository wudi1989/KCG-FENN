import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from data_loader import Data_loader
from model import Cat_model
from Evaluate import eval_all

def eliminate_0(inputs, near_0 = 1e-8):
    return torch.clamp(inputs, min=near_0)

def save_snapshot(model, filename):
    with open(filename, 'wb') as f:
        torch.save(model.state_dict(), f)

def train():
    user_num = 536
    exer_num = 20
    know_num = 8
    exp_num = 128
    datasetName = 'FrcSub'
    batch_size = 32
    batch_frequency = 200
    lr = 0.008
    weight_decay = 0
    epochs = 200
    clipper = True
    
    data_loader = Data_loader(f"train_1", know_num, batch_size)
    net = Cat_model(user_num=user_num, exer_num=exer_num, know_num=know_num, exp_num=exp_num).to('cuda')
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    
    best_val_auc = 0.0
    checkpoint_path = f"1.pth"
    
    for epoch in range(1, epochs + 1):
        net.train()
        data_loader.reset()
        
        while not data_loader.is_end():
            input_stu_ids, input_exer_ids, labels = data_loader.next_batch()
            input_stu_ids, input_exer_ids, labels = input_stu_ids.to('cuda'), input_exer_ids.to('cuda'), labels.to('cuda')
            
            optimizer.zero_grad()
            output_1 = net.forward(input_stu_ids, input_exer_ids)
            output_0 = torch.ones(output_1.size()).to('cuda') - output_1
            output = torch.cat((output_0, output_1), 1)
            output = eliminate_0(output, 1e-8)
            
            output_1 = output_1.view(-1)
            labels_f = labels.float()
            bce_loss = torch.nn.functional.binary_cross_entropy(output_1, labels_f, reduction='none')
            pt = torch.exp(-bce_loss)
            gamma = 2.0
            loss = torch.mean(((1.0 - pt) ** gamma) * bce_loss)
            
            loss.backward()
            optimizer.step()

        acc, aupr, auc, f1, _, _ = tt(net, epoch, know_num, batch_size)
        
        if auc > best_val_auc:
            best_val_auc = auc
            save_snapshot(net, checkpoint_path)
            
    return best_val_auc

def tt(model, epoch, know_num, batch_size):
    data_loader = Data_loader(f"test_1", know_num, batch_size)
    model.eval()
    data_loader.reset()
    
    pred_all, label_all = [], []

    while not data_loader.is_end():
        input_stu_ids, input_exer_ids, labels = data_loader.next_batch()
        input_stu_ids, input_exer_ids, labels = input_stu_ids.to('cuda'), input_exer_ids.to('cuda'), labels.to('cuda')
        output = model.forward(input_stu_ids, input_exer_ids)
        output = output.view(-1)
        pred_all += output.to(torch.device('cpu')).tolist()
        label_all += labels.to(torch.device('cpu')).tolist()

    Acc, AUPR, AUC, F1, Pre, Rec = eval_all(pred=pred_all, label=label_all)
    print('epoch= %d, accuracy= %f, aupr= %f, auc= %f, f1= %f, pre= %f, rec= %f' % (epoch, Acc, AUPR, AUC, F1, Pre, Rec))

    return Acc, AUPR, AUC, F1, Pre, Rec

def validate(model, epoch,k):
    data_loader = Data_loader('valid_%s'%str(k+1),know_num)
    net = Cat_model(user_num=user_num,exer_num=exer_num,know_num=know_num,exp_num=exp_num)
    print('validating model...')
    data_loader.reset()
    # load model parameters
    net.load_state_dict(model.state_dict())
    net = net.to('cuda')
    net.eval()

    batch_count, batch_avg_loss = 0, 0.0
    pred_all, label_all = [], []
    while not data_loader.is_end():
        batch_count =batch_count+ 1
        input_stu_ids, input_exer_ids, labels = data_loader.next_batch()
        input_stu_ids, input_exer_ids, labels = input_stu_ids.to('cuda'), input_exer_ids.to('cuda'), labels.to('cuda')
        output = net.forward(input_stu_ids, input_exer_ids)
        output = output.view(-1)

        pred_all =pred_all+ output.to(torch.device('cpu')).tolist()
        label_all =label_all+ labels.to(torch.device('cpu')).tolist()


    Acc,AUPR,AUC,F1,Pre,Rec=eval_all(pred=pred_all,label=label_all)
    print('epoch= %d, accuracy= %f, aupr= %f, auc= %f, f1= %f, pre= %f, rec= %f' % (epoch+1, Acc,AUPR,AUC,F1,Pre,Rec))
    with open('result/model_val_%s.txt'%str(i+1), 'a', encoding='utf8') as f:
        f.write('epoch= %d, accuracy= %f, aupr= %f, auc= %f, f1= %f, pre= %f, rec= %f\n' % (epoch+1, Acc,AUPR,AUC,F1,Pre,Rec))

    return Acc,AUPR,AUC,F1,Pre,Rec

if __name__ == '__main__':
    train()