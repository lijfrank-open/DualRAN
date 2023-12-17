import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np, argparse, time, random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import IEMOCAPRobertaDataset
import torch.nn as nn
from loss import Focal_Loss, MaskedNLLLoss
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import confusion_matrix, classification_report

from model_vanilla_rnn import MyDialogue

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_IEMOCAP_loaders(path='', batch_size=32, classify='emotion', num_workers=0, pin_memory=False):
    trainset = IEMOCAPRobertaDataset(path=path, split='train', classify=classify)
    validset = IEMOCAPRobertaDataset(path=path, split='valid', classify=classify)
    testset = IEMOCAPRobertaDataset(path=path, split='test', classify=classify)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(validset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader

def train_or_eval_model(model, loss_function, dataloader, epoch, optimizer=None, train=False, cuda_flag=False):
    losses, preds, labels, masks  = [], [], [], []
    vids = []
    max_sequence_len = []

    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()

    seed_everything(seed)
    for data in dataloader:
        if train:
            optimizer.zero_grad()
            
        r1, r2, r3, r4, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda_flag else data[:-1]
        lengths0 = []
        for j, umask_ in enumerate(umask):
            lengths0.append((umask[j] == 1).nonzero()[-1][0] + 1)
        seq_lengths = torch.stack(lengths0)

        log_prob = model(r1, r2, r3, r4, qmask, umask, seq_lengths)

        lp_ = log_prob.transpose(0,1).contiguous().view(-1, log_prob.size()[2])
        labels_ = label.view(-1)

        loss = loss_function(lp_, labels_)

        pred_ = torch.argmax(lp_,1)
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())
        losses.append(loss.item()*masks[-1].sum())

        if train:
            total_loss = loss
            total_loss.backward()
            if args.tensorboard:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()
        else:
            vids += data[-1]

    if preds!=[]:
        preds  = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks  = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), float('nan'), [], [], [], float('nan'),[]

    avg_loss = round(np.sum(losses)/np.sum(masks), 4)

    avg_accuracy = round(accuracy_score(labels,preds, sample_weight=masks)*100, 2)
    avg_fscore = round(f1_score(labels,preds, sample_weight=masks, average='weighted')*100, 2)
    
    return avg_loss, avg_accuracy, labels, preds, masks, [avg_fscore], vids


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', default='/home/lijfrank/code/dataset/IEMOCAP_features/iemocap_features_roberta.pkl', help='dataset dir')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--lr', type=float, default=0.00004, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.0003, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--dropout', type=float, default=0.05, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=32, metavar='BS', help='batch size')
    parser.add_argument('--epochs', type=int, default=80, metavar='E', help='number of epochs')
    parser.add_argument('--class-weight', action='store_true', default=True, help='use class weights')
    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')
    parser.add_argument('--feature_mode', default='concat2')
    parser.add_argument('--ran_mode', default='dran', help='dran, sran1, sran2')
    parser.add_argument('--seed', type=int, default=2023, metavar='seed', help='seed')
    parser.add_argument('--mu', type=float, default=0, help='class_weight_mu')
    parser.add_argument('--classify', default='emotion', help='sentiment, emotion')
    parser.add_argument('--norm_first', action='store_true', default=True, help='norm_first')
    parser.add_argument('--att_residual', action='store_true', default=True, help='use residual connection')
    parser.add_argument('--rnn_residual', action='store_true', default=True, help='use residual connection')
    parser.add_argument('--use_speaker', action='store_true', default=True, help='use speaker connection')
    parser.add_argument('--rnn_model', default='LSTM', help='GRU,LSTM')
    parser.add_argument('--rnn_layer', type=int, default=4, help='')
    parser.add_argument('--attention_head', type=int, default=4, help='')
    parser.add_argument('--attention_layer', type=int, default=5, help='')
    parser.add_argument('--input_size', type=int, default=1024, help='')
    parser.add_argument('--input_in_size', type=int, default=384, help='')
    parser.add_argument('--hidden_size', type=int, default=512, help='')

    args = parser.parse_args()
    print(args)

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    if args.tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter()

    emo_gru = True
    if args.classify == 'emotion':
        n_classes  = 6
    elif args.classify == 'sentiment':
        n_classes  = 3
    cuda       = args.cuda
    n_epochs   = args.epochs
    batch_size = args.batch_size
    n_speakers = 2

    global seed
    seed = args.seed
    seed_everything(seed)
    
    model = MyDialogue(rnn_model=args.rnn_model,
                        rnn_layer=args.rnn_layer,
                        attention_head=args.attention_head,
                        attention_layer=args.attention_layer,
                        norm_first=args.norm_first,
                        att_residual=args.att_residual,
                        rnn_residual=args.rnn_residual,
                        input_size=args.input_size,
                        input_in_size=args.input_in_size,
                        hidden_size=args.hidden_size,
                        feature_mode=args.feature_mode,
                        n_speakers=n_speakers,
                        use_speaker=args.use_speaker,
                        n_classes=n_classes,
                        dropout=args.dropout,
                        cuda_flag=args.no_cuda,
                        ran_mode=args.ran_mode)

    print ('IEMOCAP My Model.')


    if cuda:
        model.cuda()
    if args.classify == 'emotion':
        if args.class_weight:

            loss_weights = torch.FloatTensor([0.087178797,0.145836136,0.229786089,0.148392305,0.140051123,0.248755550])

            loss_function = Focal_Loss(alpha=loss_weights,gamma=1,num_classes=n_classes,size_average=True)

        else:
            loss_function = nn.CrossEntropyLoss()

    else:
        loss_function = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.l2, amsgrad=True)
    
    lf = open('./logs/iemocap_logs.txt', 'a')
    lf.write(str(args) + '\n')
    train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(path=args.data_path,
                                                                  batch_size=batch_size,classify=args.classify,
                                                                  num_workers=0)

    valid_losses, valid_fscores = [], []
    test_fscores, test_losses = [], []
    best_loss, best_label, best_pred, best_mask = None, None, None, None
    best_fscore = []
    best_acc = None

    for e in range(n_epochs):
        start_time = time.time()
        train_loss, train_acc, _, _, _, train_fscore, _ = train_or_eval_model(model, loss_function, train_loader, e, optimizer, train=True, cuda_flag=cuda)
        valid_loss, valid_acc, _, _, _, valid_fscore, _ = train_or_eval_model(model, loss_function, valid_loader, e, cuda_flag=cuda)
        test_loss, test_acc, test_label, test_pred, test_mask, test_fscore, test_vids = train_or_eval_model(model, loss_function, test_loader, e, cuda_flag=cuda)
            
        valid_losses.append(valid_loss)
        valid_fscores.append(valid_fscore)
        test_losses.append(test_loss)
        test_fscores.append(test_fscore)
        
        if args.tensorboard:
            writer.add_scalar('test: accuracy/loss', test_acc/test_loss, e)
            writer.add_scalar('train: accuracy/loss', train_acc/train_loss, e)
            
        x = 'epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, valid_loss: {}, valid_acc: {}, valid_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'.format(e+1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore, test_loss, test_acc, test_fscore, round(time.time()-start_time, 2))
        
        print(x)

        if best_fscore == [] or best_fscore[0] < test_fscore[0]: 
            best_fscore, best_acc = test_fscore, test_acc
            best_label, best_pred, best_mask = test_label, test_pred, test_mask

        if (e+1)%10 == 0:
                x2 = classification_report(best_label, best_pred,sample_weight=best_mask, digits=4, zero_division=0)
                print(x2)
                lf.write('epoch: {}'.format(e+1) + '\n')
                lf.write(str(x2) + '\n')
                
                x3 = confusion_matrix(best_label, best_pred, sample_weight=best_mask)
                print(x3)
                lf.write(str(x3) + '\n')

                x4 = 'test_best_acc: {}, [test_best_fscore]: {}'.format(best_acc, best_fscore)
                print(x4)
                lf.write(x4 + '\n')

                print('-'*150)
               
    if args.tensorboard:
        writer.close()
        
    valid_fscores = np.array(valid_fscores).transpose()
    test_fscores = np.array(test_fscores).transpose()

    valid_best_fscore = np.max(valid_fscores[0])
    test_best_fscore = np.max(test_fscores[0])

    print('valid_best_fscore:', valid_best_fscore)
    print('test_best_fscore:', test_best_fscore)
       
    scores = [valid_best_fscore, test_best_fscore]
    scores = [str(item) for item in scores]

    rf = open('results/iemocap_results.txt', 'a')
    rf.write('\t'.join(scores) + '\t' + str(args) + '\n')
    rf.close()
    