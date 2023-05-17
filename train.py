import torch
import numpy as np
import time
import warnings
import parser
import argparse
from utils import *
from model import *
import torch.optim as optim
import json


# torch.set_printoptions(threshold=np.inf)
torch.set_printoptions(sci_mode=False)
warnings.filterwarnings('ignore')
torch.set_default_dtype(torch.float64)


def train( idx_clean, idx_unknown, idx_train, idx_val, idx_test, adj, features, 
          noise_labels, origi_labels, y_clean, y_unknown, args):
    new_labels = noise_labels.clone()
    model = MLP_NORM(
        nnodes=adj.shape[0],
        nfeat=features.shape[1],
        nhid=args.hidden,
        nclass=origi_labels.max().item() + 1,
        dropout=args.dropout,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        delta=args.delta,
        norm_func_id=args.norm_func_id,
        norm_layers=args.norm_layers,
        orders=args.orders,
        orders_func_id=args.orders_func_id,
        cuda=args.cuda)
    
    if args.cuda:
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        origi_labels = origi_labels.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    t_total = time.time()
    final_acc = float('-inf')
    for _ in range(args.epochs_lp):
        best_val_loss = float('inf')
        acc = float('-inf')
        cost_val = []

        if args.cuda:
            new_labels = new_labels.cuda()
            y_clean = y_clean.cuda()
            y_unknown = y_unknown.cuda()
            idx_clean = idx_clean.cuda()
            idx_unknown = idx_unknown.cuda()
        
        # train gnn
        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()
            output = model(features, adj, y_clean, y_unknown, args, if_lp=False)

            loss_train = F.nll_loss(output[idx_clean], new_labels[idx_clean])
            acc_train = accuracy(output[idx_clean], new_labels[idx_clean])
            loss_train.backward()
            optimizer.step()

            model.eval()
            output = model(features, adj, y_clean, y_unknown, args, if_lp=False)
            
            loss_val = F.nll_loss(output[idx_val], origi_labels[idx_val])
            acc_val = accuracy(output[idx_val], origi_labels[idx_val])
            # print(origi_labels[idx_test])
            # print(output[idx_test])
            loss_test = F.nll_loss(output[idx_test], origi_labels[idx_test])
            acc_test = accuracy(output[idx_test], origi_labels[idx_test])
            
            # print('Epoch: {:04d}'.format(epoch+1),
            #       'loss_train: {:.4f}'.format(loss_train.item()),
            #       'acc_train: {:.4f}'.format(acc_train.item()),
            #       'loss_test: {:.4f}'.format(loss_test.item()),
            #       'acc_test: {:.4f}'.format(acc_test.item()),
            #       'time: {:.4f}s'.format(time.time() - t))

            if loss_val <= best_val_loss: 
                best_val_loss = loss_val
                if acc_test > acc:
                    acc = acc_test
                    test_loss = loss_test
                    para = model.state_dict()

            cost_val.append(loss_val.item())
            if epoch > args.early_stopping and cost_val[-1] > np.mean(cost_val[-(args.early_stopping+1):-1]):
                # print("Early stopping...")
                break

        print("GNN Test set results:",
            "loss= {:.4f}".format(test_loss.item()),
            "accuracy= {:.4f}".format(acc.item()))
        
        torch.save(para, 'RoGNN/para/'+args.dataset+args.noise_type+'.pth')

        # label propagation
        model.load_state_dict(torch.load('RoGNN/para/'+args.dataset+args.noise_type+'.pth'))
        model.eval()
        output, predict_z, y_predict, h_l, fl = model(features, adj, y_clean, y_unknown, args, if_lp=True)

        # if args.lp_function == 1:
        #     F_t, new_idx_unknown, new_idx_clean, labels_lp, y_clean = label_correction(args.num_propagations, 
        #         new_labels, predict_z.detach(), y_clean, y_unknown, idx_clean, idx_unknown, y_predict.detach(), h_l.detach(),
        #         args.lamada1, args.lamada2, args.lamada3, args.pre_select, args.eps_adj, args.func_Z, origi_labels)
        # else:
        F_t = fl
        new_idx_unknown, new_idx_clean, labels_lp = new_clean(F_t, args.pre_select, idx_clean, idx_unknown, new_labels, y_clean, origi_labels)

        idx_unknown = new_idx_unknown
        idx_clean = new_idx_clean
        new_labels = labels_lp
        print('num_celan:', idx_clean.shape[0], 'num_unknown:', idx_unknown.shape[0])

        # acc_test_lp = accuracy(F_t[idx_test], origi_labels[idx_test])
        # print("Test set results of lp:",
        #     # "loss= {:.4f}".format(test_loss.item()),
        #     "accuracy= {:.4f}".format(acc_test_lp.item()))
        
        if acc > final_acc:
            final_acc = acc  
            # lp_acc = acc_test_lp
        
    print("Finished!")
    print('Final accuracy:', final_acc)
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    return final_acc
    

def Process():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--split', type=int, default=0, help='Split part of dataset')
    parser.add_argument('--dataset', type=str, default='cora', help='Name of dataset')
    parser.add_argument('--pre_clean', type=float, default=0.1, help='The pre of clean label in all samples')
    parser.add_argument('--pre_unknown', type=float, default=0.5, help='The pre of unkonwn label in all samples')
    parser.add_argument('--pre_noise', type=float, default=0.2, help='The pre of noise label in unkonwn samples')
    parser.add_argument('--noise_type', type=str, default='uniform', help='type of noise')

    parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
    parser.add_argument('--norm_layers', type=int, default=2, help='Number of groupnorm layers')
    parser.add_argument('--lr', type=float, default=0.03, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-7, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--dropout', type=float, default=0.9, help='Dropout rate (1 - keep probability).')

    parser.add_argument('--delta', type=float, default=0.9, help='Weight for nodes feature kept')
    parser.add_argument('--alpha', type=float, default=77, help='Weight for nodes feature kept')
    parser.add_argument('--beta', type=float, default=1000, help='Weight for frobenius norm on Z-A')
    parser.add_argument('--gamma', type=float, default=0.8, help='Weight for MLP results kept')
    parser.add_argument('--orders', type=int, default=6, help='Number of adj orders in norm layer')
    parser.add_argument('--orders_func_id', type=int, default=2, help='Sum function of adj orders in norm layer, ids \in [1, 2, 3]')
    parser.add_argument('--norm_func_id', type=int, default=1, help='Function of norm layer, ids \in [1, 2]')

    parser.add_argument('--epochs_lp', type=int, default=20, help='Number of epochs to train.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--early_stopping', type=int, default=40, help='Early stopping')

    parser.add_argument('--lp_function', type=int, default=2, help='Function of label propagation')
    parser.add_argument('--num_propagations', type=int, default=8)
    parser.add_argument('--alpha1', type=float, default=0.4,)
    parser.add_argument('--alpha2', type=float, default=0.3,)
    parser.add_argument('--alpha3', type=float, default=0.5,)
    parser.add_argument('--pre_select', type=float, default=0.5)
    # parser.add_argument('--eps_adj', type=float, default=0.1) 
    parser.add_argument('--func_Z', type=int, default=1)

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print(args.cuda)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    
    gnn_acc = []
    lp_acc= []
    for _ in range(10):
        args.split = _

        # data
        adj, features, labels, idx_train, idx_val, idx_test = load_data_new(args.dataset, args.split)
        features = features.to(torch.float64)
        adj = adj.to(torch.float64)
        idx_clean, idx_unknown, idx_val, idx_test = reset_idx(idx_train, idx_val, idx_test, args.pre_clean, args.pre_unknown)
        origi_labels = labels.clone()
        noise_labels, y_clean, y_unknown = add_noise(args.pre_noise, labels, idx_clean, idx_unknown, args.noise_type)
        print('dataset:', args.dataset)
        print('num_celan:', idx_clean.shape[0])
        print('num_unknown:', idx_unknown.shape[0])

        # train
        final_acc = train(idx_clean, idx_unknown, idx_train, idx_val, idx_test, adj, features, 
                        noise_labels, origi_labels, y_clean, y_unknown, args)
        
        # lp_acc.append(lp.cpu())
        gnn_acc.append(final_acc.cpu())

    print('ave_acc: {:.4f}'.format(np.mean(gnn_acc)), '+/- {:.4f}'.format(np.std(gnn_acc)))
    # print('ave_acc_lp: {:.4f}'.format(np.mean(lp_acc)), '+/- {:.4f}'.format(np.std(lp_acc)))

    outfile_name = f"{args.dataset}_noise_type{args.noise_type}_results.txt"
    Hyperparameters = f"noisepre{args.pre_noise}_lr{args.lr}_do{args.dropout}_" +\
        f"wd{args.weight_decay}_alpha{args.alpha}_beta{args.beta}_gamma{args.gamma}_" +\
        f"delta{args.delta}_nlid{args.norm_func_id}_nl{args.norm_layers}_" +\
        f"ordersid{args.orders_func_id}_orders{args.orders}_split{args.split}_cleanpre{args.pre_clean}_" +\
        f"alpha1{args.alpha1}_alpha2{args.alpha2}_alpha3{args.alpha3}_num_propagations{args.num_propagations}_pre_select{args.pre_select}"
    print(outfile_name)

    results_dict = {}
    results_dict['test_acc_mean'] = float(np.mean(gnn_acc))
    results_dict['test_acc_std'] = float(np.std(gnn_acc))
    # results_dict['test_acc_mean_lp'] = float(np.mean(lp_acc))
    # results_dict['test_acc_std_lp'] = float(np.std(lp_acc))

    with open(os.path.join('RoGNN/run', outfile_name), 'a') as outfile:
        outfile.write(Hyperparameters)
        outfile.write('\n')
        outfile.write(json.dumps(results_dict))
        outfile.write('\n')


if __name__ == "__main__":
    Process()