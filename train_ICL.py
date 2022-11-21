import torch
import torch_geometric.data
import os
import numpy as np
from criterion import loss_cal_ICL_old, loss_cal_ICL_new
from augmentation import aug_graph, aug_img
import time
from util import evaluate_embedding


def train_ICL(data, old_data, new_data, encoder, args):
    # optimizer
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # data loader
    if args.mode == 'graph':
        loader = torch_geometric.data.DataLoader(data, batch_size=args.batch_size, shuffle=True)
        old_loader = torch_geometric.data.DataLoader(old_data, batch_size=args.batch_size, shuffle=True)
        new_loader = torch_geometric.data.DataLoader(new_data, batch_size=args.batch_size, shuffle=True)
        aug = aug_graph
    elif args.mode == 'cv':
        loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=True)
        old_loader = torch.utils.data.DataLoader(old_data, batch_size=args.batch_size, shuffle=True)
        new_loader = torch.utils.data.DataLoader(new_data, batch_size=args.batch_size, shuffle=True)
        aug = aug_img
    
    # train data 
    final_epoch = 0
    results = []
    total_time = 0
    count = 0
    pkl_save = '{}.{}.{}.pkl'.format(args.dataset, args.alpha, time.time())
    
    for epoch in range(args.epochs):
        loss_all = 0
        
        # train
        encoder.train()        
        start_time = time.time()
        for old_batch in old_loader:
            old_batch, old_batch_aug = aug(old_batch, np.random.randint(3))
            optimizer.zero_grad()
            old_out = encoder(old_batch)
            old_out_aug = encoder(old_batch_aug)
            
            try:
                if args.mode == 'graph':
                    new_batch = new4old_iter.next()
                elif args.mode == 'cv':
                    new_batch = new4old_iter.next()[0][0]
            except:
                if args.mode == 'graph':
                    new4old_iter = iter(torch_geometric.data.DataLoader(new_data, batch_size=args.batch_size-1, shuffle=True))
                    new_batch = new4old_iter.next()   
                elif args.mode == 'cv':
                    new4old_iter = iter(torch.utils.data.DataLoader(new_data, batch_size=args.batch_size-1, shuffle=True))
                    new_batch = new4old_iter.next()[0][0]
                
            new_out = encoder(new_batch)
            
            loss = loss_cal_ICL_old(old_out, old_out_aug, new_out, args.alpha)
                        
            loss_all += loss.item() #* data.num_graphs
            loss.backward()
            optimizer.step()
        
        for new_batch in new_loader:
            if args.mode == 'graph':
                batch_size = new_batch.num_graphs
            elif args.mode == 'cv':
                batch_size = len(new_batch[0][0])
            old_num = int((1 - args.alpha) * (batch_size-1))
            new_num = batch_size-1 - old_num
            
            new_batch, new_batch_aug = aug(new_batch, np.random.randint(3))

            optimizer.zero_grad()
            new_out = encoder(new_batch)
            new_out_aug = encoder(new_batch_aug)
            
            if old_num == 0:
                old_out = None
            else:                
                try:
                    if args.mode == 'graph':
                        old_batch = old4new_iter.next()
                    elif args.mode == 'cv':
                        old_batch = old4new_iter.next()[0][0]
                except:
                    if args.mode == 'graph':
                        old4new_iter = iter(torch_geometric.data.DataLoader(old_data, batch_size=old_num, shuffle=True))
                        old_batch = old4new_iter.next()
                    elif args.mode == 'cv':
                        old4new_iter = iter(torch.utils.data.DataLoader(old_data, batch_size=old_num, shuffle=True))
                        old_batch = old4new_iter.next()[0][0]
                    
                old_out = encoder(old_batch)
            
            loss = loss_cal_ICL_new(new_out, new_out_aug, old_out, new_num)
                        
            loss_all += loss.item() #* data.num_graphs
            loss.backward()
            optimizer.step()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        end_time = time.time()
       
        # get result
        average_loss = loss_all / (len(old_loader) + len(new_loader))
        epoch_time = end_time - start_time
        total_time += epoch_time
        result = [epoch, average_loss, epoch_time, total_time]
        results.append(result)
        if epoch % 10 == 0:
            print('Epoch {:<5}, Loss {:.4f}, time {:.4f}, total time {:.4f}'.format(*result))
        
        if results[final_epoch][1] >= average_loss:
            final_epoch = epoch
            try:
                torch.save(encoder.module.state_dict(), pkl_save)
            except:
                torch.save(encoder.state_dict(), pkl_save)
            count = 0
        else:
            count += 1
            if count == args.patience:
                break
    
    print('Eval!')
    encoder.load_state_dict(torch.load(pkl_save))
    encoder.eval()

    embs, y = encoder.get_embeddings(loader)
    acc = evaluate_embedding(embs, y)
    embs, y = encoder.get_embeddings(old_loader)
    acc_old = evaluate_embedding(embs, y)
    embs, y = encoder.get_embeddings(new_loader)
    acc_new = evaluate_embedding(embs, y)
    print('Final Epoch {:<5}, Loss {:.4f}, time {:.4f}, total time {:.4f}, acc {:.4f}, acc-old {:.4f}, acc-new {:.4f}'.format(*(results[final_epoch]+[acc, acc_old, acc_new])))
    
    os.remove(pkl_save)
    
