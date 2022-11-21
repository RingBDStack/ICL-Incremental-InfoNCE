import torch
import torch_geometric.data
import os
import numpy as np
from criterion import loss_cal
from augmentation import aug_graph, aug_img
import time
from util import evaluate_embedding

def train(data, old_data, new_data, encoder, args, mode='old', save=False):
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
    
    if mode == 'old':
        train_loader = old_loader
    elif mode == 'new':
        train_loader = new_loader
    elif mode == 'all':
        train_loader = loader
    
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
        for batch in train_loader:
            batch, batch_aug = aug(batch, np.random.randint(3))
            
            optimizer.zero_grad()
            
            out = encoder(batch)            
            out_aug = encoder(batch_aug)

            loss = loss_cal(out, out_aug)
            if torch.isnan(loss):
                # loss = loss_cal(out, out_aug, debug=False)
                break
            loss_all += loss.item()  # * batch.num_graphs
            loss.backward()
            optimizer.step()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        end_time = time.time()

        # get result
        average_loss = loss_all / len(train_loader)
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
    
    if not save:
        os.remove(pkl_save)
        return
    
    return pkl_save, loss.item()
