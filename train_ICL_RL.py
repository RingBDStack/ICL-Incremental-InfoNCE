import torch
import torch_geometric.data
import os
import numpy as np
from criterion import loss_cal_ICL_old, loss_cal_ICL_new
from augmentation import aug_graph, aug_img
import time
from util import evaluate_embedding

from collections import OrderedDict
from ddpg import DDPG
from copy import deepcopy

def train_ICL_RL(data, old_data, new_data, encoder, args, loss0):
    # RL_agent
    old_agent = DDPG(args.batch_size, 1, args)
    old_step = 0
    old_observation = [loss0]
    old_agent.reset(old_observation[-args.batch_size:]+[0]*(args.batch_size-len(old_observation)))
    new_agent = DDPG(args.batch_size, 1, args)
    new_step = 0
    new_observation = [loss0]
    new_agent.reset(new_observation[-args.batch_size:]+[0]*(args.batch_size-len(new_observation)))
    
        
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
    
    fast_weights = OrderedDict(encoder.named_parameters())
    
    old_rl = 0
    pre_old_rl = 0
    pre_old_rl_count = 0
    old_done = False
    new_rl = 0
    pre_new_rl = 0
    pre_new_rl_count = 0
    new_done = False
    
    for epoch in range(args.epochs):
        loss_all = 0
        
        # train
        encoder.train()        
        start_time = time.time()
        for old_batch in old_loader:
            # select and execute an action (lr)
            if not old_done:
                if old_step <= args.warmup:
                    action = old_agent.random_action()
                else:
                    action = old_agent.select_action(old_observation[-args.batch_size:]+[0]*(args.batch_size-len(old_observation)))[0]
                pre_old_rl = old_rl
                old_lr = action[0]
                if pre_old_rl == old_rl:
                    pre_old_rl_count += 1
                    if pre_old_rl_count == len(old_loader):
                        old_done = True
                else:
                    pre_old_rl_count = 0

            old_batch, old_batch_aug = aug(old_batch, np.random.randint(3)) 
            
            old_out = encoder(old_batch, fast_weights)
            old_out_aug = encoder(old_batch_aug, fast_weights)
            
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
                           
            new_out = encoder(new_batch, fast_weights)
            loss = loss_cal_ICL_old(old_out, old_out_aug, new_out, args.alpha)
            loss_all += loss.item()  # * data.num_graphs         
            gradients = torch.autograd.grad(loss, fast_weights.values())
            fast_weights = OrderedDict((name, param - old_lr * grad) for ((name, param), grad) in zip(fast_weights.items(), gradients))

            if not old_done:
                # response with next_observation, reward, terminate_info
                old_observation2 = old_observation + [loss.item()]
                reward = old_observation[-1] - loss.item()
                # agent observe and update policy
                old_agent.observe(reward, old_observation2[-args.batch_size:]+[0]*(args.batch_size-len(old_observation2)), old_done)
                if old_step > args.warmup:
                    old_agent.update_policy()            

                # update 
                old_step += 1
                old_observation = deepcopy(old_observation2)
        
        for new_batch in new_loader:
            # select and execute an action (lr)
            if not new_done:
                if new_step <= args.warmup:
                    action = new_agent.random_action()
                else:
                    action = new_agent.select_action(new_observation[-args.batch_size:]+[0]*(args.batch_size-len(new_observation)))[0]
                new_lr = action[0]
                pre_new_rl = new_rl
                new_lr = action[0]
                if pre_new_rl == new_rl:
                    pre_new_rl_count += 1
                    if pre_new_rl_count == new_loader:
                        new_done = True
                else:
                    pre_new_rl_count = 0

            if args.mode == 'graph':
                batch_size = new_batch.num_graphs
            elif args.mode == 'cv':
                batch_size = len(new_batch[0][0])
            old_num = int((1 - args.alpha) * (batch_size-1))
            new_num = batch_size - 1 - old_num
            
            new_batch, new_batch_aug = aug(new_batch, np.random.randint(3))
            new_out = encoder(new_batch, fast_weights)
            new_out_aug = encoder(new_batch_aug, fast_weights)
            
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
                    
                old_out = encoder(old_batch, fast_weights)
            
            loss = loss_cal_ICL_new(new_out, new_out_aug, old_out, new_num)                        
            loss_all += loss.item() #* data.num_graphs
            gradients = torch.autograd.grad(loss, fast_weights.values())
            fast_weights = OrderedDict((name, param - new_lr * grad)for ((name, param), grad) in zip(fast_weights.items(), gradients))

            if not new_done:
                # response with next_observation, reward, terminate_info
                new_observation2 = new_observation + [loss.item()]
                reward = new_observation[-1] - loss.item()
                # agent observe and update policy
                new_agent.observe(reward, new_observation2[-args.batch_size:]+[0]*(args.batch_size-len(new_observation2)), new_done)
                if new_step > args.warmup:
                    new_agent.update_policy()            

                # update 
                new_step += 1
                new_observation = deepcopy(new_observation2)
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
