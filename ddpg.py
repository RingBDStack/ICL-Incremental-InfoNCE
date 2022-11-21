
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

from RL_model import (Actor, Critic)
from memory import SequentialMemory
from random_process import OrnsteinUhlenbeckProcess
from util import *


criterion = nn.MSELoss()

class DDPG(object):
    def __init__(self, nb_states, nb_actions, args):
        
        if args.seed > 0:
            self.seed(args.seed)

        self.nb_states = nb_states
        self.nb_actions= nb_actions
        
        # Create Actor and Critic Network
        self.hidden = args.hidden
        self.actor = Actor(self.nb_states//args.batch_size, self.nb_actions, args.hidden)
        self.actor_target = Actor(self.nb_states//args.batch_size, self.nb_actions, args.hidden)
        self.actor_optim  = Adam(self.actor.parameters(), lr=1e-5)

        self.critic = Critic(self.nb_states, self.nb_actions, args.hidden)
        self.critic_target = Critic(self.nb_states, self.nb_actions, args.hidden)
        self.critic_optim = Adam(self.critic.parameters(), lr=1e-5)

        hard_update(self.actor_target, self.actor) # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)
        
        # Create replay buffer
        self.memory = SequentialMemory(limit=args.rmsize, window_length=args.window_length)
        self.random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=args.ou_theta, mu=args.ou_mu, sigma=args.ou_sigma)

        # Hyper-parameters
        self.batch_size = args.batch_size
        self.tau = args.tau
        self.discount = args.discount
        self.depsilon = 1.0 / args.epsilon
        self.max_lr = 1e-1
        self.min_lr = 1e-6

        # 
        self.epsilon = 1.0
        self.s_t = None # Most recent state
        self.a_t = None # Most recent action
        self.is_training = True

        # 
        if USE_CUDA: self.cuda()

    def update_policy(self):
        # Sample batch
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = self.memory.sample_and_split(self.batch_size)

        # Prepare for the target q batch

        s_t = to_tensor(next_state_batch, volatile=True).unsqueeze(-1)
        h0 = Variable(torch.randn(2, s_t.shape[0], self.hidden)).cuda()
        c0 = Variable(torch.randn(2, s_t.shape[0], self.hidden)).cuda()
        next_q_values = self.critic_target([
            to_tensor(next_state_batch, volatile=True),
            self.actor_target(s_t, (h0, c0)),
        ])
        
        target_q_batch = to_tensor(reward_batch) + self.discount*to_tensor(terminal_batch.astype(np.float))*next_q_values

        # Critic update
        self.critic.zero_grad()

        q_batch = self.critic([to_tensor(state_batch), to_tensor(action_batch)])
        
        value_loss = criterion(q_batch, target_q_batch)
        value_loss.backward()
        self.critic_optim.step()

        # Actor update
        self.actor.zero_grad()

        s_t = to_tensor(state_batch, volatile=True).unsqueeze(-1)
        h0 = Variable(torch.randn(2, s_t.shape[0], self.hidden)).cuda()
        c0 = Variable(torch.randn(2, s_t.shape[0], self.hidden)).cuda()
        policy_loss = -self.critic([
            to_tensor(state_batch),
            self.actor(s_t, (h0, c0))
        ])

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        # Target update
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)


    def cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()

    def observe(self, r_t, s_t1, done):
        if self.is_training:
            self.memory.append(self.s_t, self.a_t, r_t, done)
            self.s_t = s_t1

    def random_action(self):
        action = np.random.uniform(self.min_lr, self.max_lr, self.nb_actions)
        self.a_t = action
        return action

    def select_action(self, s_t, decay_epsilon=True):
        s_t = to_tensor(np.array(s_t)).unsqueeze(0).unsqueeze(-1)
        h0 = Variable(torch.randn(2, 1, self.hidden)).cuda()
        c0 = Variable(torch.randn(2, 1, self.hidden)).cuda()
        action = to_numpy(self.actor(s_t, (h0, c0)))
        action += self.is_training*max(self.epsilon, 0)*self.random_process.sample()
        action = np.clip(action, self.min_lr, self.max_lr)
        if decay_epsilon:
            self.epsilon -= self.depsilon
        
        self.a_t = action
        return action

    def reset(self, obs):
        self.s_t = obs
        self.random_process.reset_states()
        