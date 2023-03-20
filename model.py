from cmath import cos
import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
import tensorflow as tf
import tensorflow.compat.v1 as tfv1
from reckit import randint_choice
import random as rd


class MF(nn.Module):
    def __init__(self, args, data):
        super(MF, self).__init__()
        self.n_users = data.n_users
        self.n_items = data.n_items
        self.lr = args.lr
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size
        self.decay = args.regs
        self.device = torch.device(args.cuda)
        self.saveID = args.saveID

        self.train_user_list = data.train_user_list
        self.valid_user_list = data.valid_user_list
        # = torch.tensor(data.population_list).cuda(self.device)
        self.user_pop = torch.tensor(data.user_pop_idx).type(torch.LongTensor).cuda(self.device)
        self.item_pop = torch.tensor(data.item_pop_idx).type(torch.LongTensor).cuda(self.device)
        self.user_pop_max = data.user_pop_max
        self.item_pop_max = data.item_pop_max        

        self.embed_user = nn.Embedding(self.n_users, self.emb_dim)
        self.embed_item = nn.Embedding(self.n_items, self.emb_dim)

        nn.init.xavier_normal_(self.embed_user.weight)
        nn.init.xavier_normal_(self.embed_item.weight)

    # Prediction function used when evaluation
    def predict(self, users, items=None):
        if items is None:
            items = list(range(self.n_items))

        users = self.embed_user(torch.tensor(users).cuda(self.device))
        items = torch.transpose(self.embed_item(torch.tensor(items).cuda(self.device)), 0, 1)
        rate_batch = torch.matmul(users, items)

        return rate_batch.cpu().detach().numpy()

class LGN(MF):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.Graph = data.getSparseGraph()
        self.n_layers = args.n_layers

    def compute(self):
        users_emb = self.embed_user.weight
        items_emb = self.embed_item.weight
        all_emb = torch.cat([users_emb, items_emb])

        embs = [all_emb]
        g_droped = self.Graph

        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
            
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.n_users, self.n_items])

        return users, items

    def forward(self, users, pos_items, neg_items):
        all_users, all_items = self.compute()

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)
        negEmb0 = self.embed_item(neg_items)

        pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim=1)  # users, pos_items, neg_items have the same shape
        neg_scores = torch.syum(torch.mul(users_emb, neg_emb), dim=1)

        regularizer = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2 + 0.5 * torch.norm(negEmb0) ** 2
        regularizer = regularizer / self.batch_size

        maxi = torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10)

        mf_loss = torch.negative(torch.mean(maxi))
        reg_loss = self.decay * regularizer

        return mf_loss, reg_loss

    def predict(self, users, items=None):
        if items is None:
            items = list(range(self.n_items))

        all_users, all_items = self.compute()

        users = all_users[torch.tensor(users).cuda(self.device)]
        items = torch.transpose(all_items[torch.tensor(items).cuda(self.device)], 0, 1)
        rate_batch = torch.matmul(users, items)

        return rate_batch.cpu().detach().numpy()

class LGN_plot(LGN):
    def __init__(self, args, data):
        super().__init__(args, data)
    
        self.user_pop_idx = data.user_pop_idx
        self.item_pop_idx = data.item_pop_idx

        self.n_users_pop = data.n_user_pop
        self.n_items_pop = data.n_item_pop
        self.embed_user_pop = nn.Embedding(self.n_users_pop, self.emb_dim)
        self.embed_item_pop = nn.Embedding(self.n_items_pop, self.emb_dim)
        nn.init.xavier_normal_(self.embed_user_pop.weight)
        nn.init.xavier_normal_(self.embed_item_pop.weight)

    def compute_p(self):

        users_emb = self.embed_user_pop.weight[self.user_pop_idx]
        items_emb = self.embed_item_pop.weight[self.item_pop_idx]
        all_emb = torch.cat([users_emb, items_emb])

        embs = [all_emb]
        g_droped = self.Graph

        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)

        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.n_users, self.n_items])

        return users, items

    def forward(self, users, pos_items, neg_items, users_pop, pos_items_pop, neg_items_pop):

        userEmb0_p = self.embed_user_pop(users_pop)
        posEmb0_p = self.embed_item_pop(pos_items_pop)
        negEmb0_p = self.embed_item_pop(neg_items_pop)

        all_users_p, all_items_p = self.compute_p()

        users_pop = all_users_p[users]
        pos_items_pop = all_items_p[pos_items]
        neg_items_pop = all_items_p[neg_items]

        pos_scores = torch.sum(torch.mul(users_pop, pos_items_pop), dim=1)  # users, pos_items, neg_items have the same shape
        neg_scores = torch.sum(torch.mul(users_pop, neg_items_pop), dim=1)

        regularizer = 0.5 * torch.norm(userEmb0_p) ** 2 + 0.5 * torch.norm(posEmb0_p) ** 2 + 0.5 * torch.norm(negEmb0_p) ** 2
        regularizer = regularizer / self.batch_size

        maxi = torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10)

        mf_loss = torch.negative(torch.mean(maxi))
        reg_loss = self.decay * regularizer

        return mf_loss, reg_loss

    def predict(self, users, items=None):
        if items is None:
            items = list(range(self.n_items))

        all_users_p, all_items_p = self.compute_p()

        users_pop = self.user_pop_idx[users]
        items_pop = self.item_pop_idx[items]

        users = all_users_p[torch.tensor(users_pop).cuda(self.device)]
        items = torch.transpose(all_items_p[torch.tensor(items_pop).cuda(self.device)], 0, 1)
        rate_batch = torch.matmul(users, items)

        return rate_batch.cpu().detach().numpy()

class IPS(LGN):
    def __init__(self, args, data):
        super().__init__(args, data)

    def forward(self, users, pos_items, neg_items, pos_weights):
        all_users, all_items = self.compute()

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)
        negEmb0 = self.embed_item(neg_items)

        pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim=1)  # users, pos_items, neg_items have the same shape
        neg_scores = torch.sum(torch.mul(users_emb, neg_emb), dim=1)

        regularizer = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2 + 0.5 * torch.norm(negEmb0) ** 2
        regularizer = regularizer / self.batch_size

        maxi = torch.mul(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10), pos_weights)

        mf_loss = torch.negative(torch.mean(maxi))
        reg_loss = self.decay * regularizer

        return mf_loss, reg_loss

class CausE(LGN):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.cf_pen = args.cf_pen
        self.embed_item_ctrl = nn.Embedding(self.n_items, self.emb_dim)
        nn.init.xavier_normal_(self.embed_item_ctrl.weight)
    
    
    def forward(self, users, pos_items, neg_items, all_reg, all_ctrl):

        all_users, all_items = self.compute()
        all_items=torch.cat([all_items,self.embed_item_ctrl.weight])

        users = all_users[users]
        pos_items = all_items[pos_items]
        neg_items = all_items[neg_items]
        item_embed = all_items[all_reg]
        control_embed = all_items[all_ctrl]

        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)   #users, pos_items, neg_items have the same shape
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 0.5 * torch.norm(users) ** 2 + 0.5 * torch.norm(pos_items) ** 2 + 0.5 * torch.norm(neg_items) ** 2
        regularizer = regularizer/self.batch_size

        maxi = torch.log(torch.sigmoid(pos_scores - neg_scores)+1e-10)

        mf_loss = torch.negative(torch.mean(maxi))
        reg_loss = self.decay * regularizer

        cf_loss = torch.sqrt(torch.sum(torch.square(torch.subtract(F.normalize(item_embed,p=2,dim=0), F.normalize(control_embed,p=2,dim=0)))))
        cf_loss = cf_loss * self.cf_pen #/ self.batch_size

        return mf_loss, reg_loss, cf_loss

class MACR(LGN):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.alpha = args.alpha
        self.beta = args.beta
        self.w = nn.Embedding(self.emb_dim, 1)
        self.w_user = nn.Embedding(self.emb_dim, 1)
        nn.init.xavier_normal_(self.w.weight)
        nn.init.xavier_normal_(self.w_user.weight)

        self.pos_item_scores = torch.empty((self.batch_size, 1))
        self.neg_item_scores = torch.empty((self.batch_size, 1))
        self.user_scores = torch.empty((self.batch_size, 1))

        self.rubi_c = args.c * torch.ones([1]).cuda(self.device)

    def forward(self, users, pos_items, neg_items):
        # Original scores
        all_users, all_items = self.compute()

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)
        negEmb0 = self.embed_item(neg_items)

        pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim=1)
        neg_scores = torch.sum(torch.mul(users_emb, neg_emb), dim=1)


        # Item module and User module
        self.pos_item_scores =torch.matmul(pos_emb, self.w.weight)
        self.neg_item_scores = torch.matmul(neg_emb, self.w.weight)
        self.user_scores = torch.matmul(users_emb, self.w_user.weight)

        # fusion
        # [batch_size,] [batch_size, 1] -> [batch_size, batch_size] * [batch_size, 1]
        # [batch_size * (bs-1)]
        pos_scores = pos_scores * torch.sigmoid(self.pos_item_scores) * torch.sigmoid(self.user_scores)
        neg_scores = neg_scores * torch.sigmoid(self.neg_item_scores) * torch.sigmoid(self.user_scores)
        #pos_scores = torch.mean(pos_scores) * torch.squeeze(torch.sigmoid(self.pos_item_scores)) * torch.squeeze(torch.sigmoid(self.user_scores))
        #neg_scores = torch.mean(neg_scores) * torch.squeeze(torch.sigmoid(self.neg_item_scores)) * torch.squeeze(torch.sigmoid(self.user_scores))

        # loss
        mf_loss_ori = torch.mean(torch.negative(torch.log(torch.sigmoid(pos_scores) + 1e-10)) + torch.negative(
            torch.log(1 - torch.sigmoid(neg_scores) + 1e-10)))

        mf_loss_item = torch.mean(
            torch.negative(torch.log(torch.sigmoid(self.pos_item_scores) + 1e-10)) + torch.negative(
                torch.log(1 - torch.sigmoid(self.neg_item_scores) + 1e-10)))

        mf_loss_user = torch.mean(torch.negative(torch.log(torch.sigmoid(self.user_scores) + 1e-10)) + torch.negative(
            torch.log(1 - torch.sigmoid(self.user_scores) + 1e-10)))

        mf_loss = mf_loss_ori + self.alpha * mf_loss_item + self.beta * mf_loss_user

        regularizer = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2 + 0.5 * torch.norm(negEmb0) ** 2
        regularizer = regularizer / self.batch_size
        reg_loss = self.decay * regularizer

        return mf_loss, reg_loss

    def predict(self, users, items=None):
        if items is None:
            items = list(range(self.n_items))

        all_users, all_items = self.compute()

        users = all_users[torch.tensor(users).cuda(self.device)]
        items = torch.transpose(all_items[torch.tensor(items).cuda(self.device)], 0, 1)

        rate_batch = torch.matmul(users, items)

        item_scores = torch.matmul(torch.transpose(items,0,1), self.w.weight)
        user_scores = torch.matmul(users, self.w_user.weight)

        rubi_rating_both = (rate_batch - self.rubi_c) * (torch.sigmoid(user_scores)) * torch.transpose(torch.sigmoid(item_scores),0,1)

        return rubi_rating_both.cpu().detach().numpy()

class IPSMF(LGN):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.rweight=args.rweight

    
    def get_correlation_loss(self,y_true, y_pred):
        x = y_true
        y = y_pred
        mx = torch.mean(x)
        my = torch.mean(y)
        xm, ym = x-mx, y-my
        r_num = torch.sum(torch.mul(xm,ym))
        r_den = torch.sqrt(torch.mul(torch.sum(torch.square(xm)), torch.sum(torch.square(ym))))
        #print(r_den)
        r = r_num / (r_den+1e-5)
        r =torch.square(torch.clamp(r,-1,1))
        return r

    def forward(self, users, pos_items, neg_items, pop_weight):

        all_users, all_items = self.compute()

        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)
        negEmb0 = self.embed_item(neg_items)

        users = all_users[users]
        pos_items = all_items[pos_items]
        neg_items = all_items[neg_items]

        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)  # users, pos_items, neg_items have the same shape
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 0.5 * torch.norm(users) ** 2 + 0.5 * torch.norm(pos_items) ** 2 + 0.5 * torch.norm(neg_items) ** 2
        regularizer = regularizer / self.batch_size

        bpr=torch.sigmoid(pos_scores - neg_scores)

        maxi = torch.log(bpr)

        mf_loss = torch.negative(torch.mean(maxi))
        reg_loss = self.decay * regularizer

        mf_loss = (1-self.rweight)*(mf_loss+reg_loss)

        cor_loss=self.rweight*self.get_correlation_loss(pop_weight,bpr)

        return mf_loss, cor_loss

class INFONCE(LGN):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.tau = args.tau
        self.neg_sample =  args.neg_sample if args.neg_sample!=-1 else self.batch_size-1

    def forward(self, users, pos_items, neg_items):

        all_users, all_items = self.compute()

        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)
        negEmb0 = self.embed_item(neg_items)

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        users_emb = F.normalize(users_emb, dim = -1)
        pos_emb = F.normalize(pos_emb, dim = -1)
        neg_emb = F.normalize(neg_emb, dim = -1)
        
        pos_ratings = torch.sum(users_emb*pos_emb, dim = -1)
        neg_ratings = torch.matmul(torch.unsqueeze(users_emb, 1), 
                                       neg_emb.permute(0, 2, 1)).squeeze(dim=1)
        ratings = torch.cat([pos_ratings[:, None], neg_ratings], dim=1)

        #分子
        numerator = torch.exp(pos_ratings / self.tau)
        #分母
        denominator = torch.sum(torch.exp(ratings / self.tau), dim = 1)
        ssm_loss = torch.mean(torch.negative(torch.log(numerator/denominator)))

        regularizer = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2 + 0.5 ** torch.norm(negEmb0)
        regularizer = regularizer / self.batch_size
        reg_loss = self.decay * regularizer

        return ssm_loss, reg_loss

class INFONCE_batch(LGN):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.tau = args.tau
        self.neg_sample =  args.neg_sample if args.neg_sample!=-1 else self.batch_size-1

    def forward(self, users, pos_items):

        all_users, all_items = self.compute()

        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]

        users_emb = F.normalize(users_emb, dim=1)
        pos_emb = F.normalize(pos_emb, dim=1)
        
        ratings = torch.matmul(users_emb, torch.transpose(pos_emb, 0, 1))
        ratings_diag = torch.diag(ratings)
        
        #分子
        numerator = torch.exp(ratings_diag / self.tau)
        #分母
        denominator = torch.sum(torch.exp(ratings / self.tau), dim = 1)
        ssm_loss = torch.mean(torch.negative(torch.log(numerator/denominator)))


        regularizer = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2
        regularizer = regularizer
        reg_loss = self.decay * regularizer

        return ssm_loss, reg_loss

class BC_LOSS(LGN):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.tau1 = args.tau1
        self.tau2 = args.tau2
        self.w_lambda = args.w_lambda
        self.neg_sample =  args.neg_sample if args.neg_sample!=-1 else self.batch_size-1
        self.n_users_pop=data.n_user_pop
        self.n_items_pop=data.n_item_pop
        self.embed_user_pop = nn.Embedding(self.n_users_pop, self.emb_dim)
        self.embed_item_pop = nn.Embedding(self.n_items_pop, self.emb_dim)
        nn.init.xavier_normal_(self.embed_user_pop.weight)
        nn.init.xavier_normal_(self.embed_item_pop.weight)
    
    def forward(self, users, pos_items, neg_items, users_pop, pos_items_pop, neg_items_pop):

        # popularity branch
        users_pop_emb = self.embed_user_pop(users_pop)
        pos_pop_emb = self.embed_item_pop(pos_items_pop)
        neg_pop_emb = self.embed_item_pop(neg_items_pop)

        pos_ratings_margin = torch.sum(users_pop_emb * pos_pop_emb, dim = -1)

        users_pop_emb = F.normalize(users_pop_emb, dim = -1)
        pos_pop_emb = F.normalize(pos_pop_emb, dim = -1)
        neg_pop_emb = F.normalize(neg_pop_emb, dim = -1)

        pos_ratings = torch.sum(users_pop_emb * pos_pop_emb, dim = -1)
        neg_ratings = torch.matmul(torch.unsqueeze(users_pop_emb, 1), 
                                       neg_pop_emb.permute(0, 2, 1)).squeeze(dim=1)
        ratings = torch.cat([pos_ratings[:, None], neg_ratings], dim=1)

        numerator = torch.exp(pos_ratings / self.tau2)
        denominator = torch.sum(torch.exp(ratings / self.tau2), dim = 1)
        loss2 = self.w_lambda * torch.mean(torch.negative(torch.log(numerator/denominator)))

        # main branch
        all_users, all_items = self.compute()

        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)
        negEmb0 = self.embed_item(neg_items)

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        users_emb = F.normalize(users_emb, dim = -1)
        pos_emb = F.normalize(pos_emb, dim = -1)
        neg_emb = F.normalize(neg_emb, dim = -1)
        
        pos_ratings = torch.sum(users_emb*pos_emb, dim = -1)
        pos_ratings = torch.cos(torch.arccos(torch.clamp(pos_ratings,-1+1e-7,1-1e-7))+(1-torch.sigmoid(pos_ratings_margin)))
        neg_ratings = torch.matmul(torch.unsqueeze(users_emb, 1), 
                                       neg_emb.permute(0, 2, 1)).squeeze(dim=1)
        ratings = torch.cat([pos_ratings[:, None], neg_ratings], dim=1)

        numerator = torch.exp(pos_ratings / self.tau1)
        denominator = torch.sum(torch.exp(ratings / self.tau1), dim = 1)
        
        loss1 = (1-self.w_lambda) * torch.mean(torch.negative(torch.log(numerator/denominator)))

        # reg loss
        regularizer1 = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2 + \
                        0.5 * torch.norm(negEmb0) ** 2
        regularizer1 = regularizer1/self.batch_size

        regularizer2= 0.5 * torch.norm(users_pop_emb) ** 2 + 0.5 * torch.norm(pos_pop_emb) ** 2 + \
                        0.5 * torch.norm(neg_pop_emb) ** 2
        regularizer2  = regularizer2/self.batch_size
        reg_loss = self.decay * (regularizer1+regularizer2)

        reg_loss_freeze=self.decay * (regularizer2)
        reg_loss_norm=self.decay * (regularizer1)

        return loss1, loss2, reg_loss, reg_loss_freeze, reg_loss_norm

    def freeze_pop(self):
        self.embed_user_pop.requires_grad_(False)
        self.embed_item_pop.requires_grad_(False)

class BC_LOSS_batch(LGN):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.tau1 = args.tau1
        self.tau2 = args.tau2
        self.w_lambda = args.w_lambda
        self.neg_sample =  args.neg_sample if args.neg_sample!=-1 else self.batch_size-1
        self.n_users_pop=data.n_user_pop
        self.n_items_pop=data.n_item_pop
        self.embed_user_pop = nn.Embedding(self.n_users_pop, self.emb_dim)
        self.embed_item_pop = nn.Embedding(self.n_items_pop, self.emb_dim)
        nn.init.xavier_normal_(self.embed_user_pop.weight)
        nn.init.xavier_normal_(self.embed_item_pop.weight)
    
    def forward(self, users, pos_items, users_pop, pos_items_pop):

        # popularity branch
        users_pop_emb = self.embed_user_pop(users_pop)
        pos_pop_emb = self.embed_item_pop(pos_items_pop)

        pos_ratings_margin = torch.sum(users_pop_emb * pos_pop_emb, dim = -1)

        users_pop_emb = F.normalize(users_pop_emb, dim = -1)
        pos_pop_emb = F.normalize(pos_pop_emb, dim = -1)

        ratings = torch.matmul(users_pop_emb, torch.transpose(pos_pop_emb, 0, 1))
        ratings_diag = torch.diag(ratings)

        numerator = torch.exp(ratings_diag / self.tau2)
        denominator = torch.sum(torch.exp(ratings / self.tau2), dim = 1)
        loss2 = self.w_lambda * torch.mean(torch.negative(torch.log(numerator/denominator)))

        # main branch
        all_users, all_items = self.compute()

        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]

        users_emb = F.normalize(users_emb, dim=1)
        pos_emb = F.normalize(pos_emb, dim=1)
        
        ratings = torch.matmul(users_emb, torch.transpose(pos_emb, 0, 1))
        ratings_diag = torch.diag(ratings)

        ratings_diag = torch.cos(torch.arccos(torch.clamp(ratings_diag,-1+1e-7,1-1e-7))+\
                                (1-torch.sigmoid(pos_ratings_margin)))
        
        numerator = torch.exp(ratings_diag / self.tau1)
        denominator = torch.sum(torch.exp(ratings / self.tau1), dim = 1)
        loss1 = (1-self.w_lambda) * torch.mean(torch.negative(torch.log(numerator/denominator)))

           # reg loss
        regularizer1 = 0.5 * torch.norm(userEmb0) ** 2 + self.batch_size * 0.5 * torch.norm(posEmb0) ** 2
        regularizer1 = regularizer1/self.batch_size

        regularizer2= 0.5 * torch.norm(users_pop_emb) ** 2 + self.batch_size * 0.5 * torch.norm(pos_pop_emb) ** 2
        regularizer2  = regularizer2/self.batch_size
        reg_loss = self.decay * (regularizer1+regularizer2)

        reg_loss_freeze=self.decay * (regularizer2)
        reg_loss_norm=self.decay * (regularizer1)

        return loss1, loss2, reg_loss, reg_loss_freeze, reg_loss_norm

    def freeze_pop(self):
        self.embed_user_pop.requires_grad_(False)
        self.embed_item_pop.requires_grad_(False)

class SimpleX(LGN):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.w_neg = args.w_neg
        self.margin = args.neg_margin
        self.neg_sample =  args.neg_sample if args.neg_sample!=-1 else self.batch_size-1
    
    def forward(self, users, pos_items, neg_items):
        all_users, all_items = self.compute()
        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)
        negEmb0 = self.embed_item(neg_items)

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        users_emb = F.normalize(users_emb, dim = -1)
        pos_emb = F.normalize(pos_emb, dim = -1)
        neg_emb = F.normalize(neg_emb, dim = -1)
        
        pos_ratings = torch.sum(users_emb*pos_emb, dim = -1)
        neg_ratings = torch.matmul(torch.unsqueeze(users_emb, 1), 
                                       neg_emb.permute(0, 2, 1)).squeeze(dim=1)
        pos_margin_loss = 1 - pos_ratings
        neg_margin_loss = torch.mean(torch.clamp(neg_ratings - self.margin, 0, 1), dim = -1)
        
        mf_loss = torch.mean(pos_margin_loss + self.w_neg * neg_margin_loss)

        regularizer = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2 + 0.5 * torch.norm(negEmb0) ** 2
        regularizer = regularizer / self.batch_size
        reg_loss = self.decay * regularizer

        return mf_loss, reg_loss

class SimpleX_batch(LGN):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.w_neg = args.w_neg
        self.margin = args.neg_margin
        self.neg_sample =  self.batch_size-1
    
    def forward(self, users, pos_items):

        all_users, all_items = self.compute()

        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]

        users_emb = F.normalize(users_emb, dim=1)
        pos_emb = F.normalize(pos_emb, dim=1)
        
        ratings = torch.matmul(users_emb, torch.transpose(pos_emb, 0, 1))
        ratings_diag = torch.diag(ratings)
        diag_mask = torch.ones_like(ratings_diag).cuda(self.device)-torch.eye(self.batch_size).cuda(self.device)
        
        pos_margin_loss = 1 - ratings_diag
        neg_margin_loss = torch.sum(torch.clamp(ratings - self.margin, 0, 1) * diag_mask, dim = -1) / self.neg_sample

        mf_loss = torch.mean(pos_margin_loss + self.w_neg * neg_margin_loss)

        regularizer = 0.5 * torch.norm(userEmb0) ** 2 + self.batch_size * 0.5 * torch.norm(posEmb0) ** 2 
        regularizer = regularizer / self.batch_size
        reg_loss = self.decay * regularizer

        return mf_loss, reg_loss

class DEBIAS_BPR(LGN):
    def __init__(self, args, data):
        super().__init__(args, data)
        
        self.lambda1 = args.lambda1
        self.lambda2 = args.lambda2
        self.lambda3 = args.lambda3
        self.n_factors = int(args.n_factors)
        self.distype = args.distype
        self.user_pop_idx = data.user_pop_idx
        self.item_pop_idx = data.item_pop_idx
        self.need_distance = args.need_distance
        self.kernel = args.kernel

        self.n_users_pop = data.n_user_pop
        self.n_items_pop = data.n_item_pop
        self.embed_user_pop = nn.Embedding(self.n_users_pop, self.emb_dim)
        self.embed_item_pop = nn.Embedding(self.n_items_pop, self.emb_dim)
        nn.init.xavier_normal_(self.embed_user_pop.weight)
        nn.init.xavier_normal_(self.embed_item_pop.weight)

    def compute_p(self):

        users_emb = self.embed_user_pop.weight[self.user_pop_idx]
        items_emb = self.embed_item_pop.weight[self.item_pop_idx]
        all_emb = torch.cat([users_emb, items_emb])

        embs = [all_emb]
        g_droped = self.Graph

        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)

        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.n_users, self.n_items])

        return users, items

    def create_cor_loss(self, cor_u_embeddings, cor_i_embeddings):
        cor_loss = torch.zeros(1).cuda(self.device)
       
        ui_embeddings = torch.cat([cor_u_embeddings, cor_i_embeddings],0)
        # TODO
        ui_factor_embeddings = torch.split(ui_embeddings, self.n_factors, 1)

        for i in range(0, self.n_factors-1):
            x = ui_factor_embeddings[i]
            y = ui_factor_embeddings[i+1]
            cor_loss += self._create_distance_correlation(x, y)

        cor_loss /= ((self.n_factors + 1.0) * self.n_factors/2)

        return cor_loss
    
    def _create_distance_correlation(self, X1, X2):

        def _create_centered_distance(X):
            '''
                Used to calculate the distance matrix of N samples
            '''
            # calculate the pairwise distance of X
            # .... A with the size of [batch_size, embed_size/n_factors]
            # .... D with the size of [batch_size, batch_size]
            # X = tf.math.l2_normalize(XX, axis=1)
            r = torch.sum(torch.square(X),1,keepdims=True)
            w = torch.bmm(X.unsqueeze(1), X.unsqueeze(-1)).squeeze(-1)
            D = torch.sqrt(torch.maximum(r - 2 * w + r.t(), torch.tensor(0.0)) + 1e-8)

            # # calculate the centered distance of X
            # # .... D with the size of [batch_size, batch_size]
            D = D - torch.mean(D,dim=0,keepdims=True)-torch.mean(D,dim=1,keepdims=True) \
                + torch.mean(D)

            return D
        
        def _create_distance_covariance(D1,D2):
            #calculate distance covariance between D1 and D2
            n_samples = D1.shape[0]
            dcov = torch.sqrt(torch.maximum(torch.sum(D1 * D2) / (n_samples * n_samples), torch.tensor(0.0)) + 1e-8)
            # dcov = torch.sqrt(torch.maximum(torch.sum(D1 * D2)) / n_samples)
            return dcov
        
        D1 = _create_centered_distance(X1)
        D2 = _create_centered_distance(X2)

        dcov_12 = _create_distance_covariance(D1,D2)
        dcov_11 = _create_distance_covariance(D1,D1)
        dcov_22 = _create_distance_covariance(D2,D2)
        
        #calculate the distance correlation
        dcor = dcov_12 / (torch.sqrt(torch.maximum(dcov_11 * dcov_22, torch.tensor(0.0))) + 1e-10)


        #return tf.reduce_sum(D1) + tf.reduce_sum(D2)
        return dcor

    def distance_loss(self, users, pos_items, users_pop, pos_items_pop):
        if self.distype == 'l1':
            return self.l_norm(users, pos_items, users_pop, pos_items_pop, 1)
        elif self.distype == 'l2':
            return self.l_norm(users, pos_items, users_pop, pos_items_pop, 2)
        elif self.distype == 'dcor':
            return self.dcor_loss(users, pos_items, users_pop, pos_items_pop)
        elif self.distype == 'mmd':
            return self.mmd_loss(users, users_pop) + self.mmd_loss(pos_items, pos_items_pop)
            print("Error distance")

    def l_norm(self, users, pos_items, users_pop, pos_items_pop, p_value):
        
        user_loss = torch.norm(( users - users_pop ), p = p_value)
        item_loss = torch.norm(( pos_items - pos_items_pop ), p = p_value)
        return (user_loss + item_loss)/self.batch_size

    def dcor_loss(self, users, pos_items, users_pop, pos_items_pop):
        
        user_loss = self.create_cor_loss(users, users_pop)
        pos_loss = self.create_cor_loss(pos_items, pos_items_pop)
        #neg_loss = self.create_cor_loss(neg_items, neg_items_pop)

        return user_loss + pos_loss # + neg_loss

    def mmd_loss(self,x, y):
        xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
        rx = (xx.diag().unsqueeze(0).expand_as(xx))
        ry = (yy.diag().unsqueeze(0).expand_as(yy))
    
        dxx = rx.t() + rx - 2. * xx # Used for A in (1)
        dyy = ry.t() + ry - 2. * yy # Used for B in (1)
        dxy = rx.t() + ry - 2. * zz # Used for C in (1)
    
        XX, YY, XY = (torch.zeros(xx.shape).cuda(self.device),
                  torch.zeros(xx.shape).cuda(self.device),
                  torch.zeros(xx.shape).cuda(self.device))
         
        if self.kernel == "multiscale":
            
            bandwidth_range = [0.2, 0.5, 0.9, 1.3]
            for a in bandwidth_range:
                XX += a**2 * (a**2 + dxx)**-1
                YY += a**2 * (a**2 + dyy)**-1
                XY += a**2 * (a**2 + dxy)**-1
                
        if self.kernel == "rbf":
        
            bandwidth_range = [10, 15, 20, 50]
            for a in bandwidth_range:
                XX += torch.exp(-0.5*dxx/a)
                YY += torch.exp(-0.5*dyy/a)
                XY += torch.exp(-0.5*dxy/a)
        
        

        return torch.mean(XX + YY - 2. * XY)
        
    def bpr_loss(self, users, pos_items, neg_items, userEmb0, posEmb0, negEmb0, batch_size = 1024):

        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)  # users, pos_items, neg_items have the same shape
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2 + 0.5 * torch.norm(negEmb0) ** 2
        regularizer = regularizer / batch_size

        maxi = torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10)

        mf_loss = torch.negative(torch.mean(maxi))
        reg_loss = self.decay * regularizer


        return mf_loss, reg_loss 
   
    def switch_concat(self, users, pos_items, neg_items, users_pop, pos_items_pop, neg_items_pop, \
                            userEmb0, posEmb0, negEmb0, userEmb0_p, posEmb0_p, negEmb0_p):

        users_ori = torch.cat((users_pop, users), -1)
        pos_items_ori = torch.cat((pos_items_pop, pos_items), -1)
        neg_items_ori = torch.cat((neg_items, neg_items_pop), -1)
        users_ori_0 = torch.cat((userEmb0_p, userEmb0), -1)
        pos_items_ori_0 = torch.cat((posEmb0_p, posEmb0), -1)
        neg_items_ori_0 = torch.cat((negEmb0_p, negEmb0), -1)

        mf_loss, reg_loss = self.bpr_loss(users_ori, pos_items_ori, neg_items_ori, users_ori_0, pos_items_ori_0, neg_items_ori_0)

        random_order = torch.randperm(pos_items_pop.size()[0])
        pos_items_pop_new = pos_items_pop[random_order]
        neg_items_pop_new = neg_items_pop[random_order]
        pos_items_new = torch.cat((pos_items_pop_new, pos_items), -1)
        neg_items_new = torch.cat((neg_items_pop_new, neg_items), -1)
        users_pop_new = users_pop[random_order]
        users_new = torch.cat((users_pop_new, users), -1)

        users_new_0 = torch.cat((userEmb0_p[random_order], userEmb0), -1)
        pos_items_new_0 = torch.cat((posEmb0_p[random_order], posEmb0), -1)
        neg_items_new_0 = torch.cat((negEmb0_p[random_order], negEmb0), -1)

        mf_loss_new, reg_loss_new = self.bpr_loss(users_new, pos_items_new,  neg_items_new, users_new_0, pos_items_new_0, neg_items_new_0)

        return mf_loss + mf_loss_new, reg_loss + reg_loss_new
    
    def forward(self, users, pos_items, neg_items, users_pop, pos_items_pop, neg_items_pop):

        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)
        negEmb0 = self.embed_item(neg_items)

        userEmb0_p = self.embed_user_pop(users_pop)
        posEmb0_p = self.embed_item_pop(pos_items_pop)
        negEmb0_p = self.embed_item_pop(neg_items_pop)

        all_users, all_items = self.compute()
        all_users_p, all_items_p = self.compute_p()

        users_pop = all_users_p[users]
        pos_items_pop = all_items_p[pos_items]
        neg_items_pop = all_items_p[neg_items]
        
        users = all_users[users]
        pos_items = all_items[pos_items]
        neg_items = all_items[neg_items]
        
        mf_loss, reg_loss = self.bpr_loss(users, pos_items, neg_items, userEmb0, posEmb0, negEmb0)
        pop_mf_loss, pop_reg_loss =  self.bpr_loss(users_pop, pos_items_pop, neg_items_pop, userEmb0_p, posEmb0_p, negEmb0_p)
        mf_loss_new, reg_loss_new = self.switch_concat(users, pos_items, neg_items, users_pop, pos_items_pop, neg_items_pop, \
                                                        userEmb0, posEmb0, negEmb0, userEmb0_p, posEmb0_p, negEmb0_p)
            
        disc_loss = 0
        if self.need_distance == 1:
            disc_loss = self.lambda2 * self.distance_loss(users, pos_items, users_pop, pos_items_pop)
        
        return mf_loss, reg_loss, self.lambda1 * pop_mf_loss, self.lambda1 * pop_reg_loss, self.lambda3 * mf_loss_new, self.lambda3 * reg_loss_new, disc_loss

class DEBIAS(LGN):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.tau = args.tau
        self.neg_sample =  args.neg_sample if args.neg_sample!=-1 else self.batch_size-1
    
        self.lambda1 = args.lambda1
        self.lambda2 = args.lambda2
        self.lambda3 = args.lambda3
        self.n_factors = int(args.n_factors)
        self.distype = args.distype
        self.user_pop_idx = data.user_pop_idx
        self.item_pop_idx = data.item_pop_idx
        self.need_distance = args.need_distance
        self.kernel = args.kernel

        self.n_users_pop = data.n_user_pop
        self.n_items_pop = data.n_item_pop
        self.embed_user_pop = nn.Embedding(self.n_users_pop, self.emb_dim)
        self.embed_item_pop = nn.Embedding(self.n_items_pop, self.emb_dim)
        nn.init.xavier_normal_(self.embed_user_pop.weight)
        nn.init.xavier_normal_(self.embed_item_pop.weight)

    def compute_p(self):

        users_emb = self.embed_user_pop.weight[self.user_pop_idx]
        items_emb = self.embed_item_pop.weight[self.item_pop_idx]
        all_emb = torch.cat([users_emb, items_emb])

        embs = [all_emb]
        g_droped = self.Graph

        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)

        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.n_users, self.n_items])

        return users, items

    def create_cor_loss(self, cor_u_embeddings, cor_i_embeddings):
        cor_loss = torch.zeros(1).cuda(self.device)
       
        ui_embeddings = torch.cat([cor_u_embeddings, cor_i_embeddings],0)
        # TODO
        ui_factor_embeddings = torch.split(ui_embeddings, self.n_factors, 1)

        for i in range(0, self.n_factors-1):
            x = ui_factor_embeddings[i]
            y = ui_factor_embeddings[i+1]
            cor_loss += self._create_distance_correlation(x, y)

        cor_loss /= ((self.n_factors + 1.0) * self.n_factors/2)

        return cor_loss
    
    def _create_distance_correlation(self, X1, X2):

        def _create_centered_distance(X):
            '''
                Used to calculate the distance matrix of N samples
            '''
            # calculate the pairwise distance of X
            # .... A with the size of [batch_size, embed_size/n_factors]
            # .... D with the size of [batch_size, batch_size]
            # X = tf.math.l2_normalize(XX, axis=1)
            r = torch.sum(torch.square(X),1,keepdims=True)
            w = torch.bmm(X.unsqueeze(1), X.unsqueeze(-1)).squeeze(-1)
            D = torch.sqrt(torch.maximum(r - 2 * w + r.t(), torch.tensor(0.0)) + 1e-8)

            # # calculate the centered distance of X
            # # .... D with the size of [batch_size, batch_size]
            D = D - torch.mean(D,dim=0,keepdims=True)-torch.mean(D,dim=1,keepdims=True) \
                + torch.mean(D)

            return D
        
        def _create_distance_covariance(D1,D2):
            #calculate distance covariance between D1 and D2
            n_samples = D1.shape[0]
            dcov = torch.sqrt(torch.maximum(torch.sum(D1 * D2) / (n_samples * n_samples), torch.tensor(0.0)) + 1e-8)
            # dcov = torch.sqrt(torch.maximum(torch.sum(D1 * D2)) / n_samples)
            return dcov
        
        D1 = _create_centered_distance(X1)
        D2 = _create_centered_distance(X2)

        dcov_12 = _create_distance_covariance(D1,D2)
        dcov_11 = _create_distance_covariance(D1,D1)
        dcov_22 = _create_distance_covariance(D2,D2)
        
        #calculate the distance correlation
        dcor = dcov_12 / (torch.sqrt(torch.maximum(dcov_11 * dcov_22, torch.tensor(0.0))) + 1e-10)


        #return tf.reduce_sum(D1) + tf.reduce_sum(D2)
        return dcor

    def distance_loss(self, users, pos_items, users_pop, pos_items_pop):
        if self.distype == 'l1':
            return self.l_norm(users, pos_items, users_pop, pos_items_pop, 1)
        elif self.distype == 'l2':
            return self.l_norm(users, pos_items, users_pop, pos_items_pop, 2)
        elif self.distype == 'dcor':
            return self.dcor_loss(users, pos_items, users_pop, pos_items_pop)
        elif self.distype == 'mmd':
            return self.mmd_loss(users, users_pop) + self.mmd_loss(pos_items, pos_items_pop)
            print("Error distance")

    def l_norm(self, users, pos_items, users_pop, pos_items_pop, p_value):
        
        user_loss = torch.norm(( users - users_pop ), p = p_value)
        item_loss = torch.norm(( pos_items - pos_items_pop ), p = p_value)
        return (user_loss + item_loss)/self.batch_size

    def dcor_loss(self, users, pos_items, users_pop, pos_items_pop):
        
        user_loss = self.create_cor_loss(users, users_pop)
        pos_loss = self.create_cor_loss(pos_items, pos_items_pop)
        #neg_loss = self.create_cor_loss(neg_items, neg_items_pop)

        return user_loss + pos_loss # + neg_loss

    def mmd_loss(self,x, y):
        xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
        rx = (xx.diag().unsqueeze(0).expand_as(xx))
        ry = (yy.diag().unsqueeze(0).expand_as(yy))
    
        dxx = rx.t() + rx - 2. * xx # Used for A in (1)
        dyy = ry.t() + ry - 2. * yy # Used for B in (1)
        dxy = rx.t() + ry - 2. * zz # Used for C in (1)
    
        XX, YY, XY = (torch.zeros(xx.shape).cuda(self.device),
                  torch.zeros(xx.shape).cuda(self.device),
                  torch.zeros(xx.shape).cuda(self.device))
         
        if self.kernel == "multiscale":
            
            bandwidth_range = [0.2, 0.5, 0.9, 1.3]
            for a in bandwidth_range:
                XX += a**2 * (a**2 + dxx)**-1
                YY += a**2 * (a**2 + dyy)**-1
                XY += a**2 * (a**2 + dxy)**-1
                
        if self.kernel == "rbf":
        
            bandwidth_range = [10, 15, 20, 50]
            for a in bandwidth_range:
                XX += torch.exp(-0.5*dxx/a)
                YY += torch.exp(-0.5*dyy/a)
                XY += torch.exp(-0.5*dxy/a)
        
        

        return torch.mean(XX + YY - 2. * XY)
        
    def infonce_loss(self, users, pos_items, neg_items, userEmb0, posEmb0, negEmb0, batch_size = 1024):

        users_emb = F.normalize(users, dim = -1)
        pos_emb = F.normalize(pos_items, dim = -1)
        neg_emb = F.normalize(neg_items, dim = -1)
        
        pos_ratings = torch.sum(users_emb*pos_emb, dim = -1)
        neg_ratings = torch.matmul(torch.unsqueeze(users_emb, 1), neg_emb.permute(0, 2, 1)).squeeze(dim=1)
        ratings = torch.cat([pos_ratings[:, None], neg_ratings], dim=1)

 
        numerator = torch.exp(pos_ratings / self.tau)
        denominator = torch.sum(torch.exp(ratings / self.tau), dim = 1)
        ssm_loss = torch.mean(torch.negative(torch.log(numerator/denominator)))

        regularizer = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2 + 0.5 ** torch.norm(negEmb0)
        regularizer = regularizer / batch_size
        reg_loss = self.decay * regularizer

        return ssm_loss, reg_loss 
   
    def switch_concat(self, users, pos_items, neg_items, users_pop, pos_items_pop, neg_items_pop, \
                            userEmb0, posEmb0, negEmb0, userEmb0_p, posEmb0_p, negEmb0_p):

        users_ori = torch.cat((users_pop, users), -1)
        pos_items_ori = torch.cat((pos_items_pop, pos_items), -1)
        neg_items_ori = torch.cat((neg_items, neg_items_pop), -1)
        users_ori_0 = torch.cat((userEmb0_p, userEmb0), -1)
        pos_items_ori_0 = torch.cat((posEmb0_p, posEmb0), -1)
        neg_items_ori_0 = torch.cat((negEmb0_p, negEmb0), -1)

        mf_loss, reg_loss = self.infonce_loss(users_ori, pos_items_ori, neg_items_ori, users_ori_0, pos_items_ori_0, neg_items_ori_0)

        random_order = torch.randperm(pos_items_pop.size()[0])
        pos_items_pop_new = pos_items_pop[random_order]
        pos_items_new = torch.cat((pos_items_pop_new, pos_items), -1)
        users_pop_new = users_pop[random_order]
        users_new = torch.cat((users_pop_new, users), -1)

        users_new_0 = torch.cat((userEmb0_p[random_order], userEmb0), -1)
        pos_items_new_0 = torch.cat((posEmb0_p[random_order], posEmb0), -1)

        mf_loss_new, reg_loss_new = self.infonce_loss(users_new, pos_items_new,  neg_items_ori, users_new_0, pos_items_new_0, neg_items_ori_0)

        return mf_loss + mf_loss_new, reg_loss + reg_loss_new
       
    def forward(self, users, pos_items, neg_items, users_pop, pos_items_pop, neg_items_pop):

        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)
        negEmb0 = self.embed_item(neg_items)

        userEmb0_p = self.embed_user_pop(users_pop)
        posEmb0_p = self.embed_item_pop(pos_items_pop)
        negEmb0_p = self.embed_item_pop(neg_items_pop)

        all_users, all_items = self.compute()
        all_users_p, all_items_p = self.compute_p()

        users_pop = all_users_p[users]
        pos_items_pop = all_items_p[pos_items]
        neg_items_pop = all_items_p[neg_items]

        #users_pop = userEmb0_p
        #pos_items_pop = posEmb0_p
        #neg_items_pop = negEmb0_p

        users = all_users[users]
        pos_items = all_items[pos_items]
        neg_items = all_items[neg_items]
        
        mf_loss, reg_loss = self.infonce_loss(users, pos_items, neg_items, userEmb0, posEmb0, negEmb0)
        pop_mf_loss, pop_reg_loss =  self.infonce_loss(users_pop, pos_items_pop, neg_items_pop, userEmb0_p, posEmb0_p, negEmb0_p)
        mf_loss_new, reg_loss_new = self.switch_concat(users, pos_items, neg_items, users_pop, pos_items_pop, neg_items_pop, \
                                                        userEmb0, posEmb0, negEmb0, userEmb0_p, posEmb0_p, negEmb0_p)
            
        disc_loss = 0
        if self.need_distance == 1:
            disc_loss = self.lambda2 * self.distance_loss(users, pos_items, users_pop, pos_items_pop)
        
        return mf_loss, reg_loss, self.lambda1 * pop_mf_loss, self.lambda1 * pop_reg_loss, self.lambda3 * mf_loss_new, self.lambda3 * reg_loss_new, disc_loss

class DEBIAS_plot(LGN):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.tau = args.tau
        self.neg_sample =  args.neg_sample if args.neg_sample!=-1 else self.batch_size-1
    
        self.lambda1 = args.lambda1
        self.lambda2 = args.lambda2
        self.lambda3 = args.lambda3
        self.n_factors = int(args.n_factors)
        self.distype = args.distype
        self.user_pop_idx = data.user_pop_idx
        self.item_pop_idx = data.item_pop_idx
        self.need_distance = args.need_distance
        self.kernel = args.kernel

        self.n_users_pop = data.n_user_pop
        self.n_items_pop = data.n_item_pop
        self.embed_user_pop = nn.Embedding(self.n_users_pop, self.emb_dim)
        self.embed_item_pop = nn.Embedding(self.n_items_pop, self.emb_dim)
        nn.init.xavier_normal_(self.embed_user_pop.weight)
        nn.init.xavier_normal_(self.embed_item_pop.weight)

    def compute_p(self):

        users_emb = self.embed_user_pop.weight[self.user_pop_idx]
        items_emb = self.embed_item_pop.weight[self.item_pop_idx]
        all_emb = torch.cat([users_emb, items_emb])

        embs = [all_emb]
        g_droped = self.Graph

        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)

        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.n_users, self.n_items])

        return users, items
      
    def infonce_loss(self, users, pos_items, neg_items, userEmb0, posEmb0, negEmb0, batch_size = 1024):

        users_emb = F.normalize(users, dim = -1)
        pos_emb = F.normalize(pos_items, dim = -1)
        neg_emb = F.normalize(neg_items, dim = -1)
        
        pos_ratings = torch.sum(users_emb*pos_emb, dim = -1)
        neg_ratings = torch.matmul(torch.unsqueeze(users_emb, 1), neg_emb.permute(0, 2, 1)).squeeze(dim=1)
        ratings = torch.cat([pos_ratings[:, None], neg_ratings], dim=1)

 
        numerator = torch.exp(pos_ratings / self.tau)
        denominator = torch.sum(torch.exp(ratings / self.tau), dim = 1)
        ssm_loss = torch.mean(torch.negative(torch.log(numerator/denominator)))

        regularizer = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2 + 0.5 ** torch.norm(negEmb0)
        regularizer = regularizer / batch_size
        reg_loss = self.decay * regularizer

        return ssm_loss, reg_loss 
         
    def forward(self, users, pos_items, neg_items, users_pop, pos_items_pop, neg_items_pop):

        userEmb0_p = self.embed_user_pop(users_pop)
        posEmb0_p = self.embed_item_pop(pos_items_pop)
        negEmb0_p = self.embed_item_pop(neg_items_pop)

        all_users_p, all_items_p = self.compute_p()

        users_pop = all_users_p[users]
        pos_items_pop = all_items_p[pos_items]
        neg_items_pop = all_items_p[neg_items]
        
        pop_mf_loss, pop_reg_loss =  self.infonce_loss(users_pop, pos_items_pop, neg_items_pop, userEmb0_p, posEmb0_p, negEmb0_p)
           
        return pop_mf_loss, pop_reg_loss

    def predict(self, users, items=None):
        if items is None:
            items = list(range(self.n_items))

        all_users_p, all_items_p = self.compute_p()

        users_pop = self.user_pop_idx[users]
        items_pop = self.item_pop_idx[items]

        users = all_users_p[torch.tensor(users_pop).cuda(self.device)]
        items = torch.transpose(all_items_p[torch.tensor(items_pop).cuda(self.device)], 0, 1)
        rate_batch = torch.matmul(users, items)

        return rate_batch.cpu().detach().numpy()

class DEBIAS_plot_batch(DEBIAS_plot):
    def __init__(self, args, data):
        super().__init__(args, data)
   
    def infonce_loss(self, users, pos_items, userEmb0, posEmb0, batch_size = 1024):

        users_emb = F.normalize(users, dim=1)
        pos_emb = F.normalize(pos_items, dim=1)
        
        ratings = torch.matmul(users_emb, torch.transpose(pos_emb, 0, 1))
        ratings_diag = torch.diag(ratings)
        
        numerator = torch.exp(ratings_diag / self.tau)
        denominator = torch.sum(torch.exp(ratings / self.tau), dim = 1)
        ssm_loss = torch.mean(torch.negative(torch.log(numerator/denominator)))

        regularizer = 0.5 * torch.norm(userEmb0) ** 2  +  0.5 * torch.norm(posEmb0) ** 2 
        regularizer = regularizer
        reg_loss = self.decay * regularizer

        return ssm_loss, reg_loss 
        
    def forward(self, users, pos_items, users_pop, pos_items_pop):

        userEmb0_p = self.embed_user_pop(users_pop)
        posEmb0_p = self.embed_item_pop(pos_items_pop)

        all_users_p, all_items_p = self.compute_p()

        users_pop = all_users_p[users]
        pos_items_pop = all_items_p[pos_items]
        
        pop_mf_loss, pop_reg_loss =  self.infonce_loss(users_pop, pos_items_pop, userEmb0_p, posEmb0_p)
           
        return pop_mf_loss, pop_reg_loss
 
class DEBIAS_batch(DEBIAS):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.neg_sample =  self.batch_size-1
    
    def infonce_loss(self, users, pos_items, userEmb0, posEmb0, batch_size = 1024):

        users_emb = F.normalize(users, dim=1)
        pos_emb = F.normalize(pos_items, dim=1)
        
        ratings = torch.matmul(users_emb, torch.transpose(pos_emb, 0, 1))
        ratings_diag = torch.diag(ratings)
        
        numerator = torch.exp(ratings_diag / self.tau)
        denominator = torch.sum(torch.exp(ratings / self.tau), dim = 1)
        ssm_loss = torch.mean(torch.negative(torch.log(numerator/denominator)))

        regularizer = 0.5 * torch.norm(userEmb0) ** 2  +  0.5 * torch.norm(posEmb0) ** 2 
        regularizer = regularizer
        reg_loss = self.decay * regularizer

        return ssm_loss, reg_loss 
   
    def switch_concat(self, users, pos_items, users_pop, pos_items_pop, userEmb0, posEmb0, userEmb0_p, posEmb0_p):

        users_ori = torch.cat((users_pop, users), -1)
        pos_items_ori = torch.cat((pos_items_pop, pos_items), -1)
        users_ori_0 = torch.cat((userEmb0_p, userEmb0), -1)
        pos_items_ori_0 = torch.cat((posEmb0_p, posEmb0), -1)

        mf_loss, reg_loss = self.infonce_loss(users_ori, pos_items_ori, users_ori_0, pos_items_ori_0)

        random_order = torch.randperm(pos_items_pop.size()[0])
        pos_items_pop_new = pos_items_pop[random_order]
        pos_items_new = torch.cat((pos_items_pop_new, pos_items), -1)
        users_pop_new = users_pop[random_order]
        users_new = torch.cat((users_pop_new, users), -1)

        users_new_0 = torch.cat((userEmb0_p[random_order], userEmb0), -1)
        pos_items_new_0 = torch.cat((posEmb0_p[random_order], posEmb0), -1)

        mf_loss_new, reg_loss_new = self.infonce_loss(users_new, pos_items_new, users_new_0, pos_items_new_0)

        return mf_loss + mf_loss_new, reg_loss + reg_loss_new
        
        return mf_loss, reg_loss
    
    def forward(self, users, pos_items, users_pop, pos_items_pop):

        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)

        userEmb0_p = self.embed_user_pop(users_pop)
        posEmb0_p = self.embed_item_pop(pos_items_pop)

        all_users, all_items = self.compute()
        all_users_p, all_items_p = self.compute_p()

        users_pop = all_users_p[users]
        pos_items_pop = all_items_p[pos_items]
        #users_pop = userEmb0_p
        #pos_items_pop = posEmb0_p
        
        users = all_users[users]
        pos_items = all_items[pos_items]
        
        mf_loss, reg_loss = self.infonce_loss(users, pos_items, userEmb0, posEmb0)
        pop_mf_loss, pop_reg_loss =  self.infonce_loss(users_pop, pos_items_pop, userEmb0_p, posEmb0_p)
        mf_loss_new, reg_loss_new = self.switch_concat(users, pos_items, users_pop, pos_items_pop, \
                                                    userEmb0, posEmb0, userEmb0_p, posEmb0_p, )
            
        disc_loss = 0
        if self.need_distance == 1:
            disc_loss = self.lambda2 * self.distance_loss(users, pos_items, users_pop, pos_items_pop)
        
        return mf_loss, reg_loss, self.lambda1 * pop_mf_loss, self.lambda1 * pop_reg_loss, self.lambda3 * mf_loss_new, self.lambda3 * reg_loss_new, disc_loss     

class sDRO(LGN):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.tau = args.tau
        self.neg_sample =  args.neg_sample if args.neg_sample!=-1 else self.batch_size-1
        self.Graph = data.getSparseGraph()
        self.n_layers = args.n_layers
        self.n_items = data.n_items
        self.n_users = data.n_users

        # 0 niche, 1 diverse, 2 blockbuster
        self._group_labels = [0, 1, 2]
        self._group_labels_matrix = torch.reshape(torch.tensor([0, 1, 2]), (-1, 1)).cuda(self.device)
        self._num_groups = 3
        self._group_reweight_strategy = 'loss-dro'
        self._dro_temperature = args.dro_temperature
        self._streaming_group_loss_lr = args.str_lr

        self._group_weights = torch.tensor([1/3, 1/3, 1/3], requires_grad = False).cuda(self.device)
        self._group_loss = torch.tensor([0.0, 0.0, 0.0], requires_grad = False).cuda(self.device)


    def _compute_group_loss(self, sample_loss, group_identity):

        group_mask = torch.eq(group_identity, self._group_labels_matrix).type(torch.LongTensor).cuda(self.device)
        group_cnts = torch.sum(group_mask, axis=1)
        group_cnts += (group_cnts == 0).long()
        group_loss = torch.divide(torch.sum(group_mask * sample_loss, axis=1), group_cnts)

        return group_loss, group_mask


    def forward(self, users, pos_items, neg_items, users_group):

        all_users, all_items = self.compute()

        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)
        negEmb0 = self.embed_item(neg_items)

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        users_emb = F.normalize(users_emb, dim = -1)
        pos_emb = F.normalize(pos_emb, dim = -1)
        neg_emb = F.normalize(neg_emb, dim = -1)
        
        pos_ratings = torch.sum(users_emb*pos_emb, dim = -1)
        neg_ratings = torch.matmul(torch.unsqueeze(users_emb, 1), 
                                       neg_emb.permute(0, 2, 1)).squeeze(dim=1)
        ratings = torch.cat([pos_ratings[:, None], neg_ratings], dim=1)

        #分子
        numerator = torch.exp(pos_ratings / self.tau)
        #分母
        denominator = torch.sum(torch.exp(ratings / self.tau), dim = 1)
        ssm_loss = torch.negative(torch.log(numerator/denominator))

        regularizer = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2 + 0.5 ** torch.norm(negEmb0)
        regularizer = regularizer / self.batch_size
        reg_loss = self.decay * regularizer

        # sDRO part
        cur_group_loss, group_mask = self._compute_group_loss(ssm_loss, users_group)
        # Note: only update loss/metric estimations when subgroup exists in a batch.
        # group_exist_in_batch: [num_groups], bool
        group_exist_in_batch = (torch.sum(group_mask, axis=1) > 1e-16).type(torch.LongTensor).cuda(self.device)

        # Perform streaming estimation of group loss.
        stream_group_loss = (1 - group_exist_in_batch * self._streaming_group_loss_lr) \
                            * self._group_loss.data + self._streaming_group_loss_lr * cur_group_loss
        self._group_loss.data = stream_group_loss

        # update group weight
        m = nn.Softmax(dim=-1)
        new_group_weights = m(torch.log(self._group_weights.data) + self._dro_temperature * self._group_loss.data)
        self._group_weights.data = new_group_weights

        sDRO_loss = torch.sum(self._group_weights.data * cur_group_loss) * self.batch_size

        return sDRO_loss, reg_loss

class sDRO_batch(sDRO):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.neg_sample =  self.batch_size-1

    def forward(self, users, pos_items, users_group):

        all_users, all_items = self.compute()

        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]

        users_emb = F.normalize(users_emb, dim=1)
        pos_emb = F.normalize(pos_emb, dim=1)
        
        ratings = torch.matmul(users_emb, torch.transpose(pos_emb, 0, 1))
        ratings_diag = torch.diag(ratings)
        
        #分子
        numerator = torch.exp(ratings_diag / self.tau)
        #分母
        denominator = torch.sum(torch.exp(ratings / self.tau), dim = 1)
        ssm_loss = torch.negative(torch.log(numerator/denominator))

        regularizer = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2
        regularizer = regularizer
        reg_loss = self.decay * regularizer

        # sDRO part
        cur_group_loss, group_mask = self._compute_group_loss(ssm_loss, users_group)
        # Note: only update loss/metric estimations when subgroup exists in a batch.
        # group_exist_in_batch: [num_groups], bool
        group_exist_in_batch = (torch.sum(group_mask, axis=1) > 1e-16).type(torch.LongTensor).cuda(self.device)

        # Perform streaming estimation of group loss.
        stream_group_loss = (1 - group_exist_in_batch * self._streaming_group_loss_lr) \
                            * self._group_loss.data + self._streaming_group_loss_lr * cur_group_loss
        self._group_loss.data = stream_group_loss

        # update group weight
        m = nn.Softmax(dim=-1)
        new_group_weights = m(torch.log(self._group_weights.data) + self._dro_temperature * self._group_loss.data)
        self._group_weights.data = new_group_weights

        sDRO_loss = torch.sum(self._group_weights.data * cur_group_loss) * self.batch_size

        return sDRO_loss, reg_loss

class MLP(nn.Module):

    def __init__(self, input_size = 128, output_size = 64):
        super(MLP , self).__init__()
        self.layer = nn.Linear(  input_size  , output_size  )
        
    def forward(self, x):  
        scores   = self.layer(x)
        return scores

class CDAN(LGN):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.tau = args.tau
        self.neg_sample =  self.batch_size-1
        self.hidden_size = args.hidden_size
        self.lambda1 = args.lambda1
        self.lambda2 = args.lambda2
        self.bias = args.bias

        self.item_prop = MLP(self.emb_dim, self.emb_dim)
        self.item_pop = MLP(self.emb_dim, self.emb_dim)
        self.item_final = MLP(2*self.emb_dim, self.emb_dim)
        
        self.item_prop.cuda(self.device)
        self.item_pop.cuda(self.device)
        self.item_final.cuda(self.device)

        self.user_pop_idx = data.user_pop_idx
        self.item_pop_idx = data.item_pop_idx
        self.n_users_pop = data.n_user_pop
        self.n_items_pop = data.n_item_pop
        self.embed_user_pop = nn.Embedding(self.n_users_pop, self.emb_dim)
        self.embed_item_pop = nn.Embedding(self.n_items_pop, self.emb_dim)
        nn.init.xavier_normal_(self.embed_user_pop.weight)
        nn.init.xavier_normal_(self.embed_item_pop.weight)
    
    def compute_p(self):

        users_emb = self.embed_user_pop.weight[self.user_pop_idx]
        items_emb = self.embed_item_pop.weight[self.item_pop_idx]
        all_emb = torch.cat([users_emb, items_emb])

        embs = [all_emb]
        g_droped = self.Graph

        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)

        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.n_users, self.n_items])

        return users, items

    def infonce_loss(self, users, pos_items, userEmb0, posEmb0, pos_weights, flag = False, batch_size = 1024):

        users_emb = F.normalize(users, dim=1)
        pos_emb = F.normalize(pos_items, dim=1)
        
        ratings = torch.matmul(users_emb, torch.transpose(pos_emb, 0, 1))
        ratings_diag = torch.diag(ratings)
        
        numerator = torch.exp(ratings_diag / self.tau)
        denominator = torch.sum(torch.exp(ratings / self.tau), dim = 1)

        if flag :
            maxi = torch.mul(torch.log(torch.log(numerator/denominator)), pos_weights)
            ssm_loss = torch.mean(torch.negative(maxi))

        else:
            ssm_loss = torch.mean(torch.negative(torch.log(numerator/denominator)))

        regularizer = 0.5 * torch.norm(userEmb0) ** 2  +  0.5 * torch.norm(posEmb0) ** 2 
        regularizer = regularizer
        reg_loss = self.decay * regularizer

        return ssm_loss, reg_loss 
     
    def disentangle(self, pos_items, pos_items_pop):
        # get property and popularity branch of items
        item_prop = self.item_prop(pos_items)
        item_pop = self.item_pop(pos_items)

        item_prop = F.normalize(item_prop, dim=1)
        item_pop = F.normalize(item_pop, dim=1)
        pos_items_pop = F.normalize(pos_items_pop, dim=1)

        pos_sim = self.batch_size - torch.sum(torch.mul(item_pop, pos_items_pop ), dim=1)  # users, pos_items, neg_items have the same shape
        orthogonal = torch.sum(torch.square(torch.mul(item_pop, item_prop)), dim=1)

        return torch.sum(pos_sim + orthogonal)

    def forward(self, users, pos_items, users_pop, pos_items_pop, next_pos_item, pos_weights):

        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)

        userEmb0_p = self.embed_user_pop(users_pop)
        posEmb0_p = self.embed_item_pop(pos_items_pop)

        all_users, all_items = self.compute()
        all_users_p, all_items_p = self.compute_p()

        users_pop = all_users_p[users]
        pos_items_pop = all_items_p[pos_items]
        #users_pop = userEmb0_p
        #pos_items_pop = posEmb0_p
        
        users = all_users[users]
        pos_items = all_items[pos_items]

        # get property and popularity branch of items
        item_prop = self.item_prop(pos_items)
        item_pop = self.item_pop(pos_items)
        
        # unbias
        unbias_loss, unbias_reg = self.infonce_loss(users, item_prop, userEmb0, posEmb0, pos_weights)

        # bias 
        item_final = self.item_final(torch.cat((item_prop, pos_items_pop), -1))
        bias_loss, bias_reg = self.infonce_loss(users, item_final, userEmb0, posEmb0, pos_weights)
            
        # disentangled
        dis_loss = self.lambda1 * self.disentangle(pos_items, pos_items_pop)
        # long tail
        #next_item_prop = self.item_prop(all_items[next_pos_item])

        #long_tail_loss, long_tail_reg = self.lambda2 * self.infonce_loss(next_item_prop, item_prop, posEmb0, posEmb0, pos_weights, flag = True)
        
        #long_tail_loss  = self.lambda2 * (self.long_tail_loss + long_tail_reg)

        return unbias_loss, bias_loss, dis_loss, unbias_reg , bias_reg

    def predict(self, users, items =  None):

        if self.bias == 0:
            return self.predict_unbias(users, items)
        else:
            return self.predict_bias(users, items)
    
    def predict_unbias(self, users, items=None):
        if items is None:
            items = list(range(self.n_items))

        all_users, all_items = self.compute()

        users = all_users[torch.tensor(users).cuda(self.device)]

        items = all_items[torch.tensor(items).cuda(self.device)]
        items_prop = self.item_prop(items)
        items_prop = torch.transpose(items_prop, 0, 1)
        rate_batch = torch.matmul(users, items_prop)

        return rate_batch.cpu().detach().numpy()

    def predict_bias(self, users, items=None):
        if items is None:
            items = list(range(self.n_items))

        all_users, all_items = self.compute()
        all_users_pop, all_items_pop = self.compute_p()

        users = all_users[torch.tensor(users).cuda(self.device)]
        items_pop = all_items_pop[torch.tensor(items).cuda(self.device)]

        items = all_items[torch.tensor(items).cuda(self.device)]
        items_prop = self.item_prop(items)
        
        items_final = self.item_final(torch.cat((items_prop, items_pop), -1))
        items_final = torch.transpose(items_final, 0, 1)
        
        rate_batch = torch.matmul(users, items_final)

        return rate_batch.cpu().detach().numpy()
    
class CDAN_MF(CDAN):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.neg_sample = args.neg_sample 
    
    def infonce_loss(self, users, pos_items, neg_items, pos_weights, flag = False, batch_size = 1024):

        users_emb = F.normalize(users, dim = -1)
        pos_emb = F.normalize(pos_items, dim = -1)
        neg_emb = F.normalize(neg_items, dim = -1)
        
        pos_ratings = torch.sum(users_emb*pos_emb, dim = -1)
        neg_ratings = torch.matmul(torch.unsqueeze(users_emb, 1), neg_emb.permute(0, 2, 1)).squeeze(dim=1)
        ratings = torch.cat([pos_ratings[:, None], neg_ratings], dim=1)
 
        numerator = torch.exp(pos_ratings / self.tau)
        denominator = torch.sum(torch.exp(ratings / self.tau), dim = 1)

        if flag:
            maxi = torch.mul(torch.log(torch.log(numerator/denominator)), pos_weights)
            ssm_loss = torch.mean(torch.negative(maxi))

        else:
            ssm_loss = torch.mean(torch.negative(torch.log(numerator/denominator)))


        regularizer = 0.5 * torch.norm(users_emb) ** 2 + 0.5 * torch.norm(pos_emb) ** 2 + 0.5 ** torch.norm(neg_emb)
        regularizer = regularizer / batch_size
        reg_loss = self.decay * regularizer

        return ssm_loss, reg_loss 
 
    def forward(self, users, pos_items, neg_items, users_pop, pos_items_pop, neg_items_pop, next_pos_item, pos_weights):

        
        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)
        negEmb0 = self.embed_item(neg_items)

        userEmb0_p = self.embed_user_pop(users_pop)
        posEmb0_p = self.embed_item_pop(pos_items_pop)
        negEmb0_p = self.embed_item_pop(neg_items_pop)

        all_users, all_items = self.compute()
        all_users_p, all_items_p = self.compute_p()

        users_pop = all_users_p[users]
        pos_items_pop = all_users_p[pos_items]
        neg_items_pop = all_users_p[neg_items]

        users_pop = userEmb0_p
        pos_items_pop = posEmb0_p 
        neg_items_pop = negEmb0_p

        users = all_users[users]
        pos_items = all_items[pos_items]
        neg_items = all_items[neg_items]

        # get property and popularity branch of items
        item_prop = self.item_prop(pos_items)
        neg_item_prop = self.item_prop(neg_items)
        item_pop = self.item_pop(pos_items)
        neg_item_pop_mlp = self.item_pop(neg_items)
        
        # unbias
        unbias_loss, unbias_reg = self.infonce_loss(users, item_prop, neg_item_prop, pos_weights)

        # bias 
        item_final = self.item_final(torch.cat((item_prop, pos_items_pop), -1))
        neg_item_final = self.item_final(torch.cat((neg_item_prop, neg_items_pop), -1))
        bias_loss, bias_reg = self.infonce_loss(users, item_final, neg_item_final, pos_weights)
        
        # disentangled
        dis_loss = self.lambda1 * self.disentangle(pos_items, pos_items_pop)
        # long tail
        #next_item_prop = self.item_prop(self.embed_item(next_pos_item))

        #long_tail_loss, long_tail_reg =  self.infonce_loss(next_item_prop, item_prop, neg_item_prop, pos_weights, flag = True)
        #long_tail_loss = self.lambda2 * (long_tail_loss + long_tail_reg)

        return unbias_loss, bias_loss, dis_loss, unbias_reg, bias_reg
   
class CDAN_test(CDAN):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.neg_sample = args.neg_sample 

        self.item_prop = MLP(self.emb_dim, self.emb_dim)
        
        self.item_prop.cuda(self.device)
    
    
    def infonce_loss(self, users, pos_items, neg_items, pos_weights, flag = False, batch_size = 1024):

        users_emb = F.normalize(users, dim = -1)
        pos_emb = F.normalize(pos_items, dim = -1)
        neg_emb = F.normalize(neg_items, dim = -1)
        
        pos_ratings = torch.sum(users_emb*pos_emb, dim = -1)
        neg_ratings = torch.matmul(torch.unsqueeze(users_emb, 1), neg_emb.permute(0, 2, 1)).squeeze(dim=1)
        ratings = torch.cat([pos_ratings[:, None], neg_ratings], dim=1)
 
        numerator = torch.exp(pos_ratings / self.tau)
        denominator = torch.sum(torch.exp(ratings / self.tau), dim = 1)

        if flag:
            maxi = torch.mul(torch.log(torch.log(numerator/denominator)), pos_weights)
            ssm_loss = torch.mean(torch.negative(maxi))

        else:
            ssm_loss = torch.mean(torch.negative(torch.log(numerator/denominator)))


        regularizer = 0.5 * torch.norm(users_emb) ** 2 + 0.5 * torch.norm(pos_emb) ** 2 + 0.5 ** torch.norm(neg_emb)
        regularizer = regularizer / batch_size
        reg_loss = self.decay * regularizer

        return ssm_loss, reg_loss 
 
    def forward(self, users, pos_items, neg_items, users_pop, pos_items_pop, neg_items_pop, pos_weights):

        users = self.embed_user(users)
        pos_items = self.embed_item(pos_items)
        neg_items = self.embed_item(neg_items)

        #users_pop = self.embed_user_pop(users_pop)
        #pos_items_pop = self.embed_item_pop(pos_items_pop)
        neg_items_pop = self.embed_item_pop(neg_items_pop)

        # get property and popularity branch of items
        item_prop = self.item_prop(pos_items)
        neg_item_prop = self.item_prop(neg_items)
        #item_pop = self.item_pop(pos_items)
        #neg_item_pop_mlp = self.item_pop(neg_items)
        
        # unbias
        unbias_loss, unbias_reg = self.infonce_loss(users, item_prop, neg_item_prop, pos_weights)

        return unbias_loss, unbias_reg

    # UNBIAS
    
    def predict(self, users, items=None):
        if items is None:
            items = list(range(self.n_items))

        users = self.embed_user(torch.tensor(users).cuda(self.device))
        items = self.embed_item(torch.tensor(items).cuda(self.device))

        items_prop = self.item_prop(items)
        items_prop = torch.transpose(items_prop, 0, 1)
        rate_batch = torch.matmul(users, items_prop)

        return rate_batch.cpu().detach().numpy()
    '''
    # BIAS

    def predict(self, users, items=None):
        if items is None:
            items = list(range(self.n_items))

        all_users, all_items = self.compute()
        all_users_pop, all_items_pop = self.compute_p()

        users = all_users[torch.tensor(users).cuda(self.device)]
        items_pop = all_items_pop[torch.tensor(items).cuda(self.device)]

        items = all_items[torch.tensor(items).cuda(self.device)]
        items_prop = self.item_prop(items)
        
        items_final = self.item_final(torch.cat((items_prop, items_pop), -1))
        items_final = torch.transpose(items_final, 0, 1)
        
        rate_batch = torch.matmul(users, items_final)

        return rate_batch.cpu().detach().numpy()
    '''

class DEBIAS_ablation(DEBIAS):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.aug_type = args.aug_type
        self.head = data.head
        self.mid = data.mid
        self.tail = data.tail
        self.len_head = data.len_head
        self.len_mid = data.len_mid
        self.len_tail = data.len_tail

        self.head_user = data.head_user
        self.mid_user = data.mid_user
        self.tail_user = data.tail_user
        self.len_head_user = data.len_head_user
        self.len_mid_user = data.len_mid_user
        self.len_tail_user = data.len_tail_user

        self.idx = data.idx
        self.idx_user = data.idx_user

        self.head_pt = 0
        self.mid_pt = 0
        self.tail_pt = 0
        self.head_user_pt = 0
        self.mid_user_pt = 0
        self.tail_user_pt = 0

        self.no_head = data.no_head
        self.no_mid = data.no_mid
        self.no_tail = data.no_tail
        self.no_head_user = data.no_head_user
        self.no_mid_user = data.no_mid_user
        self.no_tail_user = data.no_tail_user

        self.current_group = []
        self.current_group_user = []

        self.mid_head = data.mid_head
        self.mid_tail = data.mid_tail
        self.head_tail = data.head_tail

        self.mid_head_u = data.mid_head_u
        self.mid_tail_u = data.mid_tail_u
        self.head_tail_u = data.head_tail_u

        #self.head_items, self.mid_items, self.tail_items = 0, 0, 0
        #self.head_users, self.mid_users, self.tail_users = 0, 0, 0
        
    def infonce_loss(self, users, pos_items, neg_items, batch_size = 1024):

        users_emb = F.normalize(users, dim = -1)
        pos_emb = F.normalize(pos_items, dim = -1)
        neg_emb = F.normalize(neg_items, dim = -1)
        
        pos_ratings = torch.sum(users_emb*pos_emb, dim = -1)
        neg_ratings = torch.matmul(torch.unsqueeze(users_emb, 1), neg_emb.permute(0, 2, 1)).squeeze(dim=1)
        ratings = torch.cat([pos_ratings[:, None], neg_ratings], dim=1)
 
        numerator = torch.exp(pos_ratings / self.tau)
        denominator = torch.sum(torch.exp(ratings / self.tau), dim = 1)
        ssm_loss = torch.mean(torch.negative(torch.log(numerator/denominator)))

        regularizer = 0.5 * torch.norm(users_emb) ** 2 + 0.5 * torch.norm(pos_emb) ** 2 + 0.5 ** torch.norm(neg_emb)
        regularizer = regularizer / batch_size
        reg_loss = self.decay * regularizer

        return ssm_loss, reg_loss 
   
    def switch_concat(self, users, pos_items, neg_items, users_pop, pos_items_pop, neg_items_pop):

        users_ori = torch.cat((users_pop, users), -1)
        pos_items_ori = torch.cat((pos_items_pop, pos_items), -1)
        neg_items_ori = torch.cat((neg_items, neg_items_pop), -1)

        mf_loss, reg_loss = self.infonce_loss(users_ori, pos_items_ori, neg_items_ori)

        # TODO let them all append popular pop go
        '''
        if self.aug_type == 'head':
            random_order = randint_choice(self.len_head, size=self.batch_size)
            pos_items_pop_new1 = self.head[random_order]

            random_order = randint_choice(self.len_mid, size=self.batch_size)
            pos_items_pop_new2 = self.mid[random_order]

            random_order = randint_choice(self.len_tail, size=self.batch_size)
            pos_items_pop_new3 = self.tail[random_order]

            pos_items_pop_new = np.concatenate((pos_items_pop_new1, pos_items_pop_new2))
            pos_items_pop_new = np.concatenate((pos_items_pop_new, pos_items_pop_new3))  

            random_order_user = randint_choice(self.len_head_user, size=self.batch_size)
            users_pop_new1 = self.head_user[random_order_user]

            random_order_user = randint_choice(self.len_mid_user, size=self.batch_size)
            users_pop_new2 = self.mid_user[random_order_user]

            random_order_user = randint_choice(self.len_tail_user, size=self.batch_size)
            users_pop_new3 = self.tail_user[random_order_user]

            users_pop_new = np.concatenate((users_pop_new1, users_pop_new2))
            users_pop_new = np.concatenate((users_pop_new, users_pop_new3))         
        '''

        if self.aug_type == 'head':
            random_order = randint_choice(self.len_head, size=self.batch_size)
            pos_items_pop_new = self.head[random_order]

            random_order_user = randint_choice(self.len_head_user, size=self.batch_size)
            users_pop_new = self.head_user[random_order_user]         

        elif self.aug_type  == 'tail':
            random_order = randint_choice(self.len_tail, size=self.batch_size)
            pos_items_pop_new = self.tail[random_order]

            random_order_user = randint_choice(self.len_tail_user, size=self.batch_size)
            users_pop_new = self.tail_user[random_order_user]
        
        elif self.aug_type == 'other':
            rad = rd.random()
            pos_items_pop_new = []
            users_pop_new = []

            for group_idx in self.current_group:
                # head组，在mid和tail中挑一个
                if (group_idx == 2 and rad < self.mid_tail) or (group_idx == 0 and rad < self.mid_head):
                    pos_items_pop_new.append(self.mid[self.mid_pt])
                    self.mid_pt += 1
                    self.mid_pt = self.mid_pt % self.len_mid

                if (group_idx == 2 and rad >= self.mid_tail) or (group_idx == 1 and rad >= self.head_tail):
                    pos_items_pop_new.append(self.tail[self.tail_pt])
                    self.tail_pt += 1
                    self.tail_pt = self.tail_pt % self.len_tail

                if (group_idx == 1 and rad < self.head_tail) or (group_idx == 0 and rad >= self.mid_head):
                    pos_items_pop_new.append(self.head[self.head_pt])
                    self.head_pt += 1
                    self.head_pt = self.head_pt % self.len_head

            for group_idx in self.current_group_user:
                # head组，在mid和tail中挑一个
                if (group_idx == 2 and rad < self.mid_tail_u) or (group_idx == 0 and rad < self.mid_head_u):
                    users_pop_new.append(self.mid_user[self.mid_user_pt])
                    self.mid_user_pt += 1
                    self.mid_user_pt = self.mid_user_pt % self.len_mid_user

                if (group_idx == 2 and rad >= self.mid_tail_u) or (group_idx == 1 and rad >= self.head_tail_u):
                    users_pop_new.append(self.tail_user[self.tail_user_pt])
                    self.tail_user_pt += 1
                    self.tail_user_pt = self.tail_user_pt % self.len_tail_user

                if (group_idx == 1 and rad < self.head_tail_u) or (group_idx == 0 and rad >= self.mid_head_u):
                    users_pop_new.append(self.head_user[self.head_user_pt])
                    self.head_user_pt += 1
                    self.head_user_pt = self.head_user_pt % self.len_head_user


        pos_items_pop_new = np.array(pos_items_pop_new)
        users_pop_new = np.array(users_pop_new)

        pos_items_pop_new = self.embed_item_pop.weight[self.item_pop_idx[pos_items_pop_new]]
        users_pop_new = self.embed_user_pop.weight[self.user_pop_idx[users_pop_new]]

        # doubles
        '''
        pos_items_temp = torch.cat((pos_items, pos_items), 0)
        pos_items = torch.cat((pos_items_temp, pos_items), 0)

        users_temp = torch.cat((users, users), 0)
        users = torch.cat((users_temp, users), 0)

        neg_items_ori_temp = torch.cat((neg_items_ori, neg_items_ori), 0)
        neg_items_ori = torch.cat((neg_items_ori_temp, neg_items_ori), 0)
        '''

        pos_items_new = torch.cat((pos_items_pop_new, pos_items), -1)
        users_new = torch.cat((users_pop_new, users), -1)

        mf_loss_new, reg_loss_new = self.infonce_loss(users_new, pos_items_new,  neg_items_ori, 2*self.batch_size)

        return mf_loss + mf_loss_new, reg_loss + reg_loss_new
       
    def forward(self, users, pos_items, neg_items, users_pop, pos_items_pop, neg_items_pop):

        if self.aug_type == 'other':
            self.current_group = self.idx[pos_items.cpu().detach().numpy()]
            self.current_group_user = self.idx_user[users.cpu().detach().numpy()]


        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)
        negEmb0 = self.embed_item(neg_items)

        userEmb0_p = self.embed_user_pop(users_pop)
        posEmb0_p = self.embed_item_pop(pos_items_pop)
        negEmb0_p = self.embed_item_pop(neg_items_pop)

        all_users, all_items = self.compute()
        all_users_p, all_items_p = self.compute_p()

        users_pop = all_users_p[users]
        pos_items_pop = all_items_p[pos_items]
        neg_items_pop = all_items_p[neg_items]

        users = all_users[users]
        pos_items = all_items[pos_items]
        neg_items = all_items[neg_items]
        
        mf_loss, reg_loss = self.infonce_loss(users, pos_items, neg_items)
        pop_mf_loss, pop_reg_loss =  self.infonce_loss(users_pop, pos_items_pop, neg_items_pop)
        mf_loss_new, reg_loss_new = self.switch_concat(users, pos_items, neg_items, users_pop, pos_items_pop, neg_items_pop)
            
        disc_loss = 0
        if self.need_distance == 1:
            disc_loss = self.lambda2 * self.distance_loss(users, pos_items, users_pop, pos_items_pop)
        
        return mf_loss, reg_loss, self.lambda1 * pop_mf_loss, self.lambda1 * pop_reg_loss, self.lambda3 * mf_loss_new, self.lambda3 * reg_loss_new, disc_loss


'''
class BPRMF(LGN):
    def __init__(self, args, data):
        super().__init__(args, data)

    def forward(self, users, pos_items, neg_items):

        all_users, all_items = self.compute()

        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)
        negEmb0 = self.embed_item(neg_items)

        users = all_users[users]
        pos_items = all_items[pos_items]
        neg_items = all_items[neg_items]

        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)  # users, pos_items, neg_items have the same shape
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 0.5 * torch.norm(users) ** 2 + 0.5 * torch.norm(pos_items) ** 2 + 0.5 * torch.norm(neg_items) ** 2
        regularizer = regularizer / self.batch_size

        maxi = torch.log(torch.sigmoid(pos_scores - neg_scores))

        mf_loss = torch.negative(torch.mean(maxi))
        reg_loss = self.decay * regularizer

        return mf_loss, reg_loss

class BCEMF(MF):
    def __init__(self, args, data):
        super().__init__(args, data)

    def forward(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)  # users, pos_items, neg_items have the same shape
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 0.5 * torch.norm(users) ** 2 + 0.5 * torch.norm(pos_items) ** 2 + 0.5 * torch.norm(neg_items) ** 2
        regularizer = regularizer / self.batch_size

        mf_loss = torch.mean(torch.negative(torch.log(torch.sigmoid(pos_scores) + 1e-9))
                             + torch.negative(torch.log(1 - torch.sigmoid(neg_scores) + 1e-9)))
        reg_loss = self.decay * regularizer

        return mf_loss, reg_loss
'''
    
