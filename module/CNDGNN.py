import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter
import math
import torch.optim as optim
import copy
import time
from sklearn.metrics import accuracy_score as ACC
from utils import row_normalized_adjacency, sparse_mx_to_torch_sparse_tensor
from dgl.nn import GraphConv
from torch.nn import init
import yaml

from torch_geometric.nn import MessagePassing, APPNP
from torch_geometric.nn.conv.gcn_conv import gcn_norm

class CNDGNN(nn.Module):

    def __init__(
            self,
            in_features: int,
            num_nodes:int,
            class_num: int,
            device,
            args,
        ) -> None:
        super().__init__()
        #------------- Parameters ----------------
        self.dropout = args.dropout
        self.device = device
        self.epochs = args.epochs
        self.patience = args.patience
        self.lr = args.lr
        self.l2_coef = args.l2_coef
        self.class_num = class_num
        self.structure_info = args.structure_info
        self.lambda_ = args.lambda_
        self.n_layers = args.layers

        args_dict = args.__dict__
        args_dict['config'] = f'./config/GPRGNN.yaml'
        with open(args.config, 'r', encoding='utf-8') as f:  
            config = yaml.safe_load(f)
        dataset_config = config[args.dataset]

        self.n_layers_pretrain=dataset_config.get('nlayers')
        self.lr_pretrain = dataset_config.get('lr')
        #self.n_hidden_pretrain= dataset_config.get('nhidden')
        self.alpha_pretrain= dataset_config.get('alpha')
        self.l2_coef_pretrain=dataset_config.get('l2_coef')
        self.dprate_pretrain=dataset_config.get('dprate')
        self.patience_pretrain =dataset_config.get('patience')
        self.dropout_pretrain = 0.5

        self.lin1 = nn.Linear(in_features, 64)
        self.lin2 = nn.Linear(64, class_num)
        # if ppnp_GPRGNN == 'PPNP':
        #     self.prop1 = APPNP(args.nlayers, args.alpha)
        # elif ppnp_GPRGNN == 'GPR_prop':
        self.prop1 = GPR_prop(self.n_layers_pretrain, self.alpha_pretrain, 'PPR', None)



        self.alpha = nn.Parameter(torch.ones(num_nodes))  # shape: [N]
        self.beta = nn.Parameter(torch.ones(num_nodes) * 5.0)  # shape: [N]


        if args.dataset in ['wikics','pubmed','photo', 'flickr']:
            self.class_feature=int(args.nhidden)
        else:
            self.class_feature=int(args.nhidden/2)
        self.feature_size=args.nhidden

        self.w = Parameter(torch.FloatTensor(self.class_feature,self.class_feature)).to(self.device)
        self.tran=nn.Linear(class_num, self.class_feature).to(self.device)
        stdv = 1.0 / math.sqrt(self.w.size(1))
        self.w.data.uniform_(-stdv, stdv)

        self.tran_feature=nn.Linear(in_features, self.feature_size).to(self.device)
        #self.w_feature = nn.Parameter(torch.zeros((num_nodes, num_nodes), dtype=torch.float32)) 
        self.row_weight = nn.Parameter(torch.zeros(num_nodes))  # [N, 1]

        #---------------- Model -------------------

        self.ccp_layers = nn.ModuleList()
        for _ in range(args.layers):
            self.ccp_layers.append(CCPLayer(args.nhidden, args.nhidden, class_num))
        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(in_features, args.nhidden))
        self.fc_layers.append(nn.Linear(args.nhidden * (args.layers + 1), class_num))
        self.act_fn = nn.ReLU()
        """
        #---------------- Layer -------------------
        layers_1 = []
        pre_dim_1 = in_features
        for i in range(self.n_layers_pretrain):
            if i == self.n_layers_pretrain-1:
                now_dim = self.class_num
            else:
                now_dim = self.n_hidden_pretrain
            layers_1.append(GraphConv(pre_dim_1, now_dim))
            pre_dim_1 = now_dim

        self.model_1 = nn.ModuleList(layers_1)
        # ---------------- Parameter Initialization -------------------
        # Initialize weights for all layers using Kaiming Uniform
        for layer in self.model_1:
            # Kaiming Uniform initialization (recommended for ReLU activation)
            init.kaiming_uniform_(layer.weight, mode='fan_out', nonlinearity='relu')
            # If the layer has a bias, initialize it to 0
            if layer.bias is not None:
                init.zeros_(layer.bias)
        """
    def RowWiseKLWeightFunction(self, A_1,A):
        """
        A_1: Tensor of shape [N, N], containing KL divergence values (non-negative)
        Return: Tensor of shape [N, N], mapped into [-alpha_i, +alpha_i] per row
        """
        N = A_1.shape[0]
        
        row_indices = torch.arange(N).view(N, 1).to(A_1.device)  # shape [N,1]
        alpha_row = self.alpha[row_indices]  # shape [N,1]
        beta_row = self.beta[row_indices]    # shape [N,1]

        weights = alpha_row * (2 * torch.exp(-beta_row * A_1) - 1)  # shape [N,N]
        weights = weights*A
        return weights
    """
    #for dataset：pubmed, photo, and wikics, use this GetWeight function; otherwise, use the next GetWeight function
    def GetWeight(self,normalized_pattern_two,A,h):
        x=self.tran(normalized_pattern_two)
        h=self.tran_feature(h)
        #d_1=torch.sqrt(torch.tensor(x.size(-1),dtype=torch.float32))
        #d_2=torch.sqrt(torch.tensor(h.size(-1),dtype=torch.float32))
        B_1=torch.tanh(torch.matmul(torch.matmul(x,self.w),x.t()))
        B_2=torch.tanh(torch.matmul(h,h.t()))

        weight=torch.sigmoid(self.row_weight)
        B=weight*B_1+(1-weight)*B_2

        B=A*B

        return B
     
    """ 
    def GetWeight(self, normalized_pattern_two, A, h):
        x = self.tran(normalized_pattern_two)
        h = self.tran_feature(h)

        d_1=torch.sqrt(torch.tensor(x.size(-1),dtype=torch.float32))
        d_2=torch.sqrt(torch.tensor(h.size(-1),dtype=torch.float32))

        weight = torch.sigmoid(self.row_weight)
        A=A.to_sparse()

        edge_index = A._indices()
        src, dst = edge_index[0], edge_index[1] 

        x_w = torch.matmul(x, self.w)
        #B_1_edge = torch.tanh((x_w[src] * x[dst]).sum(dim=1)/d_1)
        #B_2_edge = torch.tanh((h[src] * h[dst]).sum(dim=1)/d_2)

        B_1_edge = torch.tanh((x_w[src] * x[dst]).sum(dim=1))
        B_2_edge = torch.tanh((h[src] * h[dst]).sum(dim=1))

        weight_src = weight[src]
        B_edge = weight_src * B_1_edge + (1 - weight_src) * B_2_edge

        B_sparse = torch.sparse_coo_tensor(edge_index, B_edge, A.shape)

        return B_sparse
    
    def fit(self, graph, label, mixed_labels, train_mask, val_mask, test_mask,
            normalized_pattern_two, A,
            lr=None, l2_coef=None, epochs=None, dropout=None, layers=None,
            variant=None, structure_info=None, patience=None, lambda_=None,
            normalize=None, undirected=None, nhidden=None, **kwargs):

        self.lr = lr if lr is not None else self.lr
        self.l2_coef = l2_coef if l2_coef is not None else self.l2_coef
        self.epochs = epochs if epochs is not None else self.epochs
        self.patience = patience if patience is not None else self.patience
        self.dropout = dropout if dropout is not None else self.dropout
        self.n_layers = layers if layers is not None else self.n_layers
        self.nhidden = nhidden if nhidden is not None else self.nhidden
        self.variant = variant if variant is not None else self.variant
        self.structure_info = structure_info if structure_info is not None else self.structure_info
        self.lambda_ = lambda_ if lambda_ is not None else self.lambda_
        self.normalize = normalize if normalize is not None else self.normalize
        self.undirected = undirected if undirected is not None else self.undirected

        print(f"\nTraining with parameters:")
        print(f"lr: {self.lr}, l2_coef: {self.l2_coef}, epochs: {self.epochs}")
        print(f"dropout: {self.dropout}, layers: {self.n_layers}, nhidden: {self.nhidden}")
        print(f"variant: {self.variant}, structure_info: {self.structure_info}")
        print(f"lambda_: {self.lambda_}, patience: {self.patience}\n")


        graph = graph.to(self.device)
        labels = label.to(self.device)
        mixed_labels = mixed_labels.to(self.device)
        self.train_mask = train_mask.to(self.device)
        self.valid_mask = val_mask.to(self.device)
        self.test_mask = test_mask.to(self.device)
        A=A.to(self.device)
        normalized_pattern_two=normalized_pattern_two.to(self.device)
        self.to(self.device)
        graph = graph.remove_self_loop()
        X = graph.ndata["feat"]
        n_nodes, _ = X.shape
        init_adj = graph.adj().to_dense().cpu()
        adj_norm, deg = row_normalized_adjacency(init_adj, return_deg=True) #normalized adj
        deg = torch.tensor(deg, dtype=torch.float).to(self.device)
        adj = sparse_mx_to_torch_sparse_tensor(adj_norm).to(self.device)#normalized adj


        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2_coef)
        loss_fn = torch.nn.CrossEntropyLoss()
        best_epoch = 0
        best_acc = 0.
        cnt = 0
        best_state_dict = None

        t_start = time.time()
        for epoch in range(self.epochs):
            self.train()
            optimizer.zero_grad()
            Z, C = self.forward(X, adj, normalized_pattern_two,A)#normalized adj
            loss = loss_fn(C[self.train_mask], labels[self.train_mask].to(self.device))
            loss.backward()
            optimizer.step()
        
            [train_acc, valid_acc, test_acc] = self.test(X, adj, normalized_pattern_two,A,labels, [self.train_mask, self.valid_mask, self.test_mask])
            if valid_acc > best_acc:
                cnt = 0
                best_acc = valid_acc
                best_epoch = epoch
                best_state_dict = copy.deepcopy(self.state_dict())
                print(f'\nEpoch:{epoch}, Loss:{loss.item()}')
                print(f'train acc: {train_acc:.3f}, valid acc: {valid_acc:.3f}, test acc: {test_acc:.3f}')
            else:
                cnt += 1
                if cnt == self.patience:
                    print(f"Early Stopping! Best Epoch: {best_epoch}, best val acc: {best_acc}")
                    break
        t_finish = time.time()
        print("\n10 epoch cost: {:.4f}s\n".format((t_finish - t_start)/(epoch+1)*10))
        self.load_state_dict(best_state_dict)
        self.best_epoch = best_epoch
    


    def kl_divergence(p, q, epsilon=1e-8):

        p = torch.clamp(p, min=epsilon)
        q = torch.clamp(q, min=epsilon)
        return torch.sum(p * torch.log(p / q))

    def forward(self, X, adj, normalized_pattern_two,A):
        
        H_list = []
        X = F.dropout(X, self.dropout, training=self.training)
        H = self.act_fn(self.fc_layers[0](X))
        H_list.append(H)
        #B=A.float()
        B=self.GetWeight(normalized_pattern_two,A,X)
        
        for i, layer in enumerate(self.ccp_layers):
            # H = F.dropout(H, self.dropout, training=self.training)
            H = self.act_fn(layer(H, adj, B))
            H_list.append(H)
        Z = torch.cat(H_list, dim=1)
        Z = F.dropout(Z, self.dropout, training=self.training)
        C = self.fc_layers[-1](Z)
        C = F.softmax(C, dim=1)
        return Z, C

    def test(self, X, adj, A_1,A,labels, index_list):
        self.eval()
        with torch.no_grad():
            Z, C = self.forward(X, adj, A_1,A)
            y_pred = torch.argmax(C, dim=1)
        acc_list = []
        for index in index_list:
            acc_list.append(ACC(labels[index].cpu(), y_pred[index].cpu()))
        return acc_list

    def predict(self, graph,A_1,A):
        self.eval()
        graph = graph.to(self.device)
        graph = graph.remove_self_loop()
        X = graph.ndata['feat']
        init_adj = graph.adj().to_dense().cpu()
        adj_norm, deg = row_normalized_adjacency(init_adj, return_deg=True) #normalized adj
        deg = torch.tensor(deg, dtype=torch.float).to(self.device)
        adj = sparse_mx_to_torch_sparse_tensor(adj_norm).to(self.device)#normalized adj

        A=A.to(self.device)
        A_1=A_1.to(self.device)

        with torch.no_grad():
            Z, C = self.forward(X, adj, A_1,A)
            C = C[:X.shape[0]]
            y_pred = torch.argmax(C, dim=1)

        return y_pred.cpu(), C.cpu(), Z[:X.shape[0]].cpu()

    def pretrain(self, graph, labels, train_mask, val_mask, test_mask):
        # model init
        graph = graph.to(self.device)
        labels = labels.to(self.device)
        self.train_mask = train_mask.to(self.device)
        self.valid_mask = val_mask.to(self.device)
        self.test_mask = test_mask.to(self.device)
        self.to(self.device)
        X = graph.ndata["feat"]
        n_nodes, _ = X.shape


        edge_index = torch.stack(graph.edges(), dim=0)
        edge_index = edge_index.to(self.device)

        best_epoch = 0
        best_acc = 0.
        cnt = 0
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr_pretrain, weight_decay=self.l2_coef_pretrain)
        best_state_dict = None
        print("------------pretrain------------")
        for epoch in range(self.epochs):
            self.train()
            optimizer.zero_grad()
            output = self.forward_pretrain(X, edge_index)
            loss = F.nll_loss(output[train_mask], labels[train_mask].to(self.device))
            loss.backward()
            optimizer.step()

            [train_acc, valid_acc, test_acc] = self.test_pretrain(X, edge_index, labels, [self.train_mask, self.valid_mask, self.test_mask])

            if valid_acc > best_acc:
                cnt = 0
                best_acc = valid_acc
                best_epoch = epoch
                best_state_dict = copy.deepcopy(self.state_dict())
                print(f'\nEpoch:{epoch}, Loss:{loss.item()}')
                print(f'train acc: {train_acc:.3f} valid acc: {valid_acc:.3f}, test acc: {test_acc:.3f}')

            else:
                cnt += 1
                if cnt == self.patience:
                    print(f"Early Stopping! Best Epoch: {best_epoch}, best val acc: {best_acc}")
                    break
        self.load_state_dict(best_state_dict)
        self.best_epoch = best_epoch

    def forward_pretrain(self, x, edge_index, return_Z=False):
        x = F.dropout(x, p=self.dropout_pretrain, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_pretrain, training=self.training)
        x = self.lin2(x)

        if self.dprate_pretrain == 0.0:
            x = self.prop1(x, edge_index)
        else:
            x = F.dropout(x, p=self.dprate_pretrain, training=self.training)
            x = self.prop1(x, edge_index)
        
        if return_Z:
            return x, F.log_softmax(x, dim=1)
        return F.log_softmax(x, dim=1)
    
    def test_pretrain(self, X, edge_index, labels, index_list):
        self.eval()
        with torch.no_grad():
            C = self.forward_pretrain(X, edge_index)
            y_pred = torch.argmax(C, dim=1)
        acc_list = []
        for index in index_list:
            acc_list.append(ACC(labels[index].cpu(), y_pred[index].cpu()))
        return acc_list

    def predict_pretrain(self, graph):
        self.eval()
        graph = graph.to(self.device)

        self.to(self.device)
        X = graph.ndata["feat"]
        n_nodes, _ = X.shape

        edge_index = torch.stack(graph.edges(), dim=0)
        edge_index = edge_index.to(self.device)
        with torch.no_grad():
            Z, C = self.forward_pretrain(X, edge_index, return_Z=True)
            y_pred = torch.argmax(C, dim=1)

        return y_pred.cpu(), C.cpu(), Z.cpu()
    

class CCPLayer(nn.Module):

    def __init__(self, in_features, out_features, class_num):
        super(CCPLayer, self).__init__() 
        self.class_num = class_num
        self.w_0 = Parameter(torch.FloatTensor(in_features, out_features))
        self.w_1 = Parameter(torch.FloatTensor(in_features, out_features))
        self.w_2 = Parameter(torch.FloatTensor(in_features, out_features))
        self.alpha_learner = nn.Sequential(
            nn.Linear(out_features * 3, 3),
            nn.Sigmoid(),
            nn.Linear(3,3),
            nn.Softmax(dim=1),
        )
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.w_0.size(1))
        self.w_0.data.uniform_(-stdv, stdv)
        self.w_1.data.uniform_(-stdv, stdv)
        self.w_2.data.uniform_(-stdv, stdv)

    def forward(self, H, adj, A_1):
        n = H.shape[0]

        Z_0 = F.relu(torch.matmul(H, self.w_0))
        Z_1 = F.relu(torch.spmm(adj, torch.matmul(H, self.w_1)))
        Z_2 = F.tanh(torch.spmm(A_1, torch.matmul(H, self.w_2)))

        alpha = self.alpha_learner(torch.cat([Z_0, Z_1, Z_2], dim=1))
        Z = alpha[:, 0].view(-1, 1) * Z_0 + alpha[:, 1].view(-1, 1) * Z_1 + alpha[:, 2].view(-1, 1) * Z_2

        return Z

class GPR_prop(MessagePassing):
    '''
    propagation class for GPR_GNN
    '''

    def __init__(self, K, alpha, Init, Gamma=None, bias=True, **kwargs):
        super(GPR_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha

        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            # SGC-like
            TEMP = 0.0*np.ones(K+1)
            TEMP[alpha] = 1.0
        elif Init == 'PPR':
            # PPR-like
            TEMP = alpha*(1-alpha)**np.arange(K+1)
            TEMP[-1] = (1-alpha)**K
        elif Init == 'NPPR':
            # Negative PPR
            TEMP = (alpha)**np.arange(K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'Random':
            # Random
            bound = np.sqrt(3/(K+1))
            TEMP = np.random.uniform(-bound, bound, K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'WS':
            # Specify Gamma
            TEMP = Gamma

        self.temp = Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        for k in range(self.K+1):
            self.temp.data[k] = self.alpha*(1-self.alpha)**k
        self.temp.data[-1] = (1-self.alpha)**self.K

    def forward(self, x, edge_index, edge_weight=None):

        edge_index, norm = gcn_norm(
            edge_index, edge_weight, num_nodes=x.shape[0], dtype=x.dtype)

        hidden = x*(self.temp[0])
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            gamma = self.temp[k+1]
            hidden = hidden + gamma*x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)