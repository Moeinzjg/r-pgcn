import torch
import torch.nn as nn
from .. import utils
from .GCN import GCN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PolyGNN(nn.Module):
    def __init__(self,
                 feature_dim,
                 state_dim=128,
                 n_adj=4,
                 feature_grid_size=None,
                 coarse_to_fine_steps=3):

        super(PolyGNN, self).__init__()

        self.feature_dim = feature_dim
        self.state_dim = state_dim
        self.n_adj = n_adj
        self.feature_grid_size = feature_grid_size
        self.coarse_to_fine_steps = coarse_to_fine_steps

        print('Building GNN Encoder')

        # The number of GCN needed
        if self.coarse_to_fine_steps > 0:
            for step in range(self.coarse_to_fine_steps):
                if step == 0:
                    self.gnn = nn.ModuleList(
                        [GCN(state_dim=self.state_dim, feature_dim=self.feature_dim + 2).to(device)])
                else:
                    self.gnn.append(GCN(state_dim=self.state_dim, feature_dim=self.feature_dim + 2).to(device))
        else:

            self.gnn = GCN(state_dim=self.state_dim, feature_dim=self.feature_dim + 2)

        # Initialize the weight for different layers
        for m in self.modules():  # how many modules
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                # m.weight.data.normal_(0.0, 0.00002)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, feature, init_polys):
        """
        pred_polys: in scale [0,1]
        """
        pred_poly = []
        adj = []
        for poly in init_polys:
            poly = poly.unsqueeze(0)
            for i in range(self.coarse_to_fine_steps):
                if i == 0:
                    component = utils.prepare_gcn_component(poly.numpy(),
                                                            [self.feature_grid_size],
                                                            poly.size()[1],
                                                            n_adj=self.n_adj)

                    poly = poly.to(device)
                    adjacent = component['adj_matrix'].to(device)
                    poly_idx = component['feature_indexs'].to(device)

                    cnn_feature = self.sampling(poly_idx, feature)
                    input_feature = torch.cat((cnn_feature, poly), 2)  # Hypothesis Graph Features

                else:
                    poly = gcn_pred_poly
                    cnn_feature = self.interpolated_sum(feature, poly, [self.feature_grid_size])
                    input_feature = torch.cat((cnn_feature, poly), 2)

                gcn_pred = self.gnn[i].forward(input_feature, adjacent)
                gcn_pred_poly = poly.to(device) + gcn_pred
            pred_poly.append(gcn_pred_poly)
            adj.append(adjacent) 
        return pred_poly, adj

    def interpolated_sum(self, cnns, coords, grids):

        X = coords[:, :, 0]
        Y = coords[:, :, 1]

        cnn_outs = []
        for i in range(len(grids)):
            grid = grids[i]

            Xs = X * grid
            X0 = torch.floor(Xs)
            X1 = X0 + 1

            Ys = Y * grid
            Y0 = torch.floor(Ys)
            Y1 = Y0 + 1

            w_00 = (X1 - Xs) * (Y1 - Ys)
            w_01 = (X1 - Xs) * (Ys - Y0)
            w_10 = (Xs - X0) * (Y1 - Ys)
            w_11 = (Xs - X0) * (Ys - Y0)

            X0 = torch.clamp(X0, 0, grid - 1)
            X1 = torch.clamp(X1, 0, grid - 1)
            Y0 = torch.clamp(Y0, 0, grid - 1)
            Y1 = torch.clamp(Y1, 0, grid - 1)

            N1_id = X0 + Y0 * grid
            N2_id = X0 + Y1 * grid
            N3_id = X1 + Y0 * grid
            N4_id = X1 + Y1 * grid

            M_00 = utils.gather_feature(N1_id, cnns)
            M_01 = utils.gather_feature(N2_id, cnns)
            M_10 = utils.gather_feature(N3_id, cnns)
            M_11 = utils.gather_feature(N4_id, cnns)
            cnn_out = w_00.unsqueeze(2) * M_00 + \
                      w_01.unsqueeze(2) * M_01 + \
                      w_10.unsqueeze(2) * M_10 + \
                      w_11.unsqueeze(2) * M_11

            cnn_outs.append(cnn_out)
        concat_features = torch.cat(cnn_outs, dim=2)  # Hypothesis Graph Features

        return concat_features

    def sampling(self, ids, features):

        cnn_out_feature = []
        for i in range(ids.size()[1]):
            this_id = ids[:, i, :]

            cnn_out = utils.gather_feature(this_id, features)
            cnn_out_feature.append(cnn_out)

        concat_features = torch.cat(cnn_out_feature, dim=2)

        return concat_features
