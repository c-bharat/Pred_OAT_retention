'''
September 2020 by Chrianna Bharat
Adapted from https://www.github.com/sebbarb/
'''

import numpy as np
import math
import pandas as pd
import torch
import torch.utils.data as utils
import torch.nn.functional as F
import torch.nn as nn
# from torch_scatter import scatter_add
# from torch_scatter.composite import scatter_softmax
from tqdm import tqdm

from torch.nn import LSTM, GRU, ConstantPad1d
from pdb import set_trace as bp
from sklearn.preprocessing import StandardScaler


class LabTransCoxTime:
    """
    Label transforms useful for CoxTime models. It can log-transform and standardize the durations.
    It also creates `map_scaled_to_orig` which is the inverse transform of the durations data,
    enabling us to set the correct time scale for predictions.
    This can be done by passing the object to the CoxTime init:
        model = CoxTime(net, labrans=labtrans)
    which gives the correct time scale of survival predictions
        surv = model.predict_surv_df(x)

    Keyword Arguments:
        log_duration {bool} -- Log-transform durations, i.e. 'log(1+x)'. (default: {False})
        with_mean {bool} -- Center the duration before scaling.
            Passed to `sklearn.preprocessing.StandardScaler` (default: {True})
        with_std {bool} -- Scale duration to unit variance.
            Passed to `sklearn.preprocessing.StandardScaler` (default: {True})
    """

    def __init__(self, log_duration=False, with_mean=True, with_std=True):
        self.log_duration = log_duration
        self.duration_scaler = StandardScaler(True, with_mean, with_std)

    @property
    def map_scaled_to_orig(self):
        """Map from transformed durations back to the original durations, i.e. inverse transform.
        Use it to e.g. set index of survival predictions:
            surv = model.predict_surv_df(x_test)
            surv.index = labtrans.map_scaled_to_orig(surv.index)
        """
        if not hasattr(self, '_inverse_duration_map'):
            raise ValueError('Need to fit the models before you can call this method')
        return self._inverse_duration_map

    def fit(self, durations, events):
        self.fit_transform(durations, events)
        return self

    def fit_transform(self, durations, events):
        train_durations = durations
        durations = durations.astype('float32')
        events = events.astype('float32')
        if self.log_duration:
            durations = np.log1p(durations)
        durations = self.duration_scaler.fit_transform(durations.reshape(-1, 1)).flatten()
        self._inverse_duration_map = {scaled: orig for orig, scaled in zip(train_durations, durations)}
        self._inverse_duration_map = np.vectorize(self._inverse_duration_map.get)
        return durations, events

    def transform(self, durations, events):
        durations = durations.astype('float32')
        events = events.astype('float32')
        if self.log_duration:
            durations = np.log1p(durations)
        durations = self.duration_scaler.transform(durations.reshape(-1, 1)).flatten()
        return durations, events

    @property
    def out_features(self):
        """Returns the number of output features that should be used in the torch model.
        This always returns 1, and is just included for api design purposes.

        Returns:
            [int] -- Number of output features.
        """
        return 1



class Attention(torch.nn.Module):
    """
    Dot-product attention module.

    Args:
      inputs: A `Tensor` with embeddings in the last dimension.
      mask: A `Tensor`. Dimensions are the same as inputs but without the embedding dimension.
        Values are 0 for 0-padding in the input and 1 elsewhere.
    Returns:
      outputs: The input `Tensor` whose embeddings in the last dimension have undergone a weighted average.
        The second-last dimension of the `Tensor` is removed.
      attention_weights: weights given to each embedding.
    """

    def __init__(self, embedding_dim):
        super(Attention, self).__init__()
        self.norm = np.sqrt(embedding_dim)
        self.context = nn.Parameter(torch.Tensor(embedding_dim))  # context vector
        self.linear_hidden = nn.Linear(embedding_dim, embedding_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.context)

    def forward(self, input, mask):
        # Hidden representation of embeddings (no change in dimensions)
        hidden = torch.tanh(self.linear_hidden(input))
        # Compute weight of each embedding
        importance = torch.sum(hidden * self.context, dim=-1) / self.norm
        importance = importance.masked_fill(mask == 0, -1e9)
        # Softmax so that weights sum up to one
        attention_weights = F.softmax(importance, dim=-1)
        # Weighted sum of embeddings
        weighted_projection = input * torch.unsqueeze(attention_weights, dim=-1)
        # Output
        output = torch.sum(weighted_projection, dim=-2)
        return output, attention_weights

class CoxPHLoss(torch.nn.Module):
    """
    Approximated Cox Proportional Hazards loss (negative partial likelihood
        with 1 control sample)

    Args:
        risk_case: The predicted risk for people who experienced the event (batch_size,)
        risk_control: The predicted risk for people who where at risk at the time
            when the corresponding case experienced the event (batch_size,)

    Returns:
        loss: Scalar loss
    """
    def __init__(self):
        super(CoxPHLoss, self).__init__()

    def forward(self, risk_case, risk_control):
        loss = (F.softplus(risk_control - risk_case)).mean()
        return loss

def sort_and_case_indices(x, time, event):
    """
    Sort data to allow for efficient sampling of people at risk.
    Time is in descending order, in case of ties non-events come first.
    In general, after sorting, if the index of A is smaller than the index of B,
    A is at risk when B experiences the event.
    To avoid sampling from ties, the column 'MAX_IDX_CONTROL' indicates the maximum
    index from which a case can be sampled.

    Args:
        x: input data
        time: time to event/censoring
        event: binary vector, 1 if the person experienced an event or 0 if censored

    Returns:
        sort_index: index to sort indices according to risk
        case_index: index to extract cases (on data sorted by sort_index!)
        max_idx_control: maximum index to sample a control for each case
    """
    # Sort
    df = pd.DataFrame({'TIME': time, 'EVENT': event.astype(bool)})
    df.sort_values(by=['TIME', 'EVENT'], ascending=[False, True], inplace=True)
    sort_index = df.index
    df.reset_index(drop=True, inplace=True)

    # Max idx for sampling controls (either earlier times or same time but no event)
    df['MAX_IDX_CONTROL'] = -1
    max_idx_control = -1
    prev_time = df.at[0, 'TIME']
    print('Computing MAX_IDX_CONTROL, time for a(nother) coffee...')
    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        if not row['EVENT']:
            max_idx_control = i
        elif (prev_time > row['TIME']):
            max_idx_control = i - 1
        df.at[i, 'MAX_IDX_CONTROL'] = max_idx_control
        prev_time = row['TIME']
    print('done')
    df_case = df[df['EVENT'] & (df['MAX_IDX_CONTROL'] >= 0)]
    case_index, max_idx_control = df_case.index, df_case['MAX_IDX_CONTROL'].values
    return sort_index, case_index, max_idx_control

def get_case_control(x_case, time_case, max_idx_control, code_case, weeks_case, diagt_case, x, code, weeks, diagt, hp):
    control_idx = (np.random.uniform(size=(x_case.shape[0],)) * max_idx_control.numpy()).astype(int)
    x_control = x[control_idx]
    x_cc = torch.cat([x_case, x_control]).to(hp.device)
    if hp.nonprop_hazards:
        time_cc = torch.unsqueeze(torch.cat([time_case, time_case]), 1).float().to(hp.device) # time is the same for cases and controls
    else:
        time_cc = None
    code_control = code[control_idx]
    code_cc = torch.cat([code_case, code_control]).to(hp.device)
    weeks_control = weeks[control_idx]
    weeks_cc = torch.cat([weeks_case, weeks_control]).to(hp.device)
    diagt_control = diagt[control_idx]
    diagt_cc = torch.cat([diagt_case, diagt_control]).to(hp.device)
    return x_cc, time_cc, code_cc, weeks_cc, diagt_cc


def trn(trn_loader, x_trn, code_trn, weeks_trn, diagt_trn, model, criterion, optimizer, hp):
    model.train()
    for batch_idx, (x_case, time_case, max_idx_control, code_case, weeks_case, diagt_case) in enumerate(tqdm(trn_loader)):
        # Get controls
        x_cc, time_cc, code_cc, weeks_cc, diagt_cc = get_case_control(x_case, time_case, max_idx_control, code_case, weeks_case, diagt_case, x_trn, code_trn, weeks_trn, diagt_trn, hp)

        # Optimise
        optimizer.zero_grad()
        risk_case, risk_control = model(x_cc, code_cc, weeks_cc, diagt_cc, time_cc).chunk(2)
        loss = criterion(risk_case, risk_control)
        loss.backward()
        optimizer.step()


def val(val_loader, x_val, code_val, weeks_val, diagt_val, model, criterion, epoch, hp):
    loss = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, (x_case, time_case, max_idx_control, code_case, weeks_case, diagt_case) in enumerate(tqdm(val_loader)):
            # Get controls
            x_cc, time_cc, code_cc, weeks_cc, diagt_cc = get_case_control(x_case, time_case, max_idx_control, code_case, weeks_case, diagt_case, x_val, code_val, weeks_val, diagt_val, hp)

            # Compute Loss
            risk_case, risk_control = model(x_cc, code_cc, weeks_cc, diagt_cc, time_cc).chunk(2)
            loss += criterion(risk_case, risk_control)

        loss_norm = loss.item()/len(val_loader.dataset)
        print('Epoch: {} Loss: {:.6f}'.format(epoch, loss_norm))
        return loss_norm

def log(model_name, concordance, brier, r2, dindex, hp):
    df = pd.DataFrame({'model_name': model_name,
                       'np_seed': hp.np_seed,
                       'torch_seed': hp.torch_seed,
                       'min_count': hp.min_count,
                       'nonprop_hazards': hp.nonprop_hazards,
                       'batch_size': hp.batch_size,
                       'max_epochs': hp.max_epochs,
                       'patience': hp.patience,
                       'embedding_dim': hp.embedding_dim,
                       'num_weeks_hx': hp.num_weeks_hx,
                       'sample_comp_bh': hp.sample_comp_bh,
                       'concordance': concordance,
                       'brier': brier,
                       'R-squared': r2,
                       'D index': dindex},
                       index=[0])
    with open(hp.data_dir + 'logfile.csv', 'a', newline='\n') as f:
        df.to_csv(f, mode='a', index=False, header=(not f.tell()))

def flip_batch(x, seq_length):
    assert x.shape[0] == seq_length.shape[0], 'Dimension Mismatch!'
    for i in range(x.shape[0]):
        x[i, :seq_length[i]] = x[i, :seq_length[i]].flip(dims=[0])
    return x

class NetRNN(nn.Module):
    def __init__(self, num_input, num_embeddings, hp):
        super().__init__()
        # Parameters ##################################################################################################
        self.nonprop_hazards = hp.nonprop_hazards
        self.add_diagt = hp.add_diagt
        self.add_weeks = hp.add_weeks
        self.num_weeks_hx = hp.num_weeks_hx - 1
        self.rnn_type = hp.rnn_type
        self.num_rnn_layers = hp.num_rnn_layers
        self.embedding_dim = hp.embedding_dim
        self.summarize = hp.summarize
        # Embedding layers ############################################################################################
        self.embed_codes = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=hp.embedding_dim, padding_idx=0)
        if self.add_weeks == 'embedding':
            self.embed_weeks = nn.Embedding(num_embeddings=hp.num_weeks_hx, embedding_dim=hp.embedding_dim,
                                            padding_idx=0)
        if self.add_diagt:
            self.embed_diagt = nn.Embedding(num_embeddings=5, embedding_dim=hp.embedding_dim, padding_idx=0)
        # RNN #########################################################################################################
        if self.add_weeks == 'concat':
            self.embedding_dim = self.embedding_dim + 1
            self.pad_fw = ConstantPad1d((1, 0), 0.)
            self.pad_bw = ConstantPad1d((0, 1), 0.)
        if self.rnn_type == 'LSTM':
            self.rnn_fw = LSTM(input_size=self.embedding_dim, hidden_size=self.embedding_dim,
                               num_layers=self.num_rnn_layers, batch_first=True, dropout=hp.dropout,
                               bidirectional=False)
            self.rnn_bw = LSTM(input_size=self.embedding_dim, hidden_size=self.embedding_dim,
                               num_layers=self.num_rnn_layers, batch_first=True, dropout=hp.dropout,
                               bidirectional=False)
        else:
            self.rnn_fw = GRU(input_size=self.embedding_dim, hidden_size=self.embedding_dim,
                              num_layers=self.num_rnn_layers, batch_first=True, dropout=hp.dropout, bidirectional=False)
            self.rnn_bw = GRU(input_size=self.embedding_dim, hidden_size=self.embedding_dim,
                              num_layers=self.num_rnn_layers, batch_first=True, dropout=hp.dropout, bidirectional=False)
        if self.summarize == 'output_attention':
            self.attention_fw = Attention(embedding_dim=self.embedding_dim)
            self.attention_bw = Attention(embedding_dim=self.embedding_dim)
        # Fully connected layers ######################################################################################
        fc_size = num_input + 2 * self.embedding_dim
        layers = []
        for i in range(hp.num_mlp_layers):
            layers.append(nn.Linear(fc_size, fc_size))
            layers.append(nn.ELU())
        layers.append(nn.Linear(fc_size, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x, code, weeks, diagt, time=None, seq_length=None):
        if self.nonprop_hazards and (time is not None):
            x = torch.cat((x, time), dim=-1)
        if seq_length is None:
            seq_length = (code > 0).sum(dim=-1)
        # Embedding layers ############################################################################################
        embedded = self.embed_codes(code.long())
        if self.add_diagt:
            embedded = embedded + self.embed_diagt(diagt.long())
        if self.add_weeks == 'embedding':
            embedded = embedded + self.embed_weeks(weeks.long())
        if self.add_weeks == 'concat':
            weeks = weeks / float(self.num_weeks_hx)
            delta = torch.clamp(weeks[:, 1:] - weeks[:, :-1], min=0)
            delta_fw = self.pad_fw(delta)
            delta_bw = self.pad_bw(delta)
            embedded_fw = torch.cat((embedded, delta_fw.unsqueeze(dim=-1)), dim=-1)
            embedded_bw = torch.cat((embedded, delta_bw.unsqueeze(dim=-1)), dim=-1)
            embedded_bw = flip_batch(embedded_bw, seq_length)
        else:
            embedded_fw = embedded
            embedded_bw = flip_batch(embedded, seq_length)
        # RNN #########################################################################################################
        packed_fw = nn.utils.rnn.pack_padded_sequence(embedded_fw, seq_length.clamp(min=1), batch_first=True,
                                                      enforce_sorted=False)
        packed_bw = nn.utils.rnn.pack_padded_sequence(embedded_bw, seq_length.clamp(min=1), batch_first=True,
                                                      enforce_sorted=False)
        if self.rnn_type == 'LSTM':
            output_fw, (hidden_fw, _) = self.rnn_fw(packed_fw)
            output_bw, (hidden_bw, _) = self.rnn_bw(packed_bw)
        elif self.rnn_type == 'GRU':
            output_fw, hidden_fw = self.rnn_fw(packed_fw)
            output_bw, hidden_bw = self.rnn_bw(packed_bw)
        if self.summarize == 'hidden':
            hidden_fw = hidden_fw[-1]  # view(num_layers, num_directions=1, batch, hidden_size)[last_state]
            hidden_bw = hidden_bw[-1]  # view(num_layers, num_directions=1, batch, hidden_size)[last_state]
            summary_0, summary_1 = hidden_fw, hidden_bw
        else:
            output_fw, _ = nn.utils.rnn.pad_packed_sequence(output_fw, batch_first=True)
            output_bw, _ = nn.utils.rnn.pad_packed_sequence(output_bw, batch_first=True)
            output_fw = output_fw.view(-1, max(1, seq_length.max()),
                                       self.embedding_dim)  # view(batch, seq_len, num_directions=1, hidden_size)
            output_bw = output_bw.view(-1, max(1, seq_length.max()),
                                       self.embedding_dim)  # view(batch, seq_len, num_directions=1, hidden_size)
            if self.summarize == 'output_max':
                output_fw, _ = output_fw.max(dim=1)
                output_bw, _ = output_bw.max(dim=1)
                summary_0, summary_1 = output_fw, output_bw
            elif self.summarize == 'output_sum':
                output_fw = output_fw.sum(dim=1)
                output_bw = output_bw.sum(dim=1)
                summary_0, summary_1 = output_fw, output_bw
            elif self.summarize == 'output_avg':
                output_fw = output_fw.sum(dim=1) / (seq_length.clamp(min=1).view(-1, 1))
                output_bw = output_bw.sum(dim=1) / (seq_length.clamp(min=1).view(-1, 1))
                summary_0, summary_1 = output_fw, output_bw
            elif self.summarize == 'output_attention':
                mask = (code > 0)[:, :max(1, seq_length.max())]
                summary_0, _ = self.attention_fw(output_fw, mask)
                summary_1, _ = self.attention_bw(output_bw, mask)

        # Fully connected layers ######################################################################################
        x = torch.cat((x, summary_0, summary_1), dim=-1)
        x = self.mlp(x)
        return x