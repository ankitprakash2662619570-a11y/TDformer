from layers.TDformer_EncDec import EncoderLayer, Encoder, DecoderLayer, Decoder, AttentionLayer, series_decomp, series_decomp_multi
import torch.nn as nn
import torch
from layers.Embed import DataEmbedding
from layers.Attention import FourierAttention, FullAttention
from layers.RevIN import RevIN
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.version = configs.version
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.output_stl = configs.output_stl
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Decomp
        kernel_size = configs.moving_avg
        if isinstance(kernel_size, list):
            self.decomp = series_decomp_multi(kernel_size)
        else:
            self.decomp = series_decomp(kernel_size)

        # Embedding
        self.enc_seasonal_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.dec_seasonal_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)

        # Encoder
        if configs.version == 'Wavelet':
            enc_self_attention = WaveletAttention(in_channels=configs.d_model,
                                                  out_channels=configs.d_model,
                                                  seq_len_q=configs.seq_len,
                                                  seq_len_kv=configs.seq_len,
                                                  ich=configs.d_model,
                                                  T=configs.temp,
                                                  activation=configs.activation,
                                                  output_attention=configs.output_attention)
            dec_self_attention = WaveletAttention(in_channels=configs.d_model,
                                                  out_channels=configs.d_model,
                                                  seq_len_q=configs.seq_len // 2 + configs.pred_len,
                                                  seq_len_kv=configs.seq_len // 2 + configs.pred_len,
                                                  ich=configs.d_model,
                                                  T=configs.temp,
                                                  activation=configs.activation,
                                                  output_attention=configs.output_attention)
            dec_cross_attention = WaveletAttention(in_channels=configs.d_model,
                                                   out_channels=configs.d_model,
                                                   seq_len_q=configs.seq_len // 2 + configs.pred_len,
                                                   seq_len_kv=configs.seq_len,
                                                   ich=configs.d_model,
                                                   T=configs.temp,
                                                   activation=configs.activation,
                                                   output_attention=configs.output_attention)
        elif configs.version == 'Fourier':
            enc_self_attention = FourierAttention(T=configs.temp, activation=configs.activation,
                                                  output_attention=configs.output_attention)
            dec_self_attention = FourierAttention(T=configs.temp, activation=configs.activation,
                                                  output_attention=configs.output_attention)
            dec_cross_attention = FourierAttention(T=configs.temp, activation=configs.activation,
                                                   output_attention=configs.output_attention)
        elif configs.version == 'Time':
            enc_self_attention = FullAttention(False, T=configs.temp, activation=configs.activation,
                                               attention_dropout=configs.dropout,
                                               output_attention=configs.output_attention)
            dec_self_attention = FullAttention(True, T=configs.temp, activation=configs.activation,
                                               attention_dropout=configs.dropout,
                                               output_attention=configs.output_attention)
            dec_cross_attention = FullAttention(False, T=configs.temp, activation=configs.activation,
                                                attention_dropout=configs.dropout,
                                                output_attention=configs.output_attention)

        # Encoder
        self.seasonal_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        enc_self_attention,
                        configs.d_model),
                    configs.d_model,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(2)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # Decoder
        self.seasonal_decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        dec_self_attention,
                        configs.d_model),
                    AttentionLayer(
                        dec_cross_attention,
                        configs.d_model),
                    configs.d_model,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(1)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

        # Trend RNN
        bidirectional = configs.rnn_type == 'BiLSTM'
        if configs.rnn_type == 'LSTM' or bidirectional:
            self.trend_rnn = nn.LSTM(
                input_size=configs.enc_in,
                hidden_size=configs.hidden_size,
                num_layers=configs.num_layers,
                batch_first=True,
                dropout=configs.dropout,
                bidirectional=bidirectional
            )
        elif configs.rnn_type == 'GRU':
            self.trend_rnn = nn.GRU(
                input_size=configs.enc_in,
                hidden_size=configs.hidden_size,
                num_layers=configs.num_layers,
                batch_first=True,
                dropout=configs.dropout
            )
        elif configs.rnn_type == 'RNN':
            self.trend_rnn = nn.RNN(
                input_size=configs.enc_in,
                hidden_size=configs.hidden_size,
                num_layers=configs.num_layers,
                batch_first=True,
                dropout=configs.dropout
            )
        hidden_dim = configs.hidden_size * (2 if bidirectional else 1)
        self.linear = nn.Linear(hidden_dim, configs.pred_len * configs.enc_in)

        self.revin_trend = RevIN(configs.enc_in).to(self.device)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        # decomp init
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]]).to(self.device)
        seasonal_enc, trend_enc = self.decomp(x_enc)
        seasonal_dec = F.pad(seasonal_enc[:, -self.label_len:, :], (0, 0, 0, self.pred_len))

        # seasonal
        enc_out = self.enc_seasonal_embedding(seasonal_enc, x_mark_enc)
        enc_out, attn_e = self.seasonal_encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_seasonal_embedding(seasonal_dec, x_mark_dec)
        seasonal_out, attn_d = self.seasonal_decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        seasonal_out = seasonal_out[:, -self.pred_len:, :]

        seasonal_ratio = seasonal_enc.abs().mean(dim=1) / seasonal_out.abs().mean(dim=1)
        seasonal_ratio = seasonal_ratio.unsqueeze(1).expand(-1, self.pred_len, -1)

        # trend
        trend_enc = self.revin_trend(trend_enc, 'norm')
        out, _ = self.trend_rnn(trend_enc)
        last = out[:, -1, :]
        trend_out = self.linear(last).view(-1, self.pred_len, -1)
        trend_out = self.revin_trend(trend_out, 'denorm')

        # final
        dec_out = trend_out + seasonal_ratio * seasonal_out

        if self.output_attention:
            return dec_out, (attn_e, attn_d)
        elif self.output_stl:
            return dec_out, trend_enc, seasonal_enc, trend_out, seasonal_ratio * seasonal_out
        else:
            return dec_out
