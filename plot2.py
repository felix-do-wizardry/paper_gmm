# %%
# python -m pip install -U plotly kaleido nbformat pandas numpy

# %%
from matplotlib.pyplot import tick_params
import numpy as np
import pandas as pd
import json, time, re, string, os

import plotly.express as px
import plotly.graph_objects as go

# %%
class FigFormat:
    def __init__(self, fig_size=[800, 800]):
        self.fig_size = fig_size
        
        self.font = {
            # 'axis': 32,
            # 'tick': 30,
            # 'legend': 16,
            
            'axis': 48,
            
            # 'tick': 48,
            'tick': 72,
            
            # 'legend': 24,
            'legend': 48,
            
        }
        self.size = {
            'line': 4,
            'marker': 12,
            
            'linewidth': 5,
            'tickwidth': 3,
            'gridwidth': 1,
        }
        self.fig_config = {
            # 'legend_opacity': 0.5,
            'legend_opacity': 1.0,
            'legend_color_value': 255,
        }
    
    def format(self,
                fig,
                x_title='Sequence Length',
                y_title='Ratio',
                legend_title='Type',
                corner='tr',
                x_dtick=None,
                y_dtick=None,
                axis_angle=90,
                showlegend=True,
                **kwargs,
                ):
        
        fig.update_layout(
            font={
                'color': '#000000',
                'family': 'Helvetica',
            },
            paper_bgcolor="#FFFFFF",
            plot_bgcolor='rgba(0,0,0,0)',
            title=dict(
                font=dict(color='#000000'),
            ),
        )

        axes = dict(
            color='#000000',
            # showgrid=False,
            linecolor='#000000',
            gridcolor='#aaaaaa',
            tickcolor='#000000',
            mirror=True,
            
            linewidth=self.size['linewidth'],
            tickwidth=self.size['tickwidth'],
            gridwidth=self.size['gridwidth'],
            
            title_font = {"size": self.font['axis'],},
            # title_standoff = 16,
            tickfont={'size': self.font['tick'],},
            ticks='outside',
            
            # dtick=0.02,
        )
        fig.update_xaxes(
            **axes,
        )
        fig.update_yaxes(
            **axes,
        )

        fig.update_traces(
            marker=dict(size=self.size['marker'], symbol='square'),
            line=dict(width=self.size['line']),
        )
        
        legend_margin = 0.02
        legend_pos_dict = {
            f'{_yk[0]}{_xk[0]}': dict(
                yanchor=_yk,
                xanchor=_xk,
                x=legend_margin + _x * (1 - legend_margin * 2),
                y=legend_margin + _y * (1 - legend_margin * 2),
            )
            for _x in [0, 1]
            for _y in [0, 1]
            for _xk in [['left', 'right'][_x]]
            for _yk in [['bottom', 'top'][_y]]
        }
        assert corner in legend_pos_dict
        _bgcolor = 'rgba({0}, {0}, {0}, {1})'.format(
            self.fig_config['legend_color_value'],
            self.fig_config['legend_opacity'],
        )
        fig.update_layout(
            width=self.fig_size[0],
            height=self.fig_size[1],
            xaxis=dict(
                title_text=x_title,
                tickangle = axis_angle,
                # dtick=None if x_dtick is None else x_dtick,
            ),
            yaxis=dict(
                title_text=y_title,
            ),
            autosize=False,
            showlegend=showlegend,
            legend=dict(
                title_text=legend_title,
                
                **legend_pos_dict[corner],
                font=dict(size=self.font['legend'],),
                # bgcolor='rgba(255, 255, 255, 0.75)',
                bgcolor=_bgcolor,
                # bordercolor="rgba(0, 0, 0, 0.25)",
                # borderwidth=3,
            ),
        )
        if x_dtick is not None:
            fig.update_xaxes(
                dtick=x_dtick,
            )
        if y_dtick is not None:
            fig.update_yaxes(
                dtick=y_dtick,
            )
        
        return fig
    
    def format_dual(self, fig, **kwargs):
        fig_clean = go.Figure(fig)
        fig_ref = go.Figure(fig)
        fig_clean = self.format(
            fig_clean,
            x_title='',
            y_title='',
            legend_title='',
            # corner='tr',
            # axis_angle=90,
            showlegend=False,
            **{
                k: v
                for k, v in kwargs.items()
                if k in ['x_dtick', 'y_dtick']
            },
        )
        fig_ref = self.format(
            fig_ref,
            **kwargs,
        )
        return fig_clean, fig_ref


FF = FigFormat(
    fig_size=[800, 800],
)
FF

# %%
fp_csv = 'data/metrics.csv'
fp_csv_ratio = 'data/metrics_ratio.csv'
fp_csv = 'data/swin_metrics.csv'
fp_csv_ratio = 'data/swin_metrics_ratio.csv'

# %%
df = pd.read_csv(fp_csv, index_col=0)
df_ratio = pd.read_csv(fp_csv_ratio, index_col=0)
df_ratio['seq_len_str'] = [str(v) for v in df_ratio['seq_len']]
df_ratio

# %% RATIO PER LEN - flops 
# group_prefix = 'test_block_ratio'
levels = ['block', 'model']
# metrics_ratio_len = ['mem', 'flops', 'time']
metrics_ratio_len = ['mem', 'flops']
figs_ratio_len = {
    'clean': {},
    'ref': {},
}
for _level in levels:
    for _metric in metrics_ratio_len:
        _name = f'ratio_test_{_level}_{_metric}'
        
        _value_diff = np.max(df_ratio[_metric]) - np.min(df_ratio[_metric])
        
        _df = df_ratio[
            (df_ratio['level'] == _level)
        ].sort_values(['seq_len', 'type_full'])
        if _df.shape[0] < 1:
            continue
        
        fig = px.line(
            _df,
            x='seq_len_str',
            # x='seq_len',
            y=_metric,
            color='type_full',
            # color='type',
            markers=dict(size=12),
            # range_y=[0.85, 1.05],
            # range_y=[.75, 1.],
            # title=f'{_metric} ratio {_method}/softmax' if GLOBAL_PLOT_TITLE else '',
        )
        # _y_sub = {
        #     'flops': 'FLOPS Ratio',
        #     'time': 'Time Ratio',
        # }
        # format_full(
        #     fig,
        #     # y_title=f"{'Train' if _training else 'Test'} {_y_sub[_metric.split('_')[-1]]}",
        #     # legend_title='Model Dim',
        #     corner='tr' if _metric.endswith('flops') else 'br',
        #     y_dtick=0.1,
        # )
        print(f'created [{_name}]')
        # fig.show()
        
        fig_clean, fig_ref = FF.format_dual(
            fig,
            y_dtick=0.2 if _value_diff >= 0.5 else 0.1,
        )
        # fig_clean.show()
        # fig_ref.show()
        figs_ratio_len['clean'][_name] = fig_clean
        figs_ratio_len['ref'][_name] = fig_ref


# print(f'created {len(figs_ratio_len)} plots for [per_len] metrics')

# %%
figs_all = {
    k: {
        k2: _figs[k2][k]
        for k2 in ['clean', 'ref']
    }
    for _figs in [figs_ratio_len]
    for i, k in enumerate(_figs['clean'].keys())
    # for k, _fig in _figs['clean'].items()
}
# figs_all = {
#     k: _fig
#     for _figs in [figs_ratio_len]
#     for k, _fig in _figs['clean'].items()
# }
# figs_all_ref = {
#     k: _fig
#     for _figs in [figs_ratio_len]
#     for k, _fig in _figs['ref'].items()
# }
len(figs_all)

# %%
for k, _figs in figs_all.items():
    print(k)
    _figs['ref'].show()

# %%
_dp = './paper_gmm/plots_deit'
_dp = './paper_gmm/plots_swin'

_dp_clean = os.path.join(_dp, 'clean')
_dp_ref = os.path.join(_dp, 'ref')
if not os.path.isdir(_dp_clean):
    os.makedirs(_dp_clean)
if not os.path.isdir(_dp_ref):
    os.makedirs(_dp_ref)

for name, figs in figs_all.items():
    for k2, ___dp in zip(['clean', 'ref'], [_dp_clean, _dp_ref]):
        figs[k2].write_image(os.path.join(___dp, f'{name}.png'))

# for name, fig in figs_all_ref.items():
#     fig.write_image(os.path.join(_dp_ref, f'{name}.png'))

print(f'[PLOT] {len(figs_all)} plots saved in <{_dp}>')

# %%
figs_groups = [
    [figs_ratio_len],
    # [figs_abs_len, figs_abs_dim],
    # [figs_head_len, figs_head_dim],
]
fp_rm = os.path.join(_dp, f'README.md')
readme_lines = [
    '# Model Metrics Plot for gmm deit',
]
image_width = 320
image_cols = 4
readme_lines.append('## PLOTS')
for gi, (name, _figs) in enumerate(figs_all.items()):
    # readme_lines.append([
    #     '## I - TEST BLOCK RATIO',
    #     '## II - TEST MODEL RATIO',
    #     # '## III - RATIO PER HEAD',
    # ][gi])
    
    name_lines = []
    name_lines.append(f'> {name}')
    readme_lines.extend([
        *name_lines,
        '<p float="left" align="left">',
        f'<img src="clean/{name}.png" width="{image_width}" />',
        f'<img src="ref/{name}.png" width="{image_width}" />',
        '</p>',
    ])

# readme_lines

# %%
readme_txt = '\n\n'.join(readme_lines)

with open(fp_rm, 'w') as fo:
    fo.writelines(readme_txt)
print(f'[PLOT] saved README.md at <{fp_rm}>')

# %%