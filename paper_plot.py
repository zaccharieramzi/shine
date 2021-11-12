import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use(['science'])
plt.rcParams['font.size'] = 8
plt.rcParams['xtick.labelsize'] = 6
plt.rcParams['ytick.labelsize'] = 6

MARKER_SIZE = 4

METHODS_ORDER = ['original', 'fpn', 'shine']
LABELS = {
    'original': 'Original Method',
    # 'SHINE (ours)',
    'fpn': 'Jacobian-Free',
    'shine': r'\textbf{SHINE (ours)}',
}
COLOR_SCHEME = {
    'original': 'C0',
    'shine': 'C2',
    # 'shine-fallback': 'C2',
    'fpn': 'C1',
}

MARKERS_STYLE = {
    0: 'o',
    1: '^',
    5: 'p',
    27: '*',
    None: None,
    2: 's',
    10: 'D',
    # 7: 'x',
    20: '*',
}

aggreg_cifar = aggreg_imagenet = False
try:
    df_cifar_perf = pd.read_csv('cifar_mdeq_results.csv')
    df_cifar_times = pd.read_csv('cifar_backward_times.csv')
except FileNotFoundError:
    try:
        df_cifar_perf = pd.read_csv('cifar_aggreg_results.csv')
        df_cifar_times = pd.read_csv('cifar_aggreg_results.csv')
        aggreg_cifar = True
    except FileNotFoundError:
        df_cifar_perf = None
        df_cifar_times = None
try:
    df_imagenet_perf = pd.read_csv('imagenet_mdeq_results.csv')
    df_imagenet_times = pd.read_csv('imagenet_backward_times.csv')
except FileNotFoundError:
    try:
        df_imagenet_perf = pd.read_csv('imagenet_aggreg_results.csv')
        df_imagenet_times = pd.read_csv('imagenet_aggreg_results.csv')
        aggreg_imagenet = True
    except FileNotFoundError:
        df_imagenet_perf = None
        df_imagenet_times = None


# Vertical line to separate vanilla and refine models
def add_vline(ax, x_pos, small_delta=False):
    """
    Adds a dashed vertical line in the graph specified by at x_pos.
    Also adds a text on the upper left side of the dashed vertical line specifying 'Vanilla'
    and a text on the upper right side of the dashed vertical line specifying 'Refined'
    """
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    ax.axvline(x=x_pos, color='k', linestyle='--')
    if small_delta:
        y_delta = 0.08
        x_delta = 3
    else:
        y_delta = 0.5
        x_delta = 10
    ax.text(
        x_pos - x_delta, ax.get_ylim()[1] - y_delta, 'Vanilla',
        horizontalalignment='right', verticalalignment='top',
        fontsize=7
    )
    ax.text(
        x_pos + x_delta, ax.get_ylim()[1] - y_delta, 'Refined',
        horizontalalignment='left', verticalalignment='top',
        fontsize=7
    )

    ax.fill_between(
        [ax.get_xlim()[0], x_pos],
        [y_lim[0]] * 2, [y_lim[1]] * 2,
        color='k', alpha=0.2
     )
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

fig = plt.figure(figsize=(5.5, 2.8), constrained_layout=False)
g_overall = fig.add_gridspec(ncols=2, nrows=1, width_ratios=[0.9, 0.1])
g = g_overall[0, 0].subgridspec(2, 1, height_ratios=[1., 1.], hspace=.4)

annotation_offset = {
   (None, 'original'): (-13, -3.8) ,
    ('SMALL', 'original'): (-13, -3.8),
    ('SMALL-refine', 'fpn'): (-13, -3.8),
}

curves = {
    k: ([], []) for k in COLOR_SCHEME.keys()
}
#CIFAR
if df_cifar_perf is not None:
    ax_cifar = fig.add_subplot(g[0, 0])
    if 'fpn' not in df_cifar_perf.columns:
        df_cifar_perf['fpn'] = False
    if 'fpn' not in df_cifar_times.columns:
        df_cifar_times['fpn'] = False
    for accel_kw in ['fpn', 'shine', 'refine']:
        df_cifar_perf[accel_kw].fillna(False, inplace=True)
        df_cifar_times[accel_kw].fillna(False, inplace=True)
    for method_name, method_color in COLOR_SCHEME.items():
        if 'shine' in method_name:
            query = 'shine'
        elif method_name == 'fpn':
            query = 'fpn'
        else:
            query = '~fpn & ~shine'
        n_refines = df_cifar_perf.query(query)['n_refine'].unique()
        for n_refine in n_refines:
            if n_refine == 7:
                continue
            if np.isnan(n_refine):
                query_refine = query + '& n_refine != n_refine & (refine or (~fpn and ~shine)) '
            else:
                if n_refine > 0:
                    query_refine = query + '& n_refine==@n_refine & (refine or (~fpn and ~shine))'
                else:
                    query_refine = query + '& (~refine or n_refine==@n_refine)'
            x = df_cifar_times.query(query_refine)['median_backward']
            if aggreg_cifar:
                perf = df_cifar_perf.query(query_refine)
                assert len(perf) == 1
                y = perf['top1']
                e = perf['std']
            else:
                y = df_cifar_perf.query(query_refine)['top1'].mean()
                e = df_cifar_perf.query(query_refine)['top1'].std()
            curves[method_name][0].append(x)
            curves[method_name][1].append(y)
            n_refine = n_refine if not np.isnan(n_refine) else 20
            ep = ax_cifar.errorbar(
                x,
                y,
                ms=MARKER_SIZE,
                yerr=e,
                color=COLOR_SCHEME[method_name],
                fmt=MARKERS_STYLE[n_refine],
                capsize=1,
            )

    #curves sorting/plotting
    for k, (x, y) in curves.items():
        x = np.array(x).flatten()
        idx = np.argsort(x)
        x_sorted, y_sorted = [x[i] for i in idx], [y[i] for i in idx]
        ax_cifar.plot(x_sorted, y_sorted, color=COLOR_SCHEME[k])
    ax_cifar.set_title('CIFAR10')

add_vline(ax_cifar, 27, small_delta=True)

#Imagenet
if df_imagenet_perf is not None:
    ax_imagenet = fig.add_subplot(g[1, 0])
    if 'fpn' not in df_imagenet_perf.columns:
        df_imagenet_perf['fpn'] = False
    if 'fpn' not in df_imagenet_times.columns:
        df_imagenet_times['fpn'] = False
    for accel_kw in ['fpn', 'shine', 'refine']:
        df_imagenet_perf[accel_kw].fillna(False, inplace=True)
        df_imagenet_times[accel_kw].fillna(False, inplace=True)
    for method_name, method_color in COLOR_SCHEME.items():
        if 'shine' in method_name:
            query = 'shine'
        elif method_name == 'fpn':
            query = 'fpn'
        else:
            query = '~fpn & ~shine'
        n_refines = df_imagenet_perf.query(query)['n_refine'].unique()
        for n_refine in n_refines:
            if np.isnan(n_refine):
                query_refine = query + '& n_refine != n_refine & (refine or (~fpn and ~shine))'
            else:
                if n_refine > 0:
                    query_refine = query + '& n_refine==@n_refine & (refine or (~fpn and ~shine))'
                else:
                    query_refine = query + '& (~refine or n_refine==@n_refine)'
            x = df_imagenet_times.query(query_refine)['median_backward']
            if aggreg_imagenet:
                perf = df_imagenet_perf.query(query_refine)
                assert len(perf) == 1
                y = perf['top1']
                e = perf['std']
            else:
                y = df_imagenet_perf.query(query_refine)['top1'].mean()
                e = df_imagenet_perf.query(query_refine)['top1'].std()
            n_refine = n_refine if not np.isnan(n_refine) else 27
            ep = ax_imagenet.errorbar(
                x,
                y,
                ms=MARKER_SIZE,
                yerr=e,
                color=COLOR_SCHEME[method_name],
                fmt=MARKERS_STYLE[n_refine],
            )
    ax_imagenet.set_title('ImageNet')
    ax_imagenet.set_xlabel('Backward pass wall-clock time [ms]')

add_vline(ax_imagenet, 75)

# legend
g_legend = g_overall[0, 1].subgridspec(
    5, 1, height_ratios=[.1, 1., .2, 1., 1.2], hspace=1.
)
ax_legend = fig.add_subplot(g_legend[1, 0])
ax_legend.axis('off')
method_handles = [
    plt.Rectangle([0, 0], 0.1, 0.1, color=COLOR_SCHEME[l])
    for l in METHODS_ORDER
]
method_labels = [LABELS[l] for l in METHODS_ORDER]
ax_legend.legend(
    method_handles, method_labels, loc='center', ncol=1,
    handlelength=1., handletextpad=.5, title=r'\textbf{Methods}'
)
# legend markers
ax_legend = fig.add_subplot(g_legend[3, 0])
ax_legend.axis('off')
handles_markers = []
markers_labels = []
for marker_name, marker_style in MARKERS_STYLE.items():
    if marker_name == 20:
        continue
    pts = plt.scatter(
        [0], [0], marker=marker_style, c='black', label=marker_name,
        alpha=1 if marker_style is not None else 0
    )
    handles_markers.append(pts)
    markers_labels.append(
        marker_name if marker_name not in [0, 27] else
        '0 - Vanilla' if marker_name == 0 else 'Full backward')
    pts.remove()

# Add legend
ax_legend.legend(
    handles_markers,
    markers_labels,
    loc='center',
    ncol=2,
    handlelength=1.5,
    handletextpad=.1,
    columnspacing=-4,
    title=r'\textbf{\# Backward iter.}'
)

# Y Label
# fig.supylabel('Top-1 accuracy (\%)')
ax_perf = fig.add_subplot(g[:], frameon=False)
ax_perf.axes.xaxis.set_ticks([])
ax_perf.axes.yaxis.set_ticks([])
ax_perf.spines['top'].set_visible(False)
ax_perf.spines['right'].set_visible(False)
ax_perf.spines['bottom'].set_visible(False)
ax_perf.spines['left'].set_visible(False)
ax_perf.set_ylabel('Top-1 accuracy (\%)', labelpad=24.)


fig.savefig('figures/merged_results_latency_style.pdf', dpi=300)
plt.show()
