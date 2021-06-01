import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use(['science'])
plt.rcParams['font.size'] = 8
plt.rcParams['xtick.labelsize'] = 6
plt.rcParams['ytick.labelsize'] = 6

try:
    df_cifar_perf = pd.read_csv('cifar_mdeq_results.csv')
    df_cifar_times = pd.read_csv('cifar_backward_times.csv')
except FileNotFoundError:
    df_cifar_perf = None
    df_cifar_times = None
try:
    df_imagenet_perf = pd.read_csv('imagenet_mdeq_results.csv')
    df_imagenet_times = pd.read_csv('imagenet_backward_times.csv')
except FileNotFoundError:
    df_imagenet_perf = None
    df_imagenet_times = None

fig = plt.figure(figsize=(5.5, 2.8), constrained_layout=False)
g = fig.add_gridspec(2, 1, height_ratios=[1., 1.], hspace=.4, bottom=0.26, top=0.99)
labels = [
    'Original Method',
    r'\textbf{SHINE (ours)}',
    'Jacobian-Free',
]
color_scheme = {
    'original': 'C0',
    'shine': 'C2',
    # 'shine-fallback': 'C2',
    'fpn': 'C1',
}
naming_scheme = {
    'original': 'OM',
    'fpn': 'JF',
    'shine': 'SH',
    'shine-fallback': 'SH',
}

markers_style = {
    0: 'o',
    1: '^',
    2: 's',
    5: 'p',
    7: 'x',
    10: 'D',
    20: 'v',
    27: '*',
}

annotation_offset = {
   (None, 'original'): (-13, -3.8) ,
    ('SMALL', 'original'): (-13, -3.8),
    ('SMALL-refine', 'fpn'): (-13, -3.8),
}

curves = {
    k: ([], []) for k in color_scheme.keys()
}
#CIFAR
if df_cifar_perf is not None:
    ax_cifar = fig.add_subplot(g[0, 0])
    if 'fpn' not in df_cifar_perf.columns:
        df_cifar_perf['fpn'] = False
        df_cifar_times['fpn'] = False
    for accel_kw in ['fpn', 'shine']:
        df_cifar_perf[accel_kw].fillna(False, inplace=True)
        df_cifar_times[accel_kw].fillna(False, inplace=True)
    for method_name, method_color in color_scheme.items():
        if 'shine' in method_name:
            query = 'shine'
        elif method_name == 'fpn':
            query = 'fpn'
        else:
            query = '~fpn & ~shine'
        n_refines = df_cifar_perf.query(query)['n_refine'].unique()
        for n_refine in n_refines:
            if np.isnan(n_refine):
                query_refine = query + f'& n_refine != n_refine'
            else:
                query_refine = query + f'& n_refine=="{n_refine}"'
            x = df_cifar_times.query(query_refine)['median_backward']
            y = df_cifar_perf.query(query_refine)['top1'].mean()
            e = df_cifar_perf.query(query_refine)['top1'].std()
            curves[method_name][0].append(x)
            curves[method_name][1].append(y)
            n_refine = n_refine if not np.isnan(n_refine) else 20
            ep = ax_cifar.errorbar(
                x,
                y,
                ms=2.5,
                yerr=e,
                color=color_scheme[method_name],
                fmt=markers_style[n_refine],
                capsize=1,
            )

    #curves sorting/plotting
    for k, (x, y) in curves.items():
        idx = np.argsort(x)
        x_sorted, y_sorted = [x[i] for i in idx], [y[i] for i in idx]
        ax_cifar.plot(x_sorted, y_sorted, color=color_scheme[k])
    ax_cifar.set_title('CIFAR10')

#Imagenet
if df_imagenet_perf is not None:
    ax_imagenet = fig.add_subplot(g[1, 0])
    if 'fpn' not in df_imagenet_perf.columns:
        df_imagenet_perf['fpn'] = False
        df_imagenet_times['fpn'] = False
    for accel_kw in ['fpn', 'shine']:
        df_imagenet_perf[accel_kw].fillna(False, inplace=True)
        df_imagenet_times[accel_kw].fillna(False, inplace=True)
    for method_name, method_color in color_scheme.items():
        if 'shine' in method_name:
            query = 'shine'
        elif method_name == 'fpn':
            query = 'fpn'
        else:
            query = '~fpn & ~shine'
        n_refines = df_imagenet_perf.query(query)['n_refine'].unique()
        for n_refine in n_refines:
            if np.isnan(n_refine):
                query_refine = query + f'& n_refine != n_refine'
            else:
                query_refine = query + f'& n_refine=="{n_refine}"'
            x = df_imagenet_times.query(query_refine)['median_backward']
            y = df_imagenet_perf.query(query_refine)['top1'].mean()
            e = df_imagenet_perf.query(query_refine)['top1'].std()
            n_refine = n_refine if not np.isnan(n_refine) else 27
            ep = ax_imagenet.errorbar(
                x,
                y,
                ms=3,
                yerr=e,
                color=color_scheme[method_name],
                fmt=markers_style[n_refine],
            )
    ax_imagenet.set_title('ImageNet')
    ax_imagenet.set_xlabel('Median backward pass in ms, on a single V100 GPU, Batch size = 32')

# legend
g_legend = fig.add_gridspec(2, 1, height_ratios=[1., 1.], hspace=.005, bottom=0.05, top=0.15)
ax_legend = fig.add_subplot(g_legend[0, 0])
ax_legend.axis('off')
handles = [
    plt.Rectangle([0, 0], 0.1, 0.1, color=f'C{i}')
    for i in [0, 2, 1]
]
ax_legend.legend(handles, labels, loc='center', ncol=3, handlelength=1., handletextpad=.5)
# legend markers
ax_legend = fig.add_subplot(g_legend[1, 0])
ax_legend.axis('off')
handles_markers = []
markers_labels = []
for marker_name, marker_style in markers_style.items():
    pts = plt.scatter([0], [0], marker=marker_style, c='black', label=marker_name)
    handles_markers.append(pts)
    markers_labels.append(marker_name)
    pts.remove()
# for title
ph = [plt.plot([],marker="", ls="")[0]] # Canvas
handles_markers = ph + handles_markers
markers_labels = [r'\textbf{\# Backward iter.}'] + markers_labels
ax_legend.legend(
    handles_markers,
    markers_labels,
    loc='center',
    ncol=len(markers_labels)+1,
    handlelength=1.5,
    handletextpad=.1,
    columnspacing=1.,
)


fig.supylabel(r'Top-1 accuracy (\%)')


fig.savefig('fig4.pdf', dpi=300);
