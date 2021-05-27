import copy

import matplotlib.pyplot as plt
import numpy as np

plt.style.use(['science'])
plt.rcParams['font.size'] = 8
plt.rcParams['xtick.labelsize'] = 6
plt.rcParams['ytick.labelsize'] = 6

results_cifar = {
    None: {
        'original': {'perf': (93.512, 0.15210626), 'time': (14.5, 0), 'backward-time': (209)},
        'shine': {'perf': (93.506, 0.18445666), 'time': (11.5, 0), 'backward-time': (273)},
        'fpn': {'perf': (93.516, 0.17001258), 'time': (11.5, 0), 'backward-time': (198)},
    },
    10: {
        'original': {'perf': (93.47399, 0.12986298), 'time': (12.9, 0), 'backward-time': (142.8)},
        'shine': {'perf': (93.467995, 0.15638326), 'time': (13, 0), 'backward-time': (157)},
        'fpn': {'perf': (93.602005, 0.14147879), 'time': (12.7, 0), 'backward-time': (142.6)},
    },
    7: {
        'original': {'perf': (93.467995, 0.14958589), 'time': (12.9, 0), 'backward-time': (108)},
        'shine': {'perf': (93.498, 0.23172565), 'time': (13, 0), 'backward-time': (119)},
        'fpn': {'perf': (93.495995, 0.27557778), 'time': (12.7, 0), 'backward-time': (109)},
    },
    5: {
        'original': {'perf': (93.621994, 0.12890144), 'time': (12.9, 0), 'backward-time': (86.4)},
        'shine': {'perf': (93.54399, 0.18575287), 'time': (13, 0), 'backward-time': (96.6)},
        'fpn': {'perf': (93.476, 0.14759484), 'time': (12.7, 0), 'backward-time': (86.5)},
    },
    2: {
        'original': {'perf': (93.409996, 0.15059815), 'time': (12.9, 0), 'backward-time': (53.7)},
        'shine': {'perf': (93.404, 0.14193195), 'time': (13, 0), 'backward-time': (58.9)},
        'fpn': {'perf': (93.478004, 0.24854626), 'time': (12.7, 0), 'backward-time': (53.3)},
    },
    1: {
        'original': {'perf': (92.642, 0.09927742), 'time': (12.9, 0), 'backward-time': (41.54)},
        'shine': {'perf': (93.376, 0.24063425), 'time': (13, 0), 'backward-time': (46.9)},
        'fpn': {'perf': (93.343994, 0.18325926), 'time': (12.7, 0), 'backward-time': (41.58)},
    },
    0: {
        'fpn': {'perf': (93.091995, 0.11338575), 'time': (12.7, 0), 'backward-time': (12.9)},
        'shine': {'perf': (93.144, 0.1843464), 'time': (13, 0), 'backward-time': (16.0)},
    },
#     'TINY': {
#         'original': {'perf': (84.692, 0.23549175), 'time': (1.75, 0)},
#         'shine': {'perf': (84.14, 0.15671521), 'time': (1.25, 0)},
#         'fpn': {'perf': (84.05601, 0.19520284), 'time': (1.25, 0)},
#     }
}

results_imagenet = {
    'SMALL': {
        'original': {'perf': (75.53, 0.), 'time': (-1, 0), 'backward-time': (798)},
        'shine-fallback': {'perf': (70.374, 0.), 'time': (-1, 0), 'backward-time': (35.3)},
        'fpn': {'perf': (72.594, 0.), 'time': (-1, 0), 'backward-time': (13.5)},
    },
    'SMALL-refine': {
        'original': {'perf': (72.638, 0.), 'time': (-1, 0), 'backward-time': (212)},
        'shine': {'perf': (74.21, 0.), 'time': (-1, 0), 'backward-time': (187)},
        'fpn': {'perf': (74.518, 0.), 'time': (-1, 0), 'backward-time': (186)},
    },
}

fig = plt.figure(figsize=(5.5, 2.8), constrained_layout=False)
g = fig.add_gridspec(2, 1, height_ratios=[1., 1.], hspace=.4, bottom=0.26, top=0.99)
handles = []
labels = [
    'Original Method',
    r'\textbf{SHINE (ours)}',
    'Jacobian-Free',
]
color_scheme = {
    'original': 'C0',
    'shine': 'C2',
    'shine-fallback': 'C2',
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
    20: '8',
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
ax_cifar = fig.add_subplot(g[0, 0])
for xp_name, xp_res in results_cifar.items():
    if xp_name == 'TINY':
        continue
    for method_name, method_res in xp_res.items():
        x = method_res['backward-time']
        y = method_res['perf'][0]
        e = method_res['perf'][1]
        curves[method_name][0].append(x)
        curves[method_name][1].append(y)
        n_refine = xp_name if xp_name is not None else 20
        ep = ax_cifar.errorbar(
            x,
            y,
            ms=2,
            yerr=e,
            color=color_scheme[method_name],
            fmt=markers_style[n_refine],
            capsize=1,
        )
        if xp_name == 0 or xp_name is None:
            handles.append(ep[0])

#curves sorting/plotting
for k, (x, y) in curves.items():
    idx = np.argsort(x)
    x_sorted, y_sorted = [x[i] for i in idx], [y[i] for i in idx]
    ax_cifar.plot(x_sorted, y_sorted, color=color_scheme[k])
ax_cifar.set_title('CIFAR10')

#Imagenet
ax_imagenet = fig.add_subplot(g[1, 0])
for xp_name, xp_res in results_imagenet.items():
    if xp_name == 'TINY':
        continue
    for method_name, method_res in xp_res.items():
        x = method_res['backward-time']
        y = method_res['perf'][0]
        e = method_res['perf'][1]
        if xp_name == 'SMALL-refine':
            n_refine = 5
        else:
            if method_name == 'original':
                n_refine = 27
            else:
                n_refine = 0
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
ax_legend.legend(handles, labels, loc='center', ncol=3, handlelength=1.5, handletextpad=.1)
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


fig.savefig('merged_results_latency_style.pdf', dpi=300);
