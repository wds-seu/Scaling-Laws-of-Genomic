import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

colors = [
    'red', 'blue', 'green', 'cyan', 'magenta', 'yellow', 'orange', 'purple', 'brown', 'pink',
    'lime', 'navy', 'teal', 'olive', 'maroon', 'gold', 'coral', 'indigo', 'turquoise', 'violet',
    'gray', 'lavender', 'chocolate', 'crimson'
]

plt.figure(dpi=300)

Loss_scaling_law = []
FLOPs_scaling_law = []
Params_scaling_law = []
Tokens_scaling_law = []

# draw = 'loss'
# draw = 'scaling law'
draw = 'params'
# draw = 'tokens'


def loss_plot(loss_file, color, label, param):
    with open(loss_file) as f:
        lines = f.readlines()
    loss_values = []
    for line in lines:
        loss_search = re.search(r"Loss:\s*([0-9.]+)", line)
        if loss_search:
            loss_values.append(float(loss_search.group(1)))
    steps = range(1, len(loss_values) + 1)
    smoothed_loss_values = np.convolve(loss_values, np.ones(100) / 100, mode='valid')
    smoothed_steps = steps[:len(smoothed_loss_values)]

    if draw == 'loss':
        plt.plot(smoothed_steps, smoothed_loss_values, label=label, marker='.', color=color, markersize=0.01)
    elif draw == 'scaling law':
        FLOPs = [step * 1024 * 128 * param * 6 for step in smoothed_steps]
        plt.plot(FLOPs, smoothed_loss_values, label=label, marker='.', color=color, markersize=0.01)

    def power_law(N, L0, B, beta):
        return L0 + B / (N ** beta)

    def power_law_derivative(N, B, beta):
        return -B * beta / (N ** (beta + 1))

    step_sampled = smoothed_steps[20000::10]
    loss_sampled = smoothed_loss_values[20000::10]
    params, covariance = curve_fit(power_law, step_sampled, loss_sampled, bounds=([0, 0, 0], [100, 100, 1]))
    L0_fit, B_fit, beta_fit = params

    N_fit = np.linspace(min(step_sampled), max(step_sampled), 10000)
    # plt.plot(N_fit, power_law(N_fit, L0_fit, B_fit, beta_fit))
    gradients = power_law_derivative(N_fit, B_fit, beta_fit)
    threshold = 0.0000025
    Nopt = int(N_fit[np.where(np.abs(gradients) < threshold)[0][0]])
    FLOPs_opt = Nopt * 1024 * 128 * param * 6

    if draw == 'loss':
        plt.scatter(Nopt, smoothed_loss_values[Nopt], color='black', marker='^', zorder=10, s=20)
    elif draw == 'scaling law':
        plt.scatter(FLOPs_opt, smoothed_loss_values[Nopt], color='black', marker='^', zorder=10, s=20)

    FLOPs_scaling_law.append(FLOPs_opt)
    Loss_scaling_law.append(smoothed_loss_values[Nopt])
    Params_scaling_law.append(param)
    Tokens_scaling_law.append(Nopt * 1024 * 128)


# loss_plot('./result_reference_sampling_15M.txt', colors[0], 'reference 15M', 14963712)
# loss_plot('./result_reference_sampling_28M.txt', colors[1], 'reference 28M', 27565578)
# loss_plot('./result_reference_sampling_60M.txt', colors[2], 'reference 60M', 60417290)
# loss_plot('./result_reference_sampling_117M.txt', colors[3], 'reference 117M', 117074176)
# loss_plot('./result_reference_sampling_230M.txt', colors[4], 'reference 230M', 230403328)
# 0.8314

# loss_plot('./result_pan-genome_sampling_15M.txt', colors[5], 'pan-genome 15M', 14963712)
# loss_plot('./result_pan-genome_sampling_28M.txt', colors[6], 'pan-genome 28M', 27565578)
# loss_plot('./result_pan-genome_sampling_60M.txt', colors[7], 'pan-genome 60M', 60417290)
# loss_plot('./result_pan-genome_sampling_117M.txt', colors[8], 'pan-genome 117M', 117074176)
# loss_plot('./result_pan-genome_sampling_230M.txt', colors[9], 'pan-genome 230M', 230403328)
# 0.8387

loss_plot('./result_variation_sampling_15M.txt', colors[10], 'variation 15M', 14963712)
loss_plot('./result_variation_sampling_28M.txt', colors[11], 'variation 28M', 27565578)
loss_plot('./result_variation_sampling_60M.txt', colors[12], 'variation 60M', 60417290)
loss_plot('./result_variation_sampling_117M.txt', colors[13], 'variation 117M', 117074176)
loss_plot('./result_variation_sampling_230M.txt', colors[14], 'variation 230M', 230403328)
# 0.8297

# loss_plot('./result_reference_sampling_15M_kmer.txt', colors[15], 'reference kmer 15M', 14963712)
# loss_plot('./result_reference_sampling_28M_kmer.txt', colors[16], 'reference kmer 28M', 27565578)
# loss_plot('./result_reference_sampling_60M_kmer.txt', colors[17], 'reference kmer 60M', 60417290)
# loss_plot('./result_reference_sampling_117M_kmer.txt', colors[18], 'reference kmer 117M', 117074176)
# loss_plot('./result_reference_sampling_230M_kmer.txt', colors[19], 'reference kmer 230M', 230403328)
# 0.8074

if draw == 'loss':
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.show()
elif draw == 'scaling law':
    FLOPs_scaling_law, Loss_scaling_law = np.array(FLOPs_scaling_law), np.array(Loss_scaling_law)
    log_x = np.log10(FLOPs_scaling_law)
    coefficients = np.polyfit(log_x, Loss_scaling_law, 1)
    slope, intercept = coefficients
    x_fit = np.linspace(1e14, 1e20, 10)
    y_fit = slope * np.log10(x_fit) + intercept
    plt.plot(x_fit, y_fit, color='black', zorder=20)
    plt.xlabel('FLOPs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend(loc='lower left')
    plt.xscale('log')
    plt.show()
elif draw == 'params':
    FLOPs_scaling_law, Params_scaling_law = np.array(FLOPs_scaling_law), np.array(Params_scaling_law)
    log_x, log_y = np.log10(FLOPs_scaling_law), np.log10(Params_scaling_law)
    coefficients = np.polyfit(log_x, log_y, 1)
    slope, intercept = coefficients
    x_fit = np.linspace(1e14, 1e20, 10)
    y_fit = 10 ** (slope * np.log10(x_fit) + intercept)
    plt.scatter(FLOPs_scaling_law, Params_scaling_law, color='black')
    plt.plot(x_fit, y_fit, color='black', label=f'log10(y) = {slope:.4f} * log10(x) + {intercept:.4f}')
    plt.xlabel('FLOPs')
    plt.ylabel('Params')
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
else:
    FLOPs_scaling_law, Tokens_scaling_law = np.array(FLOPs_scaling_law), np.array(Tokens_scaling_law)
    log_x, log_y = np.log10(FLOPs_scaling_law), np.log10(Tokens_scaling_law)
    coefficients = np.polyfit(log_x, log_y, 1)
    slope, intercept = coefficients
    x_fit = np.linspace(1e14, 1e20, 10)
    y_fit = 10 ** (slope * np.log10(x_fit) + intercept)
    plt.scatter(FLOPs_scaling_law, Tokens_scaling_law, color='black')
    plt.plot(x_fit, y_fit, color='black', label=f'log10(y) = {slope:.4f} * log10(x) + {intercept:.4f}')
    plt.xlabel('FLOPs')
    plt.ylabel('Tokens')
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
