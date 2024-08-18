import matplotlib.pyplot as plt

def plot_performance(pollution_intensity, performance_metrics, model_names, metric_name, title):
    plt.figure(figsize=(10, 6))
    for model_name in model_names:
        plt.plot(pollution_intensity, performance_metrics[model_name], label=model_name)
    plt.xlabel('Pollution Intensity')
    plt.ylabel(metric_name)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
