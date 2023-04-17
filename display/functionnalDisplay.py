import random
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def display_random_samples(dataset, num_samples=5, output_file=None,experiment_kwargs=None, cmap='gray'):
    sample_indices = random.sample(range(len(dataset)), num_samples)
    samples = []
    for i, sample_idx in enumerate(sample_indices):
        frame, label = dataset[sample_idx]
        samples.append((frame,label))
    build_figure_samples(samples, output_file,experiment_kwargs, cmap)


def build_figure_samples(samples, dispLabel =True, output_file=None,experiment_kwargs=None,cmap='gray'):
    
    num_samples= min(len(samples),8)
    fig, axs = plt.subplots(1, num_samples, figsize=(num_samples * 6, 5))

    for i, sample in enumerate(samples[:num_samples]):
        if dispLabel:
            frame, label = sample
            axs[i].set_title(f"Label: {label}")
        else:
            frame=sample
        if len(frame.shape) >2:
            frame=frame[0]
        im = axs[i].imshow(frame, cmap=cmap)  # Specify the colormap here
        
        axs[i].axis('off')

        # Create an axis for the colorbar alongside the current subplot
        divider = make_axes_locatable(axs[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)  # Display the colorbar
    if output_file:
        # Save the figure as an image file
        plt.savefig(output_file, bbox_inches='tight')
    if experiment_kwargs:
        experiment_kwargs['experiment'].log_figure(figure_name=experiment_kwargs['figure_name'])
    plt.close(fig)  # Close the figure to free up memory
