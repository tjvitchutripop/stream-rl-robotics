import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

global title_size 
title_size = 27
global label_size
label_size = 25
global tick_size
tick_size = 24


def create_plot_1():
    # Load data from csv
    data = pd.read_csv("data/obgd-leg.csv")
    data2 = pd.read_csv("data/aobgd-leg.csv")
    data3 = pd.read_csv("data/aobgdcbp-leg.csv")
    data4 = pd.read_csv("data/aobgdln-leg.csv")
    data5 = pd.read_csv("data/adam-leg.csv")

    # Extract relevant columns and handle different lengths
    datasets = [
        (data, 'ObGD'),
        (data2, 'AdaptiveObGD'),
        (data3, 'AdaptiveObGD + CBP'),
        (data4, 'AdaptiveObGD + LayerNorm'),
        (data5, 'Adam')
    ]

    # Create figure with specified size (width, height in inches)
    plt.figure(figsize=(13, 8))  # Makes it longer/wider

    # Set color cycle to a colorblind-friendly palette
    plt.gca().set_prop_cycle(color=["#FFB000", '#785EF0', '#DC267F', '#FE6100', '#8EAEFF'])

    # Plot each dataset with its own timestep
    for df, label in datasets:
        timestep = df["Step"] / 2  # Divide by 2 for each dataset
        success_rate = df[df.columns[1]]
        plt.plot(timestep, success_rate, label=label)

    # Add dotted vertical line at x=500_000
    plt.axvline(x=500_000, color='gray', linestyle='--', linewidth=2)

    # Set labels and title
    plt.title('Broken Leg - Hard', fontfamily='serif', fontsize=title_size, pad=10)
    plt.xlabel('Step', fontfamily='serif', fontsize=label_size)
    plt.ylabel('Success Rate', fontfamily='serif', fontsize=label_size)
    # Add label "Adaptation Begins" at top right of vertical line
    plt.text(540_000, 0.95, '▶️ Broken Leg Hard', verticalalignment='top', fontfamily='serif', color='#444444', fontsize=tick_size)

    # Set font for tick labels
    plt.xticks(fontfamily='serif', fontsize=tick_size)
    plt.yticks(fontfamily='serif', fontsize=tick_size)

    # Set font of exponents in x-axis to serif and tick_size
    plt.gca().ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    plt.gca().xaxis.get_offset_text().set_fontsize(tick_size)
    plt.gca().xaxis.get_offset_text().set_fontfamily('serif')

    # Add legend
    plt.legend(fontsize=19)
    # Set legend font to serif
    legend = plt.gca().get_legend()
    for text in legend.get_texts():
        text.set_fontfamily('serif')

    # Add grid for better readability
    plt.grid(True, alpha=0.3)

    # Start x at 0
    plt.xlim(left=0)

    # Remove box on top and right
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save figure
    plt.savefig("figures/broken-leg.png", dpi=300, bbox_inches='tight')

    # Show plot (optional)
    plt.show()

def create_plot_2():
    data = pd.read_csv("data/extra/pretrain-adam-30m.csv")
    # Create figure with specified size (width, height in inches)
    plt.figure(figsize=(12, 7))  # Makes it longer/wider

    # Set color cycle to a colorblind-friendly palette
    plt.gca().set_prop_cycle(color=["#FFB000", '#785EF0', '#DC267F', '#FE6100', '#8EAEFF'])

    # Plot data up to step 30_000_000
    # Skip every other point to reduce line thickness
    plt.plot(data["Step"], data[data.columns[1]], label='Batch-Based PPO')
    # Plot y=0 line and label as "Stream-AC with AdaptiveObGD"
    data2 = np.zeros(len(data["Step"]))
    plt.plot(data["Step"], data2, label='Stream-AC with AdaptiveObGD')

    # Set labels and title
    plt.title('Pretraining on AnymalC-Reach-v1', fontfamily='serif', fontsize=title_size, pad=10)
    plt.xlabel('Step', fontfamily='serif', fontsize=label_size)
    plt.ylabel('Success Rate', fontfamily='serif', fontsize=label_size)

    # Set font for tick labels
    plt.xticks(fontfamily='serif', fontsize=tick_size)
    plt.yticks(fontfamily='serif', fontsize=tick_size)

    # Set font of exponents in x-axis to serif and tick_size
    plt.gca().ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    plt.gca().xaxis.get_offset_text().set_fontsize(tick_size)
    plt.gca().xaxis.get_offset_text().set_fontfamily('serif')

    # Add legend
    plt.legend(fontsize=tick_size)
    # Set legend font to serif
    legend = plt.gca().get_legend()
    for text in legend.get_texts():
        text.set_fontfamily('serif')

    # Add grid for better readability
    plt.grid(True, alpha=0.3)

    # Start x at 0
    plt.xlim(left=0, right=30_000_000)

    # Remove box on top and right
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save figure
    plt.savefig("figures/pretrain.png", dpi=300, bbox_inches='tight')

def create_plot3():
    data = pd.read_csv("data/extra/no_warmstart-aobgd-leg-e.csv")
    # Create figure with specified size (width, height in inches)
    plt.figure(figsize=(12, 7))  # Makes it longer/wider

    # Set color cycle to a colorblind-friendly palette
    plt.gca().set_prop_cycle(color=["#785EF0", '#DC267F', '#DC267F', '#FE6100', '#8EAEFF'])

    # Plot data up to step 30_000_000
    plt.plot(data["Step"]/2, data[data.columns[1]], label='Broken Leg - Easy')
    # Plot y=0 line and label as "Stream-AC with AdaptiveObGD"
    data2 = np.zeros(len(data["Step"]))
    plt.plot(data["Step"]/2, data2, label='Broken Leg - Hard')

    # Set labels and title
    plt.title('Stream-AC with AdaptiveObGD - No Warm Start', fontfamily='serif', fontsize=title_size, pad=10)
    plt.xlabel('Step', fontfamily='serif', fontsize=label_size)
    plt.ylabel('Success Rate', fontfamily='serif', fontsize=label_size)

    # Set font for tick labels
    plt.xticks(fontfamily='serif', fontsize=tick_size)
    plt.yticks(fontfamily='serif', fontsize=tick_size)

    # Set font of exponents in x-axis to serif and tick_size
    plt.gca().ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    plt.gca().xaxis.get_offset_text().set_fontsize(tick_size)
    plt.gca().xaxis.get_offset_text().set_fontfamily('serif')

    # Add legend
    plt.legend(fontsize=tick_size)
    # Set legend font to serif
    legend = plt.gca().get_legend()
    for text in legend.get_texts():
        text.set_fontfamily('serif')

    # Add grid for better readability
    plt.grid(True, alpha=0.3)

    # Start x at 0
    plt.xlim(left=0, right=1_500_000)

    # Remove box on top and right
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save figure
    plt.savefig("figures/warmstart.png", dpi=300, bbox_inches='tight')

def create_plot4():
    # Load data from csv
    data = pd.read_csv("data/extra/multieasy-aobgd.csv")
    data2 = pd.read_csv("data/extra/multieasy-aobgdcbp.csv")
    timestep = pd.read_csv("data/extra/multieasy-timestep.csv")

    # For each step in data and data2, find the nearest step in timestep and replace it with the value at that index in timestep
    data["Step"] = data["Step"].apply(lambda x: timestep.iloc[(timestep['Step'] - x).abs().argsort()[:1]][timestep.columns[1]].values[0])
    data2["Step"] = data2["Step"].apply(lambda x: timestep.iloc[(timestep['Step'] - x).abs().argsort()[:1]][timestep.columns[1]].values[0])

    # Extract relevant columns and handle different lengths
    datasets = [
        (data, 'AdaptiveObGD'),
        (data2, 'AdaptiveObGD + CBP'),
    ]

    # Create figure with specified size (width, height in inches)
    plt.figure(figsize=(13, 8))  # Makes it longer/wider

    # Set color cycle to a colorblind-friendly palette
    plt.gca().set_prop_cycle(color=["#967FFC", "#FF60AD"])

    # Plot each dataset with its own timestep
    for df, label in datasets:
        timestep = df["Step"]
        success_rate = df[df.columns[1]]
        plt.plot(timestep, success_rate, label=label)

    # Add dotted vertical line at x=500_000
    plt.axvline(x=500_000, color='gray', linestyle='--', linewidth=2)
    plt.axvline(x=1_500_000, color='gray', linestyle='--', linewidth=2)
    plt.axvline(x=2_500_000, color='gray', linestyle='--', linewidth=2)

    # Set labels and title
    plt.title('Multi-Stage Adaptation', fontfamily='serif', fontsize=title_size, pad=10)
    plt.xlabel('Step', fontfamily='serif', fontsize=label_size)
    plt.ylabel('Success Rate', fontfamily='serif', fontsize=label_size)
    # Add label "Adaptation Begins" at top right of vertical line
    plt.text(540_000, 0.95, '▶️ Broken Leg Easy', verticalalignment='top', fontfamily='serif', color='#444444', fontsize=tick_size-5)
    plt.text(1_540_000, 0.95, '▶️ Slippery Floor Easy', verticalalignment='top', fontfamily='serif', color='#444444', fontsize=tick_size-5)
    plt.text(2_540_000, 0.95, '▶️ Goal Shift Easy', verticalalignment='top', fontfamily='serif', color='#444444', fontsize=tick_size-5)

    # Set font for tick labels
    plt.xticks(fontfamily='serif', fontsize=tick_size)
    plt.yticks(fontfamily='serif', fontsize=tick_size)

    # Set font of exponents in x-axis to serif and tick_size
    plt.gca().ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    plt.gca().xaxis.get_offset_text().set_fontsize (tick_size)
    plt.gca().xaxis.get_offset_text().set_fontfamily('serif')

    # Add legend
    plt.legend(fontsize=19, loc='center right')
    # Set legend font to serif
    legend = plt.gca().get_legend()
    for text in legend.get_texts():
        text.set_fontfamily('serif')

    # Add grid for better readability
    plt.grid(True, alpha=0.3)

    # Start x at 0
    plt.xlim(left=0)

    # Remove box on top and right
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save figure
    plt.savefig("figures/multistage.png", dpi=300, bbox_inches='tight')

    # Show plot (optional)
    plt.show()



if __name__ == "__main__":
    create_plot_1()
    create_plot_2()
    # create_plot3()
    create_plot4()