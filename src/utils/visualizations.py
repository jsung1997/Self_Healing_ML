import seaborn as sns
import matplotlib.pyplot as plt

def create_policy_plot(accs_dict, filename="policies_corruption.pdf"):
    sns.set(style="whitegrid")
    label_fontsize = 30
    title_fontsize = 30
    tick_fontsize = 30
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(data=[], x=[], y=[], palette='dark', boxprops={'alpha': 0.6})

    palette = sns.color_palette('dark')
    for i, (key, values) in enumerate(accs_dict.items()):
        sns.stripplot(x=[key]*len(values), y=values, jitter=True, marker='o', size=8, alpha=0.7, color=palette[i], ax=ax)

    ax.set_xlabel('Range of values corrupted', fontsize=label_fontsize)
    ax.set_ylabel('Accuracy of policies', fontsize=label_fontsize)
    ax.set_title('Proposed policies based on corruption', fontsize=title_fontsize)
    ax.set_ylim((0.45, 0.9))
    ax.set_xticklabels(["2%", "5%", "10%", "20%", "50%", "75%", "N=7", "N=8"], fontsize=tick_fontsize)
    ax.tick_params(axis='y', labelsize=tick_fontsize)
    ax.grid(False)
    sns.despine()
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
