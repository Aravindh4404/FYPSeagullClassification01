import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style='darkgrid')

df = pd.read_csv('VGG/training_metrics.csv')


fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Loss comparison
for run in df.Run.unique():
    run_data = df[df.Run == run]
    axes[0,0].plot(run_data.Epoch, run_data['Train Loss'], label=f'{run} - Train')
    axes[0,0].plot(run_data.Epoch, run_data['Val Loss'], '--', label=f'{run} - Val')
axes[0,0].set_title('Training vs Validation Loss')
axes[0,0].legend()

# Accuracy comparison
for run in df.Run.unique():
    run_data = df[df.Run == run]
    axes[0,1].plot(run_data.Epoch, run_data['Val Acc (%)'], label=run)
    best_epochs = run_data[run_data['New Best Model'] == 'Yes']
    axes[0,1].scatter(best_epochs.Epoch, best_epochs['Val Acc (%)'], marker='*', s=100)
axes[0,1].set_title('Validation Accuracy Comparison')

# F1 Score trends
for run in df.Run.unique():
    run_data = df[df.Run == run]
    axes[1,0].plot(run_data.Epoch, run_data['Val F1'], label=run)
axes[1,0].set_title('Validation F1 Score Trend')

# ROC-AUC trends
for run in df.Run.unique():
    run_data = df[df.Run == run]
    axes[1,1].plot(run_data.Epoch, run_data['Val ROC-AUC'], label=run)
axes[1,1].set_title('Validation ROC-AUC Trend')

for ax in axes.flatten():
    ax.set_xlabel('Epoch')
    ax.grid(True)

plt.tight_layout()
plt.show()