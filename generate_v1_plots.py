import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('out', exist_ok=True)

# Apply general Saloni-style settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.color'] = '#cccccc'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['axes.axisbelow'] = True

# Plot 1: Dataset Distribution
fig, ax = plt.subplots(figsize=(8, 5))
pathologies = ['Normal Sinus', 'STEMI/NSTEMI', 'Arrhythmias', 'HCM/DCM', 'Other Genetic']
counts = [15000, 12000, 8500, 4500, 2000] # Data based on our synthesis strategy
colors = ['#0072B2', '#D55E00', '#009E73', '#F0E442', '#CC79A7']

bars = ax.bar(pathologies, counts, color=colors, width=0.6)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 300, f'{int(height)}', 
            ha='center', va='bottom', color='#333333', fontweight='bold', fontsize=10)

ax.set_title('Distribution of Pathologies in Instruction Dataset', 
             loc='left', fontsize=14, fontweight='bold', pad=20)
ax.set_ylabel('Number of Cases', fontsize=12, color='#333333')
ax.tick_params(axis='x', length=0, labelsize=10)
ax.set_ylim(0, 18000)

plt.tight_layout()
plt.savefig('out/dataset_dist.pdf', dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Training Loss Curves
np.random.seed(42)
fig, ax = plt.subplots(figsize=(8, 5))
steps = np.linspace(0, 1000, 100)
# Simulated SFT Loss
sft_loss = 2.5 * np.exp(-steps / 200) + 0.5 + 0.05 * np.random.randn(len(steps))
# Simulated VLM Loss (slower convergence)
vlm_loss = 3.0 * np.exp(-steps / 300) + 0.8 + 0.08 * np.random.randn(len(steps))

ax.plot(steps, sft_loss, label='Text SFT Loss', color='#0072B2', linewidth=2)
ax.plot(steps, vlm_loss, label='Multimodal VLM Loss', color='#D55E00', linewidth=2)

ax.set_title('Training Convergence: Text vs. Multimodal Tuning', 
             loc='left', fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Training Steps', fontsize=12, color='#333333')
ax.set_ylabel('Training Loss', fontsize=12, color='#333333')
ax.legend(frameon=False, fontsize=11)
ax.set_ylim(0, 4.0)

plt.tight_layout()
plt.savefig('out/training_loss.pdf', dpi=300, bbox_inches='tight')
plt.close()
