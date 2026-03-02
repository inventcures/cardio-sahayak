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

# Plot 1: Reduction in Errors
fig, ax = plt.subplots(figsize=(8, 4.5))
categories = ['Unassisted Cardiologists', 'LLM-Assisted Cardiologists']
values = [24.3, 13.1]
colors = ['#D55E00', '#0072B2'] # Vermillion and Blue (colorblind friendly)

bars = ax.barh(categories, values, color=colors, height=0.5)

# Direct labeling
for bar in bars:
    width = bar.get_width()
    ax.text(width - 1.5, bar.get_y() + bar.get_height()/2, f'{width}%', 
            ha='right', va='center', color='white', fontweight='bold')

ax.set_title('LLM Assistance Reduces Clinically Significant Errors', 
             loc='left', fontsize=15, fontweight='bold', pad=20)
ax.set_xlabel('Percentage of Cases with Clinically Significant Errors (%)', fontsize=12, color='#333333')
ax.tick_params(axis='y', length=0, labelsize=12) 
ax.set_xlim(0, 30)

plt.tight_layout()
plt.savefig('out/error_reduction.pdf', dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Omission Reduction
fig, ax = plt.subplots(figsize=(8, 4.5))
values = [37.4, 17.8]

bars = ax.barh(categories, values, color=colors, height=0.5)

for bar in bars:
    width = bar.get_width()
    ax.text(width - 2, bar.get_y() + bar.get_height()/2, f'{width}%', 
            ha='right', va='center', color='white', fontweight='bold')

ax.set_title('LLM Assistance Halves Missing Content/Omissions', 
             loc='left', fontsize=15, fontweight='bold', pad=20)
ax.set_xlabel('Percentage of Cases with Omissions (%)', fontsize=12, color='#333333')
ax.tick_params(axis='y', length=0, labelsize=12)
ax.set_xlim(0, 45)

plt.tight_layout()
plt.savefig('out/omission_reduction.pdf', dpi=300, bbox_inches='tight')
plt.close()

# Plot 3: Management Preference
fig, ax = plt.subplots(figsize=(8, 5))
categories = ['LLM-Assisted\nPreferred', 'Unassisted\nPreferred', 'Tie']
values = [45.8, 29.9, 24.3] # 100 - 45.8 - 29.9 = 24.3
colors = ['#0072B2', '#D55E00', '#999999']

bars = ax.bar(categories, values, color=colors, width=0.5)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 1, f'{height}%', 
            ha='center', va='bottom', color='#333333', fontweight='bold', fontsize=12)

ax.set_title('Subspecialists Prefer Management Plans by LLM-Assisted Cardiologists', 
             loc='left', fontsize=15, fontweight='bold', pad=20)
ax.set_ylabel('Percentage of Cases (%)', fontsize=12, color='#333333')
ax.tick_params(axis='x', length=0, labelsize=12)
ax.set_ylim(0, 55)

plt.tight_layout()
plt.savefig('out/management_preference.pdf', dpi=300, bbox_inches='tight')
plt.close()
