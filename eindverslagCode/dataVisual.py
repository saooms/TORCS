import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Data laden
df = pd.read_csv('combined_data.csv', header=None)
columns = [
    'gas', 'brake', 'steer', 'speed', 
    'orientation_x', 'orientation_y'
] + [f'sensor_{i}' for i in range(1, 20)]
df.columns = columns

# Beschrijvende statistieken
desc_stats = df.describe().transpose()
desc_stats['skew'] = df.skew()
desc_stats['kurtosis'] = df.kurtosis()

# Export naar LaTeX
print(desc_stats.to_latex(float_format="%.2f", 
                         caption="Beschrijvende statistieken",
                         label="tab:desc_stats"))

# Visualisaties
plt.figure(figsize=(12, 6))

# Snelheidsverdeling
plt.subplot(1, 2, 1)
sns.histplot(df['speed'], kde=True, bins=50)
plt.title('Snelheidsverdeling')
plt.xlabel('Snelheid (km/u)')
plt.savefig('speed_dist.png', dpi=300)

# Stuurhoekverdeling
plt.subplot(1, 2, 2)
sns.histplot(df['steer'], kde=True, bins=50)
plt.title('Stuurhoekverdeling')
plt.xlabel('Stuurhoek (-1 tot 1)')
plt.savefig('steering_dist.png', dpi=300)

# Correlatiematrix
corr_matrix = df.corr()
plt.figure(figsize=(15, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', 
            cmap='coolwarm', center=0)
plt.title('Correlatiematrix')
plt.tight_layout()
plt.savefig('corr_matrix.png', dpi=300)

# Outlier analyse
outliers = {
    'Extreme snelheden': len(df[df['speed'] > 300]),
    'Max sensor waarden': len(df[df.filter(like='sensor').max(axis=1) == 200]),
    'Onmogelijke stuurcombinaties': len(df[(df['gas'] > 0) & (df['brake'] > 0)])
}
outliers_df = pd.DataFrame.from_dict(outliers, orient='index', columns=['Aantal'])
outliers_df['Percentage'] = (outliers_df['Aantal'] / len(df)) * 100

print(outliers_df.to_latex(float_format="%.2f",
                          caption="Outlier analyse",
                          label="tab:outliers"))

# QQ-plots voor normaliteit
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
stats.probplot(df['speed'], plot=plt)
plt.title('QQ-plot Snelheid')

plt.subplot(1, 3, 2)
stats.probplot(df['steer'], plot=plt)
plt.title('QQ-plot Stuurhoek')

plt.subplot(1, 3, 3)
stats.probplot(df['sensor_1'], plot=plt)
plt.title('QQ-plot Sensor 1')
plt.tight_layout()
plt.savefig('qq_plots.png', dpi=300)

# Tijdreeksanalyse (voorbeeld voor eerste 1000 samples)
plt.figure(figsize=(12, 6))
df.head(1000)['speed'].plot(label='Snelheid')
df.head(1000)['steer'].abs().plot(secondary_y=True, color='r', label='Stuurhoek (abs)')
plt.title('Tijdreeks van snelheid en stuurhoek')
plt.savefig('time_series.png', dpi=300)