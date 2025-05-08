import STOM_higgs_tools
import matplotlib.pyplot as plt
import numpy as np

vals = STOM_higgs_tools.generate_data()

# Create histogram
plt.figure(figsize=(10, 6))
bin_heights, bin_edges, patches = plt.hist(vals, bins=30, alpha=0, color='blue', edgecolor='black', range=(104, 155))

# Calculate bin centers and uncertainties
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
uncertainties = np.sqrt(bin_heights)  # Poisson uncertainty is sqrt of count
bin_width = (155 - 104) / 30  # Calculate bin width
x_uncertainties = bin_width / 2  # Horizontal uncertainty is half the bin width

# Plot error bars
plt.errorbar(bin_centers, bin_heights, yerr=uncertainties, xerr=x_uncertainties, fmt='k.', color='black', capsize=3, alpha=0.5)

plt.xlabel('Mass (GeV)')
plt.ylabel('Number of Events')
plt.title('Distribution of Signal and Background Events')
plt.grid(True, alpha=0.3)
plt.show()

