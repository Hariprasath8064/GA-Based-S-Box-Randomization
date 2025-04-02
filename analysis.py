import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Data tracking lists
fitness_progress = []
nonlinearity_progress = []
avalanche_progress = []
uniformity_progress = []
bic_progress = []

def update_progress(fitness, nonlinearity, avalanche, uniformity, bic):
    """Update the progress tracking lists with new values"""
    fitness_progress.append(fitness)
    nonlinearity_progress.append(nonlinearity)
    avalanche_progress.append(avalanche)
    uniformity_progress.append(uniformity)
    bic_progress.append(bic)

def plot_results():
    if not fitness_progress:
        print("Warning: No data to plot. Make sure evolution process updates tracking metrics.")
        return
    
    # Create a figure with multiple subplots
    fig, axs = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [1, 1]})
    
    # Plot 1: All metrics together (normalized to show trends)
    ax1 = axs[0]
    generations = range(len(fitness_progress))
    
    # Normalize each metric to 0-1 range for comparison
    def normalize(data):
        if max(data) == min(data):
            return [0.5] * len(data)  # Handle flat line case
        return [(x - min(data)) / (max(data) - min(data)) if max(data) > min(data) else 0.5 for x in data]
    
    norm_fitness = normalize(fitness_progress)
    norm_nonlinearity = normalize(nonlinearity_progress)
    norm_avalanche = normalize(avalanche_progress)
    norm_uniformity = [(max(uniformity_progress) - x) / (max(uniformity_progress) - min(uniformity_progress)) 
                       if max(uniformity_progress) > min(uniformity_progress) else 0.5 for x in uniformity_progress]  # Invert as lower is better
    norm_bic = normalize(bic_progress)
    
    ax1.plot(generations, norm_fitness, label='Fitness', color='blue', linewidth=2)
    ax1.plot(generations, norm_nonlinearity, label='Nonlinearity', color='red', linewidth=2)
    ax1.plot(generations, norm_avalanche, label='Avalanche', color='green', linewidth=2)
    ax1.plot(generations, norm_uniformity, label='Differential Uniformity (inverted)', color='orange', linewidth=2)
    ax1.plot(generations, norm_bic, label='BIC', color='purple', linewidth=2)
    
    ax1.set_title('Normalized Metrics Over Generations (showing improvement trends)')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Normalized Score (higher is better)')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 2: Raw metrics with separate axes
    ax2 = axs[1]
    
    # Primary y-axis for fitness
    ax2.plot(generations, fitness_progress, label='Fitness', color='blue', linewidth=2)
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Fitness Score', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # Create a twin axis for nonlinearity
    ax2_nonlin = ax2.twinx()
    ax2_nonlin.plot(generations, nonlinearity_progress, label='Nonlinearity', color='red', linewidth=2)
    ax2_nonlin.set_ylabel('Nonlinearity', color='red')
    ax2_nonlin.tick_params(axis='y', labelcolor='red')
    
    # Create another twin axis for avalanche
    ax2_aval = ax2.twinx()
    ax2_aval.spines['right'].set_position(('outward', 60))
    ax2_aval.plot(generations, avalanche_progress, label='Avalanche', color='green', linewidth=2)
    ax2_aval.set_ylabel('Avalanche', color='green')
    ax2_aval.tick_params(axis='y', labelcolor='green')
    
    # Create legend for all curves
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_nonlin.get_legend_handles_labels()
    lines3, labels3 = ax2_aval.get_legend_handles_labels()
    ax2.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper center')
    
    ax2.set_title('Raw Metrics Over Generations')
    
    plt.tight_layout()
    plt.savefig('ga_performance_detailed.png')
    plt.show()
    
    # Also save a simpler version with just fitness
    plt.figure(figsize=(10, 6))
    plt.plot(generations, fitness_progress, 'b-', linewidth=2)
    plt.title('Fitness Improvement Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Fitness Score')
    plt.grid(True)
    plt.savefig('fitness_progress.png')
    plt.show()

def plot_sbox(sbox):
    """Create a nicer S-Box heatmap visualization"""
    plt.figure(figsize=(10, 8))
    
    # Create a more visually distinct heatmap
    ax = sns.heatmap(
        sbox.reshape(16, 16), 
        cmap='viridis', 
        annot=True,  # Show the values in each cell
        fmt="d",     # Format as integers
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 7}
    )
    
    # Add labels for every other row/column for better readability
    ax.set_xticklabels([f"{i}" if i % 2 == 0 else "" for i in range(16)])
    ax.set_yticklabels([f"{i}" if i % 2 == 0 else "" for i in range(16)])
    
    plt.title("S-Box Heatmap (16x16)", fontsize=14)
    plt.tight_layout()
    plt.savefig('sbox_heatmap.png', dpi=300)
    plt.show()
    
    # Also create a correlation heatmap to visualize S-box properties
    plt.figure(figsize=(10, 8))
    
    # Calculate differences between adjacent values
    diffs = np.zeros((16, 16))
    for i in range(16):
        for j in range(16):
            idx = i * 16 + j
            if j < 15:  # Horizontal neighbor
                diffs[i, j] = abs(int(sbox[idx]) - int(sbox[idx + 1]))
            else:
                diffs[i, j] = abs(int(sbox[idx]) - int(sbox[i * 16]))
    
    # Plot the differences
    sns.heatmap(diffs, cmap='coolwarm', annot=False, linewidths=0.5)
    plt.title("S-Box Adjacent Cell Differences", fontsize=14)
    plt.tight_layout()
    plt.savefig('sbox_differences.png', dpi=300)
    plt.show()

def plot_distribution(sbox):
    """Improved S-Box value distribution plot"""
    plt.figure(figsize=(12, 6))
    
    # Use fewer bins for better grouping
    n_bins = 16  # Instead of 256
    
    # Create a histogram with proper styling
    plt.hist(sbox, bins=n_bins, color='royalblue', edgecolor='black', alpha=0.7)
    plt.title("S-Box Value Distribution", fontsize=14)
    plt.xlabel("S-Box Value Range")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add a kernel density estimate
    try:
        from scipy.stats import gaussian_kde
        density = gaussian_kde(sbox)
        x = np.linspace(0, 255, 200)
        y = density(x) * (len(sbox) * (255/n_bins))  # Scale KDE to match histogram height
        plt.plot(x, y, 'r-', linewidth=2)
    except ImportError:
        pass  # Skip density plot if scipy not available
    
    plt.tight_layout()
    plt.savefig('sbox_distribution.png')
    plt.show()
    
    # Also create a randomness test visualization
    plt.figure(figsize=(10, 6))
    
    # Calculate autocorrelation (a measure of randomness)
    autocorr = np.correlate(sbox, sbox, mode='full')
    autocorr = autocorr[len(autocorr)//2:]  # Take only the second half
    autocorr = autocorr / autocorr[0]  # Normalize
    
    plt.plot(autocorr[:50], 'b-', linewidth=2)  # Plot first 50 lags
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title("S-Box Autocorrelation (lower values after lag 0 indicate better randomness)")
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('sbox_autocorrelation.png')
    plt.show()

def reset_progress():
    """Reset all progress tracking lists"""
    global fitness_progress, nonlinearity_progress, avalanche_progress, uniformity_progress, bic_progress
    fitness_progress = []
    nonlinearity_progress = []
    avalanche_progress = []
    uniformity_progress = []
    bic_progress = []

def analyze_sbox_properties(sbox):
    """Perform additional analysis on the S-box and print results"""
    from collections import Counter
    
    # Check uniqueness (should be 256 unique values for a good S-box)
    unique_count = len(np.unique(sbox))
    print(f"S-Box has {unique_count}/256 unique values")
    
    # Check distribution uniformity
    value_counts = Counter(sbox)
    min_count = min(value_counts.values())
    max_count = max(value_counts.values())
    print(f"Value distribution: min occurrences={min_count}, max occurrences={max_count}")
    
    # Check for patterns
    differences = np.diff(sbox)
    sign_changes = np.sum(np.diff(np.signbit(differences)))
    print(f"Sign changes in differences: {sign_changes} (higher is better)")
    
    # Check statistical properties
    print(f"Mean: {np.mean(sbox):.2f}")
    print(f"Standard deviation: {np.std(sbox):.2f}")
    print(f"Entropy: {calculate_entropy(sbox):.2f}")

def calculate_entropy(values):
    """Calculate Shannon entropy of the values"""
    try:
        from scipy.stats import entropy
        import numpy as np
        
        # Count occurrences of each value
        values, counts = np.unique(values, return_counts=True)
        return entropy(counts)
    except ImportError:
        return 0  # Return 0 if scipy not available