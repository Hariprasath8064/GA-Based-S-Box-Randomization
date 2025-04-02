from ga_core import evolve, initialize_population
from analysis import plot_results, plot_sbox, plot_distribution
import numpy as np
import time
import argparse

def main():
    #  command line arguments
    parser = argparse.ArgumentParser(description='S-Box Generation using Genetic Algorithm')
    parser.add_argument('--population', type=int, default=100, help='Population size')
    parser.add_argument('--generations', type=int, default=1000, help='Number of generations')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Configuration
    population_size = args.population
    generations = args.generations
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    random_seed = args.seed
    
    print("S-Box Generation Using Genetic Algorithm")
    print("=======================================")
    print(f"Population Size: {population_size}")
    print(f"Generations: {generations}")
    print(f"Random Seed: {random_seed}")
    print("---------------------------------------")
    
    start_time = time.time()
    
    print("Initializing Population...")
    population = initialize_population(population_size)

    print("Evolving Population...")
    best_sbox, best_score = evolve(population, generations)

    elapsed_time = time.time() - start_time
    
    print("\nEvolution completed in {:.2f} seconds".format(elapsed_time))
    print(f"\nBest Fitness: {best_score:.8f}")

    print("\nGenerating plots...")
    plot_results()
    plot_sbox(best_sbox)
    plot_distribution(best_sbox)
    
    
    np.save('best_sbox.npy', best_sbox)
    
    
    np.savetxt('best_sbox.csv', best_sbox.reshape(16, 16), delimiter=',', fmt='%d')
    
    print("\nDone! Plots and S-box have been saved to files.")
    print("- best_sbox.npy: NumPy binary format")
    print("- best_sbox.csv: CSV format (16x16 matrix)")
    print("- Various plot images (PNG files)")

if __name__ == "__main__":
    main()