import numpy as np
import random
from multiprocessing import Pool
from fitness import fitness, compute_metrics
from selection import select, crossover, mutate, create_diverse_population
from analysis import update_progress, reset_progress, analyze_sbox_properties
import time

def initialize_population(size):
    """Initialize population with better diversity"""
    return create_diverse_population(size)

def parallel_fitness(population):
    """Calculate fitness for the population in parallel"""
    return [fitness(ind) for ind in population]

def evolve(population, generations, stagnation_limit=50):
    """
    Evolve the population with improved strategies:
    - Dynamic mutation rates
    - Re-diversification when stagnated
    - Island model with migration
    """
    best_score = -float('inf')
    best_solution = None
    stagnation_counter = 0
    last_improvement = 0
    
    # Reset progress tracking
    reset_progress()
    
    # Track execution time
    start_time = time.time()
    
    # Create islands (subpopulations)
    n_islands = 3
    island_size = len(population) // n_islands
    islands = [population[i:i+island_size] for i in range(0, len(population), island_size)]
    
    for gen in range(generations):
        all_fitness_scores = []
        new_islands = []
        
        # Evolve each island separately
        for island_idx, island in enumerate(islands):
            fitness_scores = parallel_fitness(island)
            all_fitness_scores.extend(fitness_scores)
            
            # Calculate statistics for this island
            avg_fitness = sum(fitness_scores) / len(fitness_scores)
            
            # Elitism: Keep top individuals
            elite_percent = 0.1
            top_k = max(1, int(elite_percent * len(island)))
            elite_indices = np.argsort(fitness_scores)[-top_k:]
            new_island = [island[i].copy() for i in elite_indices]
            
            # Adaptive mutation rate based on stagnation
            mutation_rate = 0.2 + (stagnation_counter / stagnation_limit) * 0.3
            mutation_rate = min(0.5, mutation_rate)  # Cap at 50%
            
            # Create offspring until island is filled
            while len(new_island) < len(island):
                # Parent selection
                parent1 = select(island, fitness_scores)
                parent2 = select(island, fitness_scores)
                
                # Crossover
                child = crossover(parent1, parent2)
                
                # Mutation with adaptive rate
                if random.random() < mutation_rate:
                    mutate(child)
                
                new_island.append(child)
            
            new_islands.append(new_island)
        
        # Periodic migration between islands (every 10 generations)
        if gen % 10 == 0 and gen > 0:
            # For each island, swap some individuals with other islands
            migrants_per_island = max(1, island_size // 10)
            
            for i in range(n_islands):
                for j in range(n_islands):
                    if i != j:
                        # Select random individuals to migrate
                        migrant_indices_i = random.sample(range(len(new_islands[i])), migrants_per_island)
                        migrant_indices_j = random.sample(range(len(new_islands[j])), migrants_per_island)
                        
                        # Swap individuals
                        for idx_i, idx_j in zip(migrant_indices_i, migrant_indices_j):
                            new_islands[i][idx_i], new_islands[j][idx_j] = new_islands[j][idx_j].copy(), new_islands[i][idx_i].copy()
        
        # Flatten islands back to one population
        population = []
        for island in new_islands:
            population.extend(island)
        islands = new_islands
            
        # Find best individual in current generation
        all_fitness_scores = parallel_fitness(population)
        best_index = np.argmax(all_fitness_scores)
        current_best = all_fitness_scores[best_index]
        current_best_sbox = population[best_index]
        
        # Calculate metrics for the best individual
        nonlinearity, avalanche, uniformity, bic = compute_metrics(current_best_sbox)
        
        # Update progress tracking
        update_progress(current_best, nonlinearity, avalanche, uniformity, bic)
        
        # Check for improvement
        if current_best > best_score:
            improvement = ((current_best - best_score) / abs(best_score)) * 100 if best_score != 0 else 100
            best_score = current_best
            best_solution = current_best_sbox.copy()
            last_improvement = gen
            stagnation_counter = 0
            
            # Print detailed info for significant improvements
            if improvement > 0.01 or gen % 100 == 0 or gen == 0 or gen == generations - 1:
                elapsed_time = time.time() - start_time
                print(f'Generation {gen}/{generations} ({elapsed_time:.1f}s): Best Fitness = {best_score:.8f}')
                print(f'  Nonlinearity: {nonlinearity:.2f}, Avalanche: {avalanche:.2f}, Uniformity: {uniformity:.2f}, BIC: {bic:.2f}')
        else:
            stagnation_counter += 1
        
        # Re-diversification if stagnated too long
        if stagnation_counter >= stagnation_limit:
            print(f"Generation {gen}: Stagnation detected! Re-diversifying population...")
            
            # Keep the best 10% individuals
            top_indices = np.argsort(all_fitness_scores)[-int(0.1 * len(population)):]
            elite = [population[i].copy() for i in top_indices]
            
            # Generate new diverse individuals for the rest
            new_diverse = create_diverse_population(len(population) - len(elite))
            
            # Combine elite with new diverse individuals
            population = elite + new_diverse
            
            # Reset islands
            islands = [population[i:i+island_size] for i in range(0, len(population), island_size)]
            
            # Reset stagnation counter
            stagnation_counter = 0
    
    print(f"\nEvolution completed. Best solution found at generation {last_improvement}")
    analyze_sbox_properties(best_solution)
    
    return best_solution, best_score