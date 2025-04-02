import numpy as np
import random

# Tournament Selection
def select(population, fitness_scores):
    """Tournament selection with size 3"""
    # Select 3 random individuals
    tournament_size = 3
    indices = random.sample(range(len(population)), tournament_size)
    
    # Find the one with the best fitness
    best_idx = indices[0]
    for idx in indices[1:]:
        if fitness_scores[idx] > fitness_scores[best_idx]:
            best_idx = idx
    
    return population[best_idx].copy()  # Return a copy to prevent mutation of original

# Improved Crossover: Order Crossover (OX) for permutation problems
def crossover(parent1, parent2):
    """Order Crossover (OX) that preserves permutation properties"""
    # Create a new child with placeholders
    child = np.full(256, -1)
    
    # Select two random crossover points
    point1 = random.randint(0, 254)
    point2 = random.randint(point1 + 1, 255)
    
    # Copy the segment between points from parent1 to child
    child[point1:point2+1] = parent1[point1:point2+1]
    
    # Fill in the remaining positions with values from parent2 in order
    parent2_idx = 0
    for i in range(256):
        if child[i] == -1:  # If position is empty
            # Find the next value from parent2 that's not already in child
            while parent2[parent2_idx] in child:
                parent2_idx += 1
                if parent2_idx >= 256:  # Wrap around if needed
                    parent2_idx = 0
            child[i] = parent2[parent2_idx]
            parent2_idx += 1
    
    return child

# Improved Mutation: Multiple mutation types
def mutate(sbox, mutation_rate=0.2):
    """Apply different mutation types with a probability"""
    mutation_type = random.random()
    
    if mutation_type < 0.4:  # 40% chance for swap mutation
        # Swap two random elements
        i, j = random.sample(range(256), 2)
        sbox[i], sbox[j] = sbox[j], sbox[i]
    
    elif mutation_type < 0.7:  # 30% chance for insert mutation
        # Take a random element and insert it at a different position
        i = random.randint(0, 255)
        j = random.randint(0, 255)
        if i != j:
            element = sbox[i]
            # Remove the element and shift
            if i < j:
                sbox[i:j] = sbox[i+1:j+1]
                sbox[j] = element
            else:
                sbox[j+1:i+1] = sbox[j:i]
                sbox[j] = element
    
    else:  # 30% chance for reverse mutation
        # Reverse a random subsequence
        i = random.randint(0, 254)
        j = random.randint(i+1, 255)
        sbox[i:j+1] = np.flip(sbox[i:j+1])
    
    return sbox

# Function to create a diverse initial population
def create_diverse_population(size):
    """Create a diverse initial population with varying properties"""
    population = []
    
    # Create some completely random S-boxes
    for _ in range(size // 2):
        population.append(np.random.permutation(256))
    
    # Create some S-boxes with structured patterns
    for _ in range(size // 4):
        sbox = np.arange(256)
        # Apply some randomization to the structured S-box
        for _ in range(random.randint(50, 150)):
            i, j = random.sample(range(256), 2)
            sbox[i], sbox[j] = sbox[j], sbox[i]
        population.append(sbox)
    
    # Create some S-boxes with locally optimized patterns
    for _ in range(size - len(population)):
        sbox = np.arange(256)
        
        # For each group of 16 elements, create unique mapping
        for chunk in range(16):
            chunk_values = sbox[chunk*16:(chunk+1)*16].copy()
            np.random.shuffle(chunk_values)
            sbox[chunk*16:(chunk+1)*16] = chunk_values
            
        # Apply some additional random swaps
        for _ in range(random.randint(30, 70)):
            i, j = random.sample(range(256), 2)
            sbox[i], sbox[j] = sbox[j], sbox[i]
            
        population.append(sbox)
    
    return population