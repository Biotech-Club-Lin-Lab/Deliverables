import random

class Genome:
    def __init__(self, id):
        self.id = id
        self.genes = []  # Contains nodes and connections (the genome's structure)
        self.fitness = 0
        self.traits_used = set()  # Tracks traits used in previous generations
        self.history = []  # Stores successful traits and their outcomes

    def mutate(self):
        # Mutate the genome: Add connections, nodes, or modify existing genes
        successful_traits = self.history[-1] if self.history else set()
        trait_probability = 0.8 if successful_traits else 0.2
        
        if random.random() < trait_probability:
            self.mutate_using_successful_trait(successful_traits)
        else:
            self.random_mutate()
    
    def random_mutate(self):
        # Apply a random mutation to the genome
        new_gene = random.choice(['add_node', 'add_connection', 'modify_weight'])
        self.genes.append(new_gene)
        self.traits_used.add(new_gene)
    
    def mutate_using_successful_trait(self, successful_traits):
        if not successful_traits:
            self.random_mutate()
            return
        # Mutate using traits that were successful in the past
        trait = random.choice(list(successful_traits))  # Select a successful trait
        self.genes.append(trait)
        self.traits_used.add(trait)
        
    def evaluate_fitness(self, environment):
        # Evaluate the genome's performance in the environment (task)
        # This can be any task where fitness depends on how well the genome performs
        # For simplicity, we'll simulate fitness based on the number of used traits
        self.fitness = len(self.traits_used) + random.uniform(0, 1)
        
    def record_successful_trait(self, trait):
        # Record which traits were successful
        self.history.append(trait)
        self.traits_used.add(trait)

import matplotlib.pyplot as plt

class Population:
    def __init__(self, size):
        self.size = size
        self.genomes = [Genome(i) for i in range(size)]
        self.best_fitness_history = []
        self.avg_fitness_history = []
    
    def plot_fitness(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.best_fitness_history, label="Best Fitness", color='blue', linewidth=2)
        plt.plot(self.avg_fitness_history, label="Average Fitness", color='green', linestyle='dashed', linewidth=2)
        
        plt.xlabel("Generation")
        plt.ylabel("Fitness Score")
        plt.title("Fitness Progression Over Generations")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.show()

    def evolve(self):
        # Evaluate and evolve the population over generations
        for genome in self.genomes:
            genome.evaluate_fitness(None)  # Pass environment as needed

        # Sort by fitness, keep the best ones, and generate offspring
        self.genomes.sort(key=lambda g: g.fitness, reverse=True)
        new_genomes = self.genomes[:self.size // 2]  # Select top half

        for i in range(self.size // 2, self.size):
            parent = random.choice(new_genomes)  # Randomly select a parent from the best genomes
            offspring = Genome(i)
            offspring.genes = parent.genes[:]  # Copy parent's genes
            offspring.mutate()  # Apply mutation based on strategy
            self.genomes[i] = offspring

        # Now record successful traits for the next generation
        for genome in self.genomes:
            if genome.fitness > 1.5:  # Threshold for success, can be adjusted
                for trait in genome.traits_used:
                    genome.record_successful_trait(trait)  # Store successful traits

            # ✅ Track best & average fitness for visualization
        best_fitness = max(genome.fitness for genome in self.genomes)
        avg_fitness = sum(genome.fitness for genome in self.genomes) / len(self.genomes)

        self.best_fitness_history.append(best_fitness)
        self.avg_fitness_history.append(avg_fitness)

    def run_evolution(self, generations):
        for gen in range(generations):
            print(f"Generation {gen + 1}")
            self.evolve()
            best_fitness = self.best_fitness_history[-1]
            avg_fitness = self.avg_fitness_history[-1]

            print(f"Best fitness history : {best_fitness:.2f}, Avg fitness: {avg_fitness:.2f}")


        self.plot_fitness()

# Run the NEAT-like evolution with goal-oriented mutations
if __name__ == "__main__":
    population_size = 10  # Small size for demonstration
    generations = 20
    
    population = Population(population_size)
    population.run_evolution(generations)
