"""
Evolution module implementing Genetic Algorithms and Evolution Strategies for car evolution.

This module provides:
- Genetic Algorithm (GA) with selection, crossover, and mutation
- Evolution Strategies (ES) with self-adaptive parameters
- Multiple selection strategies (tournament, roulette, rank-based)
- Multiple crossover operators (uniform, single-point, blend)
- Adaptive mutation with configurable rates
"""

import random
import numpy as np
from typing import List, Dict, Tuple, Any
from copy import deepcopy
from scipy.spatial import ConvexHull


class EvolutionConfig:
    """Configuration for evolution parameters."""

    def __init__(self, strategy='ga'):
        """
        Initialize evolution configuration.

        Args:
            strategy: 'ga' for Genetic Algorithm, 'es' for Evolution Strategies
        """
        self.strategy = strategy  # 'ga' or 'es'

        # Population parameters
        self.population_size = 10
        self.elite_size = 2  # Number of best individuals to keep unchanged

        # GA parameters
        self.mutation_rate = 0.2  # Probability of mutating each gene
        self.mutation_strength = 0.3  # Strength of mutation (std dev as fraction of range)
        self.crossover_rate = 0.7  # Probability of crossover
        self.selection_method = 'tournament'  # 'tournament', 'roulette', 'rank'
        self.tournament_size = 3

        # ES parameters
        self.es_parents = 5  # μ - number of parents
        self.es_offspring = 10  # λ - number of offspring
        self.es_sigma = 0.2  # Initial mutation step size
        self.es_tau = 0.1  # Learning rate for self-adaptation
        self.es_selection = 'plus'  # 'plus' for (μ+λ), 'comma' for (μ,λ)

        # Gene parameter ranges
        self.gene_ranges = {
            'wheel_friction': (0.0, 1.0),
            'wheel_radius_1': (0.2, 1.0),  # Minimum radius to avoid too-small wheels
            'wheel_radius_2': (0.2, 1.0),
            'density': (0.5, 5.0),
            'wheel_torques': (5.0, 50.0),  # Per wheel
            'motor_speeds': (-30.0, -5.0),  # Per wheel
            'hz': (2.0, 15.0),  # Spring frequency
            'zeta': (0.1, 0.9),  # Damping ratio
        }


class GeneticAlgorithm:
    """Genetic Algorithm implementation for car evolution."""

    def __init__(self, config: EvolutionConfig):
        self.config = config

    def select_parents(self, population: List[Dict], fitness_scores: List[float],
                       num_parents: int) -> List[Dict]:
        """
        Select parents for breeding using configured selection method.

        Args:
            population: List of car genes
            fitness_scores: List of fitness values (higher is better)
            num_parents: Number of parents to select

        Returns:
            List of selected parent genes
        """
        if self.config.selection_method == 'tournament':
            return self._tournament_selection(population, fitness_scores, num_parents)
        elif self.config.selection_method == 'roulette':
            return self._roulette_selection(population, fitness_scores, num_parents)
        elif self.config.selection_method == 'rank':
            return self._rank_selection(population, fitness_scores, num_parents)
        else:
            raise ValueError(f"Unknown selection method: {self.config.selection_method}")

    def _tournament_selection(self, population: List[Dict], fitness_scores: List[float],
                             num_parents: int) -> List[Dict]:
        """Tournament selection: randomly pick k individuals and select the best."""
        parents = []
        for _ in range(num_parents):
            # Randomly select tournament_size individuals
            tournament_indices = random.sample(range(len(population)),
                                             min(self.config.tournament_size, len(population)))
            # Select the best from tournament
            best_idx = max(tournament_indices, key=lambda i: fitness_scores[i])
            parents.append(deepcopy(population[best_idx]))
        return parents

    def _roulette_selection(self, population: List[Dict], fitness_scores: List[float],
                           num_parents: int) -> List[Dict]:
        """Roulette wheel selection: probability proportional to fitness."""
        parents = []
        # Handle negative fitness by shifting
        min_fitness = min(fitness_scores)
        adjusted_fitness = [f - min_fitness + 1e-6 for f in fitness_scores]
        total_fitness = sum(adjusted_fitness)
        probabilities = [f / total_fitness for f in adjusted_fitness]

        for _ in range(num_parents):
            selected_idx = np.random.choice(len(population), p=probabilities)
            parents.append(deepcopy(population[selected_idx]))
        return parents

    def _rank_selection(self, population: List[Dict], fitness_scores: List[float],
                       num_parents: int) -> List[Dict]:
        """Rank-based selection: probability based on rank, not raw fitness."""
        parents = []
        # Create ranking (1 for worst, N for best)
        ranks = np.argsort(np.argsort(fitness_scores)) + 1
        total_rank = sum(ranks)
        probabilities = ranks / total_rank

        for _ in range(num_parents):
            selected_idx = np.random.choice(len(population), p=probabilities)
            parents.append(deepcopy(population[selected_idx]))
        return parents

    def crossover(self, parent1: Dict, parent2: Dict) -> Tuple[Dict, Dict]:
        """
        Perform crossover between two parents to create two offspring.

        Args:
            parent1: First parent gene dictionary
            parent2: Second parent gene dictionary

        Returns:
            Tuple of two offspring gene dictionaries
        """
        if random.random() > self.config.crossover_rate:
            # No crossover, return copies of parents
            return deepcopy(parent1), deepcopy(parent2)

        offspring1 = deepcopy(parent1)
        offspring2 = deepcopy(parent2)

        # Uniform crossover for scalar values
        scalar_keys = ['wheel_friction', 'wheel_radius_1', 'wheel_radius_2',
                       'density', 'hz', 'zeta']

        for key in scalar_keys:
            if random.random() < 0.5:
                offspring1[key], offspring2[key] = offspring2[key], offspring1[key]

        # Blend crossover for list values (wheel_torques, motor_speeds)
        for key in ['wheel_torques', 'motor_speeds']:
            alpha = random.random()  # Blend factor
            for i in range(len(parent1[key])):
                val1 = parent1[key][i]
                val2 = parent2[key][i]
                offspring1[key][i] = alpha * val1 + (1 - alpha) * val2
                offspring2[key][i] = alpha * val2 + (1 - alpha) * val1

        # Inherit wheel_drives from one parent randomly
        if random.random() < 0.5:
            offspring1['wheel_drives'] = deepcopy(parent1['wheel_drives'])
            offspring2['wheel_drives'] = deepcopy(parent2['wheel_drives'])
        else:
            offspring1['wheel_drives'] = deepcopy(parent2['wheel_drives'])
            offspring2['wheel_drives'] = deepcopy(parent1['wheel_drives'])

        # Crossover vertices: blend or swap
        if random.random() < 0.5:
            # Swap vertices
            offspring1['vertices'], offspring2['vertices'] = \
                deepcopy(parent2['vertices']), deepcopy(parent1['vertices'])
        else:
            # Blend vertices (average positions)
            min_len = min(len(parent1['vertices']), len(parent2['vertices']))
            blended1 = []
            blended2 = []
            alpha = random.random()
            for i in range(min_len):
                v1 = np.array(parent1['vertices'][i])
                v2 = np.array(parent2['vertices'][i])
                blended1.append(tuple(alpha * v1 + (1 - alpha) * v2))
                blended2.append(tuple(alpha * v2 + (1 - alpha) * v1))

            # Ensure valid convex hull
            offspring1['vertices'] = self._ensure_valid_vertices(blended1)
            offspring2['vertices'] = self._ensure_valid_vertices(blended2)

        return offspring1, offspring2

    def mutate(self, gene: Dict) -> Dict:
        """
        Mutate a gene with configured mutation rate and strength.

        Args:
            gene: Gene dictionary to mutate

        Returns:
            Mutated gene dictionary
        """
        mutated = deepcopy(gene)

        # Mutate scalar values
        for key, (min_val, max_val) in self.config.gene_ranges.items():
            if key in ['wheel_torques', 'motor_speeds']:
                continue  # Handle lists separately

            if random.random() < self.config.mutation_rate:
                # Gaussian mutation
                current_val = mutated[key]
                range_size = max_val - min_val
                mutation = np.random.normal(0, self.config.mutation_strength * range_size)
                new_val = current_val + mutation
                # Clip to valid range
                mutated[key] = np.clip(new_val, min_val, max_val)

        # Mutate list values
        for key in ['wheel_torques', 'motor_speeds']:
            min_val, max_val = self.config.gene_ranges[key]
            for i in range(len(mutated[key])):
                if random.random() < self.config.mutation_rate:
                    current_val = mutated[key][i]
                    range_size = max_val - min_val
                    mutation = np.random.normal(0, self.config.mutation_strength * range_size)
                    new_val = current_val + mutation
                    mutated[key][i] = np.clip(new_val, min_val, max_val)

        # Mutate wheel_drives (flip with low probability)
        if random.random() < self.config.mutation_rate * 0.5:
            idx = random.randint(0, 1)
            mutated['wheel_drives'][idx] = not mutated['wheel_drives'][idx]

        # Mutate vertices
        if random.random() < self.config.mutation_rate:
            mutated['vertices'] = self._mutate_vertices(mutated['vertices'])

        return mutated

    def _mutate_vertices(self, vertices: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Mutate chassis vertices while maintaining valid convex hull."""
        # Convert to numpy array
        points = np.array(vertices)

        # Add Gaussian noise to points
        noise = np.random.normal(0, self.config.mutation_strength * 0.5, points.shape)
        mutated_points = points + noise

        # Clip to valid range [0, 3.0]
        mutated_points = np.clip(mutated_points, 0.0, 3.0)

        # Ensure valid convex hull
        return self._ensure_valid_vertices(mutated_points.tolist())

    def _ensure_valid_vertices(self, points) -> List[Tuple[float, float]]:
        """Ensure points form a valid convex hull with at least 3 vertices."""
        if isinstance(points, np.ndarray):
            points = points.tolist()

        # Convert to list of tuples if needed
        if points and isinstance(points[0], (list, np.ndarray)):
            points = [tuple(p) for p in points]

        # Need at least 3 points for a hull
        if len(points) < 3:
            # Generate random points
            return self._generate_random_vertices()

        try:
            points_array = np.array(points)
            hull = ConvexHull(points_array)
            return [tuple(points_array[i]) for i in hull.vertices]
        except:
            # If hull creation fails, generate new random vertices
            return self._generate_random_vertices()

    def _generate_random_vertices(self) -> List[Tuple[float, float]]:
        """Generate random valid vertices for chassis."""
        points = np.random.rand(30, 2) * 3.0
        hull = ConvexHull(points)
        return [tuple(points[i]) for i in hull.vertices]

    def evolve_population(self, population: List[Dict], fitness_scores: List[float]) -> List[Dict]:
        """
        Evolve population for one generation using genetic algorithm.

        Args:
            population: Current population of genes
            fitness_scores: Fitness values for each individual

        Returns:
            New population for next generation
        """
        new_population = []

        # Elitism: keep best individuals
        if self.config.elite_size > 0:
            # Sort by fitness (descending)
            sorted_indices = np.argsort(fitness_scores)[::-1]
            for i in range(min(self.config.elite_size, len(population))):
                new_population.append(deepcopy(population[sorted_indices[i]]))

        # Generate offspring to fill the rest of the population
        num_offspring_needed = self.config.population_size - len(new_population)

        while len(new_population) < self.config.population_size:
            # Select two parents
            parents = self.select_parents(population, fitness_scores, 2)

            # Crossover
            offspring1, offspring2 = self.crossover(parents[0], parents[1])

            # Mutation
            offspring1 = self.mutate(offspring1)
            offspring2 = self.mutate(offspring2)

            # Add to new population
            new_population.append(offspring1)
            if len(new_population) < self.config.population_size:
                new_population.append(offspring2)

        return new_population[:self.config.population_size]


class EvolutionStrategy:
    """Evolution Strategies (ES) implementation for car evolution."""

    def __init__(self, config: EvolutionConfig):
        self.config = config
        # Each gene has its own mutation step size (self-adaptive)
        self.step_sizes = {}

    def initialize_step_sizes(self, gene: Dict) -> Dict:
        """Initialize mutation step sizes for a gene."""
        step_sizes = {
            'wheel_friction': self.config.es_sigma,
            'wheel_radius_1': self.config.es_sigma,
            'wheel_radius_2': self.config.es_sigma,
            'density': self.config.es_sigma,
            'hz': self.config.es_sigma,
            'zeta': self.config.es_sigma,
            'wheel_torques': [self.config.es_sigma, self.config.es_sigma],
            'motor_speeds': [self.config.es_sigma, self.config.es_sigma],
            'vertices_sigma': self.config.es_sigma,
        }
        return step_sizes

    def mutate_step_size(self, sigma: float) -> float:
        """Mutate step size using log-normal distribution."""
        tau = self.config.es_tau
        return sigma * np.exp(tau * np.random.normal(0, 1))

    def mutate_with_step_size(self, gene: Dict, step_sizes: Dict) -> Tuple[Dict, Dict]:
        """
        Mutate gene using ES with self-adaptive step sizes.

        Args:
            gene: Gene to mutate
            step_sizes: Current step sizes for each parameter

        Returns:
            Tuple of (mutated_gene, new_step_sizes)
        """
        mutated = deepcopy(gene)
        new_step_sizes = deepcopy(step_sizes)

        # Mutate scalar values
        for key in ['wheel_friction', 'wheel_radius_1', 'wheel_radius_2',
                    'density', 'hz', 'zeta']:
            if key not in self.config.gene_ranges:
                continue

            # Mutate step size
            new_step_sizes[key] = self.mutate_step_size(step_sizes[key])

            # Mutate value using new step size
            min_val, max_val = self.config.gene_ranges[key]
            mutation = np.random.normal(0, new_step_sizes[key] * (max_val - min_val))
            new_val = mutated[key] + mutation
            mutated[key] = np.clip(new_val, min_val, max_val)

        # Mutate list values
        for key in ['wheel_torques', 'motor_speeds']:
            min_val, max_val = self.config.gene_ranges[key]
            for i in range(len(mutated[key])):
                # Mutate step size
                new_step_sizes[key][i] = self.mutate_step_size(step_sizes[key][i])

                # Mutate value
                mutation = np.random.normal(0, new_step_sizes[key][i] * (max_val - min_val))
                new_val = mutated[key][i] + mutation
                mutated[key][i] = np.clip(new_val, min_val, max_val)

        # Mutate vertices
        new_step_sizes['vertices_sigma'] = self.mutate_step_size(step_sizes['vertices_sigma'])
        mutated['vertices'] = self._mutate_vertices(mutated['vertices'],
                                                     new_step_sizes['vertices_sigma'])

        return mutated, new_step_sizes

    def _mutate_vertices(self, vertices: List[Tuple[float, float]], sigma: float) -> List[Tuple[float, float]]:
        """Mutate vertices using Gaussian noise scaled by sigma."""
        points = np.array(vertices)
        noise = np.random.normal(0, sigma * 0.5, points.shape)
        mutated_points = points + noise
        mutated_points = np.clip(mutated_points, 0.0, 3.0)

        # Ensure valid convex hull
        try:
            hull = ConvexHull(mutated_points)
            return [tuple(mutated_points[i]) for i in hull.vertices]
        except:
            # If hull fails, return original
            return vertices

    def evolve_population(self, population: List[Dict], fitness_scores: List[float]) -> List[Dict]:
        """
        Evolve population using Evolution Strategies.

        Args:
            population: Current population
            fitness_scores: Fitness scores

        Returns:
            New population
        """
        # Select μ parents (best individuals)
        sorted_indices = np.argsort(fitness_scores)[::-1]
        parents = [population[i] for i in sorted_indices[:self.config.es_parents]]

        # Initialize step sizes if needed
        if not self.step_sizes:
            for i in range(len(parents)):
                self.step_sizes[i] = self.initialize_step_sizes(parents[i])

        # Generate λ offspring
        offspring = []
        offspring_step_sizes = {}

        for i in range(self.config.es_offspring):
            # Select random parent
            parent_idx = random.randint(0, len(parents) - 1)
            parent = parents[parent_idx]

            # Get or initialize step sizes for this parent
            if parent_idx not in self.step_sizes:
                self.step_sizes[parent_idx] = self.initialize_step_sizes(parent)

            # Mutate
            mutated, new_step_sizes = self.mutate_with_step_size(
                parent, self.step_sizes[parent_idx]
            )
            offspring.append(mutated)
            offspring_step_sizes[i] = new_step_sizes

        # Selection: (μ+λ) or (μ,λ)
        if self.config.es_selection == 'plus':
            # (μ+λ): select from parents + offspring
            combined = parents + offspring
            # We'll need to evaluate offspring fitness, so just return offspring for now
            # The app will evaluate them and call this again
            new_population = offspring[:self.config.population_size]
        else:
            # (μ,λ): select only from offspring
            new_population = offspring[:self.config.population_size]

        # Update step sizes
        self.step_sizes = offspring_step_sizes

        return new_population


class EvolutionManager:
    """Main manager for evolution process."""

    def __init__(self, config: EvolutionConfig = None):
        """Initialize evolution manager with configuration."""
        self.config = config or EvolutionConfig()

        if self.config.strategy == 'ga':
            self.algorithm = GeneticAlgorithm(self.config)
        else:  # 'es'
            self.algorithm = EvolutionStrategy(self.config)

        # Statistics tracking
        self.generation_stats = []

    def evolve(self, population: List[Dict], fitness_scores: List[float]) -> List[Dict]:
        """
        Evolve population for one generation.

        Args:
            population: List of gene dictionaries
            fitness_scores: List of fitness values (distance traveled)

        Returns:
            New population for next generation
        """
        # Record statistics
        self.record_statistics(fitness_scores)

        # Evolve
        new_population = self.algorithm.evolve_population(population, fitness_scores)

        return new_population

    def record_statistics(self, fitness_scores: List[float]):
        """Record statistics for current generation."""
        stats = {
            'min': min(fitness_scores) if fitness_scores else 0,
            'max': max(fitness_scores) if fitness_scores else 0,
            'mean': np.mean(fitness_scores) if fitness_scores else 0,
            'std': np.std(fitness_scores) if fitness_scores else 0,
        }
        self.generation_stats.append(stats)

    def get_latest_stats(self) -> Dict:
        """Get statistics for the most recent generation."""
        if self.generation_stats:
            return self.generation_stats[-1]
        return {'min': 0, 'max': 0, 'mean': 0, 'std': 0}

    def get_all_stats(self) -> List[Dict]:
        """Get all generation statistics."""
        return self.generation_stats
