# Evolution System Documentation

This document describes the genetic algorithm and evolution strategies implementation for the car evolution simulation.

## Overview

The evolution system allows cars to evolve over generations, improving their ability to navigate the terrain. Two strategies are implemented:

1. **Genetic Algorithm (GA)**: Traditional evolutionary approach with selection, crossover, and mutation
2. **Evolution Strategies (ES)**: Self-adaptive approach with Gaussian mutation

## Quick Start

### Running the Simulation

```bash
python3 src/app.py
```

### Keyboard Controls

- **R**: Start new generation (manual trigger)
- **G**: Switch to Genetic Algorithm mode
- **T**: Switch to Evolution Strategies mode
- **A/D**: Manual car control (left/right)
- **S**: Brake

## Evolution Strategies

### Genetic Algorithm (GA)

The GA uses the following components:

#### Selection Methods
- **Tournament Selection** (default): Randomly picks k individuals and selects the best
- **Roulette Wheel Selection**: Probability proportional to fitness
- **Rank-Based Selection**: Probability based on rank, not raw fitness

#### Crossover
- **Uniform Crossover**: Randomly swaps genes between parents
- **Blend Crossover**: Creates weighted average of parent genes
- **Vertices Crossover**: Special handling for chassis shape

#### Mutation
- **Gaussian Mutation**: Adds random noise from normal distribution
- **Adaptive Mutation**: Mutation strength based on parameter ranges
- **Vertices Mutation**: Maintains valid convex hull for chassis

#### Elitism
- Top 2 individuals automatically survive to next generation (configurable)

### Evolution Strategies (ES)

The ES implementation includes:

- **Self-Adaptive Mutation**: Each parameter has its own mutation step size
- **Log-Normal Step Size Adaptation**: Step sizes evolve along with parameters
- **(μ+λ) Selection**: Best individuals selected from parents + offspring
- **(μ,λ) Selection**: Best individuals selected only from offspring

## Evolvable Parameters

Each car has 10 evolvable parameters:

| Parameter | Range | Description |
|-----------|-------|-------------|
| wheel_friction | [0.0, 1.0] | Wheel grip on terrain |
| wheel_radius_1 | [0.2, 1.0] | Left wheel size |
| wheel_radius_2 | [0.2, 1.0] | Right wheel size |
| density | [0.5, 5.0] | Chassis mass density |
| wheel_torques[0,1] | [5.0, 50.0] | Motor torque per wheel |
| motor_speeds[0,1] | [-30.0, -5.0] | Target motor speed per wheel |
| hz | [2.0, 15.0] | Suspension spring frequency |
| zeta | [0.1, 0.9] | Suspension damping ratio |
| vertices | [(0,0), (3,3)] | Chassis shape (convex polygon) |

## Fitness Function

Cars are evaluated based on:

- **Primary**: Horizontal distance traveled (X position)
- **Secondary**: Life system prevents idle cars from lingering
  - Starts at 100 frames
  - Decreases if no progress made (< 1e-03 improvement)
  - Car deactivated when life reaches 0

## Configuration

Edit `EvolutionConfig` in `src/app.py` to customize:

```python
self.evolution_config = EvolutionConfig(strategy='ga')
```

### GA Configuration Parameters

```python
# Population
population_size = 10      # Number of cars per generation
elite_size = 2           # Best cars kept unchanged

# Genetic operators
mutation_rate = 0.2      # Probability of mutation
mutation_strength = 0.3  # Mutation magnitude
crossover_rate = 0.7     # Probability of crossover

# Selection
selection_method = 'tournament'  # 'tournament', 'roulette', 'rank'
tournament_size = 3      # Size of tournament
```

### ES Configuration Parameters

```python
es_parents = 5           # μ - number of parents
es_offspring = 10        # λ - number of offspring
es_sigma = 0.2          # Initial mutation step size
es_tau = 0.1            # Learning rate for adaptation
es_selection = 'plus'   # 'plus' for (μ+λ), 'comma' for (μ,λ)
```

## Statistics Tracking

The system tracks and displays:

- **Generation**: Current generation number
- **Strategy**: Current evolution strategy (GA/ES)
- **Best Ever**: Highest fitness achieved across all generations
- **Gen Best**: Best fitness in current generation
- **Gen Mean**: Average fitness in current generation
- **Gen Min**: Lowest fitness in current generation

Statistics are printed to console and displayed in the UI.

## Architecture

### Files

- `src/evolution.py`: Core evolution algorithms
  - `EvolutionConfig`: Configuration dataclass
  - `GeneticAlgorithm`: GA implementation
  - `EvolutionStrategy`: ES implementation
  - `EvolutionManager`: High-level interface

- `src/app.py`: Main simulation
  - Car generation from genes
  - Evolution integration
  - Keyboard controls

- `src/state.py`: Global state management
  - Evolution statistics
  - Current generation

- `src/ui.py`: ImGui user interface
  - Evolution stats panel
  - Car tracking table

### Evolution Flow

```
Generation 0: Random cars
    ↓
Run simulation
    ↓
All cars inactive
    ↓
Extract genes + fitness
    ↓
Evolution Manager
    ↓ (Selection)
Select parents
    ↓ (Crossover/Mutation)
Generate offspring
    ↓
Generation 1: Evolved cars
    ↓
(Repeat)
```

## Tips for Best Results

1. **Let it Run**: Evolution takes many generations (50-100+) to show significant improvement

2. **Choose Strategy**:
   - GA: Better for discrete parameters and complex fitness landscapes
   - ES: Better for continuous optimization and smooth landscapes

3. **Tune Parameters**:
   - High mutation rate: More exploration, slower convergence
   - Low mutation rate: Less exploration, faster convergence (may get stuck)
   - Large elite size: Preserves good solutions but reduces diversity
   - Small elite size: More diversity but may lose good solutions

4. **Watch Statistics**:
   - Mean increasing: Population improving overall
   - Max plateauing: May need more mutation or different strategy
   - Large std dev: High diversity (good early, less important later)

## Future Enhancements

Potential improvements:

- [ ] Add CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
- [ ] Multi-objective optimization (speed + stability + efficiency)
- [ ] Novelty search to encourage diverse solutions
- [ ] Save/load best genomes
- [ ] Visualization of gene distributions over time
- [ ] Island model (multiple populations with migration)
- [ ] Co-evolution (cars and terrain)

## References

- Holland, J.H. (1975). Adaptation in Natural and Artificial Systems
- Rechenberg, I. (1973). Evolutionsstrategie
- Schwefel, H.-P. (1981). Numerical Optimization of Computer Models
- Eiben, A.E. & Smith, J.E. (2015). Introduction to Evolutionary Computing
