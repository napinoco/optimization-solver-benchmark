import yaml
from pathlib import Path

def get_config_path(config_name):
    """Get the path to a configuration file."""
    project_root = Path(__file__).parent.parent.parent
    return project_root / "config" / config_name

def load_benchmark_config():
    """Load benchmark configuration from benchmark_config.yaml."""
    config_path = get_config_path("benchmark_config.yaml")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_solvers_config():
    """Load solvers configuration from solvers.yaml."""
    config_path = get_config_path("solvers.yaml")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_config(config_name):
    """Load any configuration file by name."""
    config_path = get_config_path(config_name)
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    # Test script to load and print configurations
    print("Benchmark Config:")
    benchmark_config = load_benchmark_config()
    print(benchmark_config)
    
    print("\nSolvers Config:")
    solvers_config = load_solvers_config()
    print(solvers_config)