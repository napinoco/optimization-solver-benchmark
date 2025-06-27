# Data Access Guide

The optimization solver benchmark system generates JSON and CSV data files as part of its HTML reporting process. This guide explains how to access and use this data.

## Overview

Data is automatically exported when generating reports:
- **JSON files** - Complete benchmark results in machine-readable format
- **CSV files** - Results in spreadsheet-compatible format  
- **HTML reports** - Interactive web reports with embedded data

**Note**: This system generates static data files alongside HTML reports, not a standalone export utility.

## Quick Start

### Generate Reports with Data
```bash
# Generate complete benchmark reports (includes data files)
python main.py --all
```

### Locate Generated Data
```bash
# Data files are generated in docs/pages/data/
ls docs/pages/data/
# benchmark_results.json  - Complete benchmark results
# benchmark_results.csv   - Results in CSV format  
# summary.json           - Summary statistics
```

### Access Data Files
```python
import json
import pandas as pd

# Load JSON data
with open('docs/pages/data/benchmark_results.json', 'r') as f:
    results = json.load(f)

# Load CSV data  
df = pd.read_csv('docs/pages/data/benchmark_results.csv')

# Load summary statistics
with open('docs/pages/data/summary.json', 'r') as f:
    summary = json.load(f)
```

## Available Data Files

### benchmark_results.json
Complete benchmark results in JSON format:
```json
{
  "results": [
    {
      "solver_name": "SciPy linprog",
      "problem_name": "control1", 
      "problem_type": "SDP",
      "status": "optimal",
      "solve_time": 0.045,
      "objective_value": -1.234,
      "library_source": "control family"
    }
  ],
  "metadata": {
    "generated_at": "2025-06-27T01:00:00",
    "git_commit": "d12a85a...", 
    "total_problems": 142,
    "total_solvers": 9
  }
}
```

### benchmark_results.csv  
Tabular format suitable for spreadsheet analysis:
- `solver_name` - Name of the solver
- `problem_name` - Name of the problem
- `problem_type` - Type (LP, QP, SOCP, SDP)
- `status` - Solution status (optimal, infeasible, error, timeout)
- `solve_time` - Time taken to solve (seconds)
- `objective_value` - Optimal objective value (if available)
- `library_source` - Problem origin (e.g., "control family", "DIMACS")

### summary.json
Aggregated statistics and solver comparison:
```json
{
  "solver_performance": {
    "SciPy linprog": {
      "total_problems": 12,
      "success_rate": 1.0,
      "avg_solve_time": 0.003
    }
  },
  "problem_type_breakdown": {
    "LP": 12, 
    "QP": 6,
    "SOCP": 31,
    "SDP": 93
  }
}
```

## Data Analysis Examples

### Basic Statistics
```python
import pandas as pd

# Load results
df = pd.read_csv('docs/pages/data/benchmark_results.csv')

# Success rate by solver
success_rates = df.groupby('solver_name')['status'].apply(
    lambda x: (x == 'optimal').mean()
)
print("Success Rates:")
print(success_rates.sort_values(ascending=False))

# Average solve time by problem type
avg_times = df[df['status'] == 'optimal'].groupby('problem_type')['solve_time'].mean()
print("\nAverage Solve Times by Problem Type:")
print(avg_times)
```

### Solver Comparison
```python
import json
import pandas as pd

# Load summary data
with open('docs/pages/data/summary.json', 'r') as f:
    summary = json.load(f)

# Convert solver performance to DataFrame
solver_df = pd.DataFrame(summary['solver_performance']).T
solver_df = solver_df.sort_values('success_rate', ascending=False)

print("Top Performing Solvers:")
print(solver_df[['success_rate', 'avg_solve_time']].head())
```

### Problem Analysis
```python
# Load full results
df = pd.read_csv('docs/pages/data/benchmark_results.csv')

# Analyze by library source
library_stats = df.groupby('library_source').agg({
    'problem_name': 'count',
    'status': lambda x: (x == 'optimal').mean(),
    'solve_time': 'mean'
}).round(3)
library_stats.columns = ['total_problems', 'success_rate', 'avg_solve_time']

print("Performance by Problem Library:")
print(library_stats)
```

## Visualization Examples

### Create Charts
```python
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('docs/pages/data/benchmark_results.csv')

# Success rate by solver
success_rates = df.groupby('solver_name')['status'].apply(
    lambda x: (x == 'optimal').mean()
)

plt.figure(figsize=(10, 6))
success_rates.plot(kind='bar')
plt.title('Solver Success Rates')
plt.ylabel('Success Rate')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('solver_success_rates.png')
plt.show()
```

### Performance Comparison
```python
# Solve time distribution for successful solves
successful_results = df[df['status'] == 'optimal']

plt.figure(figsize=(12, 6))
for solver in successful_results['solver_name'].unique():
    solver_times = successful_results[successful_results['solver_name'] == solver]['solve_time']
    plt.hist(solver_times, alpha=0.6, label=solver, bins=20)

plt.xlabel('Solve Time (seconds)')
plt.ylabel('Frequency')
plt.title('Solve Time Distribution by Solver')
plt.legend()
plt.yscale('log')
plt.tight_layout()
plt.savefig('solve_time_distribution.png')
plt.show()
```

## Integration with Research Workflows

### Export for External Analysis
```bash
# Generate fresh benchmark data
python main.py --all

# Copy data files to analysis directory
cp docs/pages/data/*.json analysis/
cp docs/pages/data/*.csv analysis/

# Process with your analysis tools
cd analysis/
python my_analysis_script.py
```

### Automated Data Collection
```bash
#!/bin/bash
# collect_benchmark_data.sh

# Generate reports
python main.py --all --library_names DIMACS,SDPLIB

# Archive results with timestamp  
timestamp=$(date +%Y%m%d_%H%M%S)
mkdir -p "archives/$timestamp"
cp docs/pages/data/* "archives/$timestamp/"

echo "Benchmark data archived to archives/$timestamp/"
```

## Data Format Notes

### JSON Structure
- **results**: Array of individual benchmark results
- **metadata**: Information about the benchmark run
- **All times in seconds**: Solve times are floating-point seconds
- **Status values**: "optimal", "infeasible", "unbounded", "error", "timeout"

### CSV Compatibility
- **Standard CSV format**: Compatible with Excel, R, Python pandas
- **UTF-8 encoding**: Handles special characters in problem names
- **Missing values**: Empty cells for unavailable data (e.g., objective_value for failed solves)

## Troubleshooting

### Common Issues

**No data files found**
```bash
# Ensure reports were generated
python main.py --all
ls docs/pages/data/
```

**Empty or incomplete data**
```bash
# Check if benchmarks ran successfully
python main.py --validate
python main.py --benchmark --verbose
```

**Data doesn't match HTML reports**
```bash
# Regenerate both together
python main.py --all
```

### Data Quality Checks
```python
import pandas as pd

df = pd.read_csv('docs/pages/data/benchmark_results.csv')

# Check for missing data
print("Missing data summary:")
print(df.isnull().sum())

# Verify solve times are positive
negative_times = df[df['solve_time'] < 0]
if len(negative_times) > 0:
    print("Warning: Found negative solve times")
    print(negative_times)

# Check status values
valid_statuses = ['optimal', 'infeasible', 'unbounded', 'error', 'timeout']
invalid_statuses = df[~df['status'].isin(valid_statuses)]
if len(invalid_statuses) > 0:
    print("Warning: Found invalid status values")
    print(invalid_statuses['status'].unique())
```

## Best Practices

1. **Regenerate data before analysis** - Always run `python main.py --all` for fresh data
2. **Check timestamps** - Verify metadata timestamps match your expectations
3. **Handle missing values** - Some solvers may not provide objective values or iteration counts
4. **Filter by status** - Focus on 'optimal' results for performance analysis
5. **Archive results** - Keep historical data for trend analysis

This data access system provides the benchmark results in standard formats suitable for research analysis, statistical evaluation, and integration with external tools.