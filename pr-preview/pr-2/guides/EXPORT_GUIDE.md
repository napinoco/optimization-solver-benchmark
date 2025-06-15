# Export Functionality Guide

The optimization solver benchmark system provides comprehensive data export capabilities for research and analysis. This guide explains how to use the various export formats and features.

## Overview

Export functionality includes:
- **CSV exports** for spreadsheet analysis
- **JSON exports** for programmatic access
- **PDF reports** for documentation
- **RESTful API** for real-time data access
- **Data validation** for quality assurance

## Quick Start

### Command Line Export
```bash
# Export all formats
python scripts/reporting/export.py --format all

# Export specific format
python scripts/reporting/export.py --format csv
python scripts/reporting/export.py --format json
python scripts/reporting/export.py --format pdf

# Export with custom output directory
python scripts/reporting/export.py --format all --output-dir my_exports
```

### Python API
```python
from scripts.reporting.export import DataExporter

# Create exporter
exporter = DataExporter()

# Export all formats
results = exporter.export_all_formats()

# Export specific formats
csv_path = exporter.export_solver_comparison_csv()
json_path = exporter.export_json_data(include_raw_results=True)
pdf_path = exporter.generate_summary_report_pdf()
```

### RESTful API Server
```bash
# Start API server
python scripts/api/simple_api.py --host 0.0.0.0 --port 5000

# Access endpoints
curl http://localhost:5000/api/health
curl http://localhost:5000/api/solvers
curl http://localhost:5000/api/results?limit=10
```

## Export Formats

### CSV Exports

#### Solver Comparison CSV
Contains solver performance statistics:
- `solver_name`: Name of the solver
- `total_problems`: Number of problems attempted
- `problems_solved`: Number successfully solved
- `success_rate`: Success rate (0.0 to 1.0)
- `avg_solve_time`: Average solving time in seconds
- `min_solve_time`: Fastest solve time
- `max_solve_time`: Slowest solve time

```python
# Export solver comparison
csv_path = exporter.export_solver_comparison_csv("my_solver_comparison.csv")
```

#### Problem Results CSV
Contains detailed results for each solver-problem combination:
- `benchmark_id`: Unique benchmark run identifier
- `solver_name`: Name of the solver
- `problem_name`: Name of the problem
- `problem_type`: Type (LP, QP, SOCP, SDP)
- `status`: Solution status (optimal, error, timeout, etc.)
- `solve_time`: Time taken to solve
- `objective_value`: Optimal objective value
- `iterations`: Number of solver iterations

```python
# Export all results
csv_path = exporter.export_problem_results_csv("my_results.csv")

# Export filtered results
csv_path = exporter.export_problem_results_csv(
    "scipy_results.csv",
    solver_filter="SciPy"
)
```

### JSON Exports

#### Structured JSON Data
Comprehensive data in machine-readable format:
```json
{
  "metadata": {
    "generated_at": "2025-06-11T20:00:00",
    "total_results": 100,
    "total_solvers": 8,
    "total_problems": 10
  },
  "solver_comparison": [
    {
      "solver_name": "SciPy",
      "success_rate": 0.95,
      "avg_solve_time": 0.0025,
      "problems_solved": 19
    }
  ],
  "problem_statistics": [...],
  "environment_info": {...}
}
```

```python
# Export comprehensive JSON
json_path = exporter.export_json_data("benchmark_data.json")

# Include raw results
json_path = exporter.export_json_data(
    "full_data.json", 
    include_raw_results=True
)
```

### PDF Reports

#### Summary Report
Basic PDF report with:
- Benchmark overview statistics
- Top-performing solvers table
- System information
- Generation metadata

```python
# Generate PDF report
pdf_path = exporter.generate_summary_report_pdf("summary.pdf")
```

**Note**: PDF generation requires `reportlab`. If not available, a text-based report is generated as fallback.

## RESTful API

### Endpoints

#### Health Check
```bash
GET /api/health
```
Returns API status and database information.

#### Solvers
```bash
GET /api/solvers
```
List all solvers with basic statistics.

#### Problems
```bash
GET /api/problems
```
List all problems with solver performance data.

#### Results
```bash
GET /api/results?solver=SciPy&limit=50&offset=0
```
Get benchmark results with filtering and pagination.

**Query Parameters:**
- `solver`: Filter by solver name
- `problem`: Filter by problem name
- `type`: Filter by problem type (LP, QP, etc.)
- `status`: Filter by result status
- `limit`: Results per page (max 1000)
- `offset`: Page offset

#### Summary
```bash
GET /api/summary
```
Get overall benchmark statistics.

#### Solver Details
```bash
GET /api/solver/<solver_name>
```
Detailed information about specific solver.

#### Problem Details
```bash
GET /api/problem/<problem_name>
```
Detailed information about specific problem.

### Example API Usage
```python
import requests

# Get solver list
response = requests.get("http://localhost:5000/api/solvers")
solvers = response.json()

# Get recent results
response = requests.get("http://localhost:5000/api/results?limit=10")
results = response.json()

# Get solver details
response = requests.get("http://localhost:5000/api/solver/SciPy")
scipy_info = response.json()
```

## Data Validation

### Automatic Validation
```python
from scripts.reporting.data_validator import DataValidator

validator = DataValidator()

# Validate CSV export
result = validator.validate_csv_export("solver_comparison.csv")
print(f"Valid: {result.valid}")
print(f"Errors: {result.errors}")

# Validate JSON export
result = validator.validate_json_export("benchmark_data.json")

# Check cross-format consistency
result = validator.validate_cross_format_consistency(
    csv_files=["results.csv"],
    json_path="data.json"
)
```

### Command Line Validation
```bash
# Validate exported files
python scripts/reporting/data_validator.py exports/*.csv exports/*.json

# Check consistency
python scripts/reporting/data_validator.py --check-consistency exports/*

# Verbose output
python scripts/reporting/data_validator.py --verbose exports/benchmark_data.json
```

## Configuration

### Export Configuration
```python
from scripts.reporting.export import ExportConfig

config = ExportConfig(
    output_directory="my_exports",
    include_metadata=True,
    format_numbers=True,
    decimal_places=4,
    validate_data=True
)

exporter = DataExporter(config=config)
```

### API Configuration
```python
from scripts.api.simple_api import BenchmarkAPI

api = BenchmarkAPI(
    database_path="my_database.db",
    debug=True
)

api.run(host="0.0.0.0", port=8080)
```

## Integration Examples

### Spreadsheet Analysis
```python
import pandas as pd

# Load CSV data
df = pd.read_csv("solver_comparison.csv")

# Analyze performance
top_solvers = df.nlargest(5, 'success_rate')
print(top_solvers[['solver_name', 'success_rate', 'avg_solve_time']])

# Create visualizations
import matplotlib.pyplot as plt
df.plot(x='solver_name', y='success_rate', kind='bar')
plt.show()
```

### Research Workflow
```python
# Export for analysis
exporter = DataExporter()
results = exporter.export_all_formats("research_data")

# Validate exports
validator = DataValidator()
for format_name, path in results.items():
    if path.suffix == '.csv':
        validation = validator.validate_csv_export(path)
        print(f"{format_name}: {'✅' if validation.valid else '❌'}")

# Use data in research scripts
import json
with open(results['json_data'], 'r') as f:
    benchmark_data = json.load(f)

solver_stats = benchmark_data['solver_comparison']
```

### Automated Reporting
```bash
#!/bin/bash
# automated_export.sh

# Export latest data
python scripts/reporting/export.py --format all --prefix "daily_$(date +%Y%m%d)"

# Validate exports
python scripts/reporting/data_validator.py exports/daily_*.* --check-consistency

# Upload to research repository
# rsync exports/ user@research-server:/data/benchmarks/
```

## Best Practices

### File Organization
```
exports/
├── daily/          # Daily automated exports
├── research/       # Research-specific exports
├── archives/       # Historical data
└── validation/     # Validation reports
```

### Performance Tips
- Use filtering to reduce export size for large datasets
- Enable data validation for critical exports
- Use JSON format for programmatic access
- Use CSV format for spreadsheet analysis
- Cache API responses for frequently accessed data

### Quality Assurance
1. **Always validate exports** before using in research
2. **Check cross-format consistency** when using multiple formats
3. **Verify against source database** for critical data
4. **Monitor export file sizes** for unexpected changes
5. **Test API endpoints** before production use

## Troubleshooting

### Common Issues

#### Empty CSV Files
```
Problem: CSV files have headers but no data rows
Solution: Check if database contains benchmark results
```

#### JSON Validation Errors
```
Problem: JSON structure validation fails
Solution: Ensure all required sections are present
```

#### PDF Generation Fails
```
Problem: PDF generation not available
Solution: Install reportlab: pip install reportlab
```

#### API Server Won't Start
```
Problem: Flask/CORS import errors
Solution: Install dependencies: pip install flask flask-cors
```

### Debug Mode
```python
# Enable debug logging
import logging
logging.getLogger("data_exporter").setLevel(logging.DEBUG)
logging.getLogger("benchmark_api").setLevel(logging.DEBUG)

# Use debug mode for API
api = BenchmarkAPI(debug=True)
```

### Support
For issues with export functionality:
1. Check logs for detailed error messages
2. Validate database schema and content
3. Ensure all dependencies are installed
4. Test with sample data first
5. Use validation tools to identify problems

This export system provides comprehensive data access for research, analysis, and integration with external tools while maintaining data quality and consistency.