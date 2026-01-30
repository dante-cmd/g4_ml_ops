"""
Validate and adjust n_eval_periodos based on available data periods.

This script calculates the maximum number of evaluation periods available
between model_periodo and ult_periodo_data, and adjusts n_eval_periodos
if it exceeds the maximum.

Example:
    model_periodo = 202506
    ult_periodo_data = 202512
    n_eval_periodos = 10
    
    Available periods: 202506, 202507, 202508, 202509, 202510, 202511, 202512
    Max n_eval_periodos = 7
    Adjusted n_eval_periodos = 7
"""

import json
from datetime import datetime
from dateutil.relativedelta import relativedelta


def periodo_to_date(periodo):
    """Convert YYYYMM periodo to datetime object.
    
    Args:
        periodo: Integer in YYYYMM format (e.g., 202506)
    
    Returns:
        datetime object for the first day of that month
    """
    year = periodo // 100
    month = periodo % 100
    return datetime(year, month, 1)


def count_months_between(start_periodo, end_periodo):
    """Count the number of months between two periodos (inclusive).
    
    Args:
        start_periodo: Starting periodo in YYYYMM format
        end_periodo: Ending periodo in YYYYMM format
    
    Returns:
        Number of months from start to end (inclusive)
    """
    start_date = periodo_to_date(start_periodo)
    end_date = periodo_to_date(end_periodo)
    
    if start_date > end_date:
        return 0
    
    # Calculate months difference
    months = 0
    current_date = start_date
    while current_date <= end_date:
        months += 1
        current_date += relativedelta(months=1)
    
    return months


def validate_and_adjust_n_eval_periodos(model_periodo, n_eval_periodos, ult_periodo_data):
    """Validate and adjust n_eval_periodos based on available data.
    
    Args:
        model_periodo: Model training periodo (YYYYMM)
        n_eval_periodos: Requested number of evaluation periods
        ult_periodo_data: Last available data periodo (YYYYMM)
    
    Returns:
        Tuple of (adjusted_n_eval_periodos, max_available_periodos, was_adjusted)
    """
    # Calculate maximum available periods
    max_available = count_months_between(model_periodo, ult_periodo_data)
    
    # Adjust if necessary
    adjusted_n_eval = min(n_eval_periodos, max_available)
    was_adjusted = adjusted_n_eval != n_eval_periodos
    
    return adjusted_n_eval, max_available, was_adjusted


def main():
    """Main function to read param.json, validate, and update if needed."""
    param_file = "param.json"
    
    # Read current parameters
    with open(param_file, 'r') as f:
        params = json.load(f)
    
    model_periodo = params['model_periodo']
    n_eval_periodos = params['n_eval_periodos']
    ult_periodo_data = params['ult_periodo_data']
    
    print(f"Current parameters:")
    print(f"  model_periodo: {model_periodo}")
    print(f"  n_eval_periodos: {n_eval_periodos}")
    print(f"  ult_periodo_data: {ult_periodo_data}")
    print()
    
    # Validate and adjust
    adjusted, max_available, was_adjusted = validate_and_adjust_n_eval_periodos(
        model_periodo, n_eval_periodos, ult_periodo_data
    )
    
    print(f"Validation results:")
    print(f"  Maximum available periods: {max_available}")
    print(f"  Adjusted n_eval_periodos: {adjusted}")
    
    if was_adjusted:
        print(f"  ⚠️  n_eval_periodos was adjusted from {n_eval_periodos} to {adjusted}")
        print(f"     (Data only available from {model_periodo} to {ult_periodo_data})")
        
        # Update param.json
        params['n_eval_periodos'] = adjusted
        with open(param_file, 'w') as f:
            json.dump(params, f, indent=4)
        
        print(f"\n✅ Updated {param_file} with adjusted n_eval_periodos = {adjusted}")
    else:
        print(f"  ✅ n_eval_periodos is valid (within available range)")


if __name__ == "__main__":
    main()
