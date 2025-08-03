# minte/cli.py
"""Command-line interface for the malaria forecast package."""

import argparse
import json
import sys

from .api import run_forecast


def get_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description="Time Series Forecasting for Malaria Prevalence and Cases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  minte --db-path data.duckdb --predictor prevalence
  
  # With hyperparameter tuning
  minte --db-path data.duckdb --predictor cases --run-tuning
  
  # Using previously tuned parameters
  minte --db-path data.duckdb --predictor cases --use-tuned-parameters
        """
    )
    
    # Data parameters
    parser.add_argument("--db-path", required=True, help="Path to DuckDB database file")
    parser.add_argument("--table-name", default="simulation_results", help="Table name inside DuckDB")
    parser.add_argument("--window-size", default=7, type=int, help="Window size for rolling average")
    parser.add_argument("--param-limit", default="all", help="Maximum parameter_index (exclusive) to include or 'all'")
    parser.add_argument("--sim-limit", default="all", help="Randomly sample this many simulation_index per parameter_index or 'all'")
    parser.add_argument("--min-prevalence", default=0.01, type=float,
                        help="Exclude entire param-sim if the average prevalence is below this threshold.")
    parser.add_argument("--use-cyclical-time", action="store_true",
                        help="Whether to encode timesteps as sin/cos of day_of_year (mod 365).")
    parser.add_argument("--predictor", default="prevalence", choices=["prevalence", "cases"],
                        help="What to predict: 'prevalence' (pfpr) or 'cases'")
    
    # Model parameters
    parser.add_argument("--hidden-size", default=64, type=int, help="Hidden size for RNNs")
    parser.add_argument("--num-layers", default=1, type=int, help="Number of RNN layers")
    parser.add_argument("--dropout", default=0.1, type=float, help="Dropout probability")
    parser.add_argument("--lookback", default=30, type=int,
                        help="Lookback window (sequence length) for RNN inputs")
    
    # Training parameters
    parser.add_argument("--epochs", default=64, type=int, help="Maximum number of training epochs")
    parser.add_argument("--learning-rate", default=1e-3, type=float, help="Initial learning rate")
    parser.add_argument("--weight-decay", default=1e-1, type=float, help="Weight adjustment for learning rate decay")
    parser.add_argument("--batch-size", default=4096, type=int, help="Batch size for training")
    parser.add_argument("--num-workers", default=0, type=int, help="Number of worker processes for DataLoader")
    parser.add_argument("--device", default=None, help="Choose device: 'cuda' or 'cpu'. If None, auto-detect.")
    parser.add_argument("--patience", default=16, type=int, help="Patience for early stopping")
    
    # File paths
    parser.add_argument("--output-dir", default="results", help="Directory to save model checkpoints and plot data")
    parser.add_argument("--use-existing-split", action="store_true", help="Use existing train/val/test split from CSV file")
    parser.add_argument("--split-file", default=None, help="Path to existing train/val/test split CSV file")
    
    # Hyperparameter tuning
    parser.add_argument("--run-tuning", action="store_true", help="Run hyperparameter tuning with Optuna")
    parser.add_argument("--tuning-output-dir", default="results_tuned", help="Directory to save tuning results")
    parser.add_argument("--tuning-timeout", type=int, default=86400, help="Timeout for tuning in seconds (default: 24 hours)")
    parser.add_argument("--tuning-trials", type=int, default=32, help="Number of trials for hyperparameter tuning")
    parser.add_argument("--use-tuned-parameters", action="store_true", 
                        help="Use previously tuned parameters instead of defaults or command line arguments")
    
    # Miscellaneous
    parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility")
    
    return parser


def main():
    """Main function for CLI."""
    parser = get_parser()
    args = parser.parse_args()
    
    # Convert args to dict
    kwargs = vars(args)
    
    try:
        # Run forecast
        results = run_forecast(**kwargs)
        
        # Print summary
        print("\n" + "="*60)
        print("MALARIA FORECAST COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"\nPredictor: {args.predictor}")
        print(f"Output directory: {args.output_dir}")
        
        print("\nModel Performance (Test Set):")
        print("-"*40)
        for model_type in ["gru", "lstm"]:
            metrics = results[model_type]["test_metrics"]
            print(f"\n{model_type.upper()} Model:")
            print(f"  R² Score: {metrics['r2']:.4f}")
            print(f"  RMSE: {metrics['rmse']:.4f}")
            print(f"  MAE: {metrics['mae']:.4f}")
            print(f"  SMAPE: {metrics['smape']:.2f}%")
            
        print("\n" + "="*60)
        
    except Exception as e:
        print(f"\nError: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()