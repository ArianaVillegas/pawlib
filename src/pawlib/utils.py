"""Utility functions for PAWLib"""
import sys


def print_metrics(metrics, title="Results"):
    """Pretty print metrics with Rich-style formatting.
    
    Args:
        metrics: dict of metric name -> value
        title: title for the table
    """
    try:
        from rich.console import Console
        from rich.table import Table
        
        console = Console()
        table = Table(title=title, show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        
        for key, value in metrics.items():
            if isinstance(value, float):
                if 0 < value < 0.01:
                    value_str = f"{value:.6f}"
                elif value > 100:
                    value_str = f"{value:.2f}"
                else:
                    value_str = f"{value:.4f}"
            else:
                value_str = str(value)
            
            table.add_row(key.replace('_', ' ').title(), value_str)
        
        console.print(table)
        
    except ImportError:
        # Fallback to simple formatting if rich not available
        _print_metrics_simple(metrics, title)


def _print_metrics_simple(metrics, title="Results"):
    """Simple fallback for pretty printing without rich."""
    print("\n" + "=" * 60)
    print(f"{title:^60}")
    print("=" * 60)
    
    max_key_len = max(len(k) for k in metrics.keys())
    
    for key, value in metrics.items():
        if isinstance(value, float):
            if 0 < value < 0.01:
                value_str = f"{value:.6f}"
            elif value > 100:
                value_str = f"{value:.2f}"
            else:
                value_str = f"{value:.4f}"
        else:
            value_str = str(value)
        
        display_key = key.replace('_', ' ').title()
        print(f"{display_key:<{max_key_len+2}}: {value_str}")
    
    print("=" * 60 + "\n")


def print_subset_metrics(subset_results, title="Subset Results"):
    """Pretty print metrics for multiple subsets.
    
    Args:
        subset_results: dict of subset_name -> dict of metrics
        title: title for the output
    """
    try:
        from rich.console import Console
        from rich.table import Table
        
        console = Console()
        
        # Get all unique metrics
        all_metrics = set()
        for metrics in subset_results.values():
            all_metrics.update(metrics.keys())
        all_metrics = sorted(all_metrics)
        
        table = Table(title=title, show_header=True, header_style="bold magenta")
        table.add_column("Subset", style="cyan", no_wrap=True)
        for metric in all_metrics:
            table.add_column(metric.replace('_', ' ').title(), style="green")
        
        for subset_name, metrics in subset_results.items():
            row = [subset_name]
            for metric in all_metrics:
                value = metrics.get(metric, 0.0)
                if isinstance(value, float):
                    if 0 < value < 0.01:
                        value_str = f"{value:.6f}"
                    elif value > 100:
                        value_str = f"{value:.2f}"
                    else:
                        value_str = f"{value:.4f}"
                else:
                    value_str = str(value)
                row.append(value_str)
            table.add_row(*row)
        
        console.print(table)
        
    except ImportError:
        # Fallback to simple formatting
        _print_subset_metrics_simple(subset_results, title)


def _print_subset_metrics_simple(subset_results, title="Subset Results"):
    """Simple fallback for subset metrics without rich."""
    print("\n" + "=" * 80)
    print(f"{title:^80}")
    print("=" * 80)
    
    # Get all unique metrics
    all_metrics = set()
    for metrics in subset_results.values():
        all_metrics.update(metrics.keys())
    all_metrics = sorted(all_metrics)
    
    # Print header
    header = f"{'Subset':<20}"
    for metric in all_metrics:
        header += f"{metric.replace('_', ' ').title():<15}"
    print(header)
    print("-" * 80)
    
    # Print rows
    for subset_name, metrics in subset_results.items():
        row = f"{subset_name:<20}"
        for metric in all_metrics:
            value = metrics.get(metric, 0.0)
            if isinstance(value, float):
                if 0 < value < 0.01:
                    value_str = f"{value:.6f}"
                elif value > 100:
                    value_str = f"{value:.2f}"
                else:
                    value_str = f"{value:.4f}"
            else:
                value_str = str(value)
            row += f"{value_str:<15}"
        print(row)
    
    print("=" * 80 + "\n")


__all__ = ["print_metrics", "print_subset_metrics"]
