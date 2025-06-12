#!/usr/bin/env python3
"""
Example usage of the SimulationPlotter class.

This script demonstrates how to use the plotter to generate
plots for simulation data from different runs.

Usage:
    PYTHONPATH=src python plot_example.py [options]
"""

import sys
import os

# Add the src directory to the path to import the plotter
src_dir = os.path.dirname(os.path.dirname(__file__))  # Go up two levels from src/utils to src
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

try:
    from utils.plotter import SimulationPlotter
except ImportError:
    print("Error: Could not import SimulationPlotter.")
    print("Please run with: PYTHONPATH=src python plot_example.py")
    sys.exit(1)

def plot_latest_run(show_plots=False):
    """Plot data from the latest run using config settings."""
    print("=== Plotting Latest Run ===")
    
    plotter = SimulationPlotter(show_plots=show_plots)
    
    if plotter.load_data():
        plotter.get_data_summary()
        plotter.plot_all_figures()
    else:
        print("Failed to load data from latest run.")

def plot_specific_run(run_name: str, show_plots=False):
    """Plot data from a specific run."""
    print(f"=== Plotting Run: {run_name} ===")
    
    plotter = SimulationPlotter(show_plots=show_plots)
    
    if plotter.load_data(run_name):
        plotter.get_data_summary()
        plotter.plot_all_figures()
    else:
        print(f"Failed to load data from run: {run_name}")

def plot_individual_figures(show_plots=False):
    """Generate individual figures separately."""
    print("=== Generating Individual Figures ===")
    
    plotter = SimulationPlotter(show_plots=show_plots)
    
    if plotter.load_data():
        print("Generating Figure 1: Distance and Speed...")
        plotter.plot_distance_and_speed()
        
        print("Generating Figure 2: All Humans Subplots...")
        plotter.plot_all_humans_subplots()
        
        print("Generating Figure 3: Minimum Each Human...")
        plotter.plot_minimum_each_human()
        
        print("Generating Figure 4: Minimum All Humans...")
        plotter.plot_minimum_all_humans()
        
        print("Generating Figure 5: All CBFs...")
        plotter.plot_all_cbfs()
    else:
        print("Failed to load data.")

def plot_individual_subplots(show_plots=False):
    """Generate the 6 individual subplot figures separately."""
    print("=== Generating Individual Subplot Figures ===")
    
    plotter = SimulationPlotter(show_plots=show_plots)
    
    if plotter.load_data():
        plotter.plot_individual_subplots()
    else:
        print("Failed to load data.")

def main():
    """Main function with command line argument handling."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate plots for BiLVL simulation data')
    parser.add_argument('--run-name', '-r', type=str, 
                       help='Name of the run to plot (e.g., "test", "experiment1")')
    parser.add_argument('--individual', '-i', action='store_true',
                       help='Generate individual figures separately')
    parser.add_argument('--subplots', '-s', action='store_true',
                       help='Generate 6 individual subplot figures separately')
    parser.add_argument('--show-now', action='store_true',
                       help='Show plots immediately (default: save only without showing)')
    
    args = parser.parse_args()
    
    if args.subplots:
        plot_individual_subplots(show_plots=args.show_now)
    elif args.individual:
        plot_individual_figures(show_plots=args.show_now)
    elif args.run_name:
        plot_specific_run(args.run_name, show_plots=args.show_now)
    else:
        plot_latest_run(show_plots=args.show_now)

if __name__ == "__main__":
    main()
