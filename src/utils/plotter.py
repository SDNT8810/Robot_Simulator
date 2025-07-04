"""
Plotter utility for BiLVL simulation data visualization.

This module provides functionality to plot simulation data from CSV files
generated by main.py runs, creating various safety and performance plots.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yaml
from typing import Optional, List, Tuple
import seaborn as sns

# Set matplotlib style for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class SimulationPlotter:
    """Class to handle plotting of simulation data."""
    
    def __init__(self, config_path: str = "config/default_config.yaml", 
                 plots_dir: str = "plots", show_plots: bool = True):
        """
        Initialize the plotter.
        
        Args:
            config_path: Path to the configuration file
            plots_dir: Directory to save plots
            show_plots: Whether to display plots interactively (default: True)
        """
        self.config_path = config_path
        self.plots_dir = plots_dir
        self.show_plots = show_plots
        self.config = self._load_config()
        self.data = None
        self.run_name = None
        
        # Create plots directory if it doesn't exist
        os.makedirs(self.plots_dir, exist_ok=True)
    
    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            print(f"Config file not found: {self.config_path}")
            return {}
    
    def load_data(self, run_name: Optional[str] = None) -> bool:
        """
        Load simulation data from CSV file.
        
        Args:
            run_name: Name of the run. If None, uses config or 'test' as default
            
        Returns:
            True if data loaded successfully, False otherwise
        """
        if run_name is None:
            run_name = self.config.get('logging', {}).get('run_name', 'test')
        
        self.run_name = run_name
        csv_path = f"runs/run_{run_name}.csv"
        
        try:
            self.data = pd.read_csv(csv_path)
            print(f"Loaded data from {csv_path}")
            print(f"Data shape: {self.data.shape}")
            return True
        except FileNotFoundError:
            print(f"Data file not found: {csv_path}")
            return False
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def _get_human_count(self) -> int:
        """Get the number of humans in the simulation."""
        if self.data is None:
            return 0
        
        # Count unique human indices from column names
        human_cols = [col for col in self.data.columns if 'human' in col]
        human_indices = set()
        for col in human_cols:
            # Extract human index from column name like 'h_distance_human0'
            parts = col.split('human')
            if len(parts) > 1:
                try:
                    idx = int(parts[1])
                    human_indices.add(idx)
                except ValueError:
                    continue
        
        return len(human_indices)
    
    def _show_or_close(self) -> None:
        """Show plot if show_plots is True, otherwise just close it."""
        if self.show_plots:
            plt.show()
        else:
            plt.close()
    
    def plot_distance_and_speed(self) -> None:
        """Figure 1: Plot h_distance and h_speed vs time."""
        if self.data is None:
            print("No data loaded. Call load_data() first.")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot h_distance
        ax1.plot(self.data['time'], self.data['h_distance'], 'b-', linewidth=2, label='h_distance')
        ax1.set_ylabel('h_distance', fontsize=12)
        ax1.set_title('Distance Barrier Function vs Time', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot h_speed
        ax2.plot(self.data['time'], self.data['h_speed'], 'r-', linewidth=2, label='h_speed')
        ax2.set_xlabel('Time (s)', fontsize=12)
        ax2.set_ylabel('h_speed', fontsize=12)
        ax2.set_title('Speed Barrier Function vs Time', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/figure1_distance_speed.png", dpi=600, bbox_inches='tight')
        self._show_or_close()
        print(f"Saved Figure 1 to {self.plots_dir}/figure1_distance_speed.png")
    
    def plot_all_humans_subplots(self) -> None:
        """Figure 2: Plot all h_distance_human{i}, h_yielding_human{i}, h_speed_human{i}, h_accel_human{i} in subplots."""
        if self.data is None:
            print("No data loaded. Call load_data() first.")
            return
        
        num_humans = self._get_human_count()
        if num_humans == 0:
            print("No human data found in the dataset.")
            return
        
        # Create subplots: 4 rows (distance, yielding, speed, accel) x num_humans columns
        fig, axes = plt.subplots(4, num_humans, figsize=(5*num_humans, 16))
        
        # If only one human, make axes 2D
        if num_humans == 1:
            axes = axes.reshape(-1, 1)
        
        metrics = ['distance', 'yielding', 'speed', 'accel']
        colors = ['blue', 'green', 'red', 'orange']
        
        for i in range(num_humans):
            for j, metric in enumerate(metrics):
                col_name = f'h_{metric}_human{i}'
                if col_name in self.data.columns:
                    axes[j, i].plot(self.data['time'], self.data[col_name], 
                                  color=colors[j], linewidth=2, label=f'Human {i}')
                    axes[j, i].set_title(f'h_{metric}_human{i}', fontweight='bold')
                    axes[j, i].grid(True, alpha=0.3)
                    axes[j, i].legend()
                    
                    if j == 3:  # Last row
                        axes[j, i].set_xlabel('Time (s)')
                    if i == 0:  # First column
                        axes[j, i].set_ylabel(f'h_{metric}')
        
        plt.suptitle('All Human Barrier Functions vs Time', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/figure2_all_humans_subplots.png", dpi=600, bbox_inches='tight')
        self._show_or_close()
        print(f"Saved Figure 2 to {self.plots_dir}/figure2_all_humans_subplots.png")
    
    def plot_minimum_each_human(self) -> None:
        """Figure 3: Plot minimum of each human h in one subplot (3 subplots in one figure)."""
        if self.data is None:
            print("No data loaded. Call load_data() first.")
            return
        
        num_humans = self._get_human_count()
        if num_humans == 0:
            print("No human data found in the dataset.")
            return
        
        # Calculate minimum h values for each human across different metrics
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        metrics = ['distance', 'speed', 'accel']
        colors = plt.cm.Set1(np.linspace(0, 1, num_humans))
        
        for j, metric in enumerate(metrics):
            for i in range(num_humans):
                col_name = f'h_{metric}_human{i}'
                if col_name in self.data.columns:
                    axes[j].plot(self.data['time'], self.data[col_name], 
                               color=colors[i], linewidth=2, label=f'Human {i}')
            
            axes[j].set_title(f'Minimum h_{metric} for Each Human', fontweight='bold')
            axes[j].set_xlabel('Time (s)')
            axes[j].set_ylabel(f'h_{metric}')
            axes[j].grid(True, alpha=0.3)
            axes[j].legend()
        
        plt.suptitle('Minimum Barrier Functions for Each Human', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/figure3_minimum_each_human.png", dpi=600, bbox_inches='tight')
        self._show_or_close()
        print(f"Saved Figure 3 to {self.plots_dir}/figure3_minimum_each_human.png")
    
    def plot_minimum_all_humans(self) -> None:
        """Figure 4: Plot minimum h of all humans in one figure."""
        if self.data is None:
            print("No data loaded. Call load_data() first.")
            return
        
        num_humans = self._get_human_count()
        if num_humans == 0:
            print("No human data found in the dataset.")
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        metrics = ['distance', 'yielding', 'speed', 'accel']
        colors = ['blue', 'green', 'red', 'orange']
        
        for j, metric in enumerate(metrics):
            # Find minimum across all humans for each metric at each time step
            min_values = []
            for _, row in self.data.iterrows():
                human_values = []
                for i in range(num_humans):
                    col_name = f'h_{metric}_human{i}'
                    if col_name in self.data.columns:
                        human_values.append(row[col_name])
                
                if human_values:
                    min_values.append(min(human_values))
                else:
                    min_values.append(np.nan)
            
            ax.plot(self.data['time'], min_values, color=colors[j], 
                   linewidth=2, label=f'min h_{metric}')
        
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Minimum h value', fontsize=12)
        ax.set_title('Minimum Barrier Functions Across All Humans', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/figure4_minimum_all_humans.png", dpi=600, bbox_inches='tight')
        self._show_or_close()
        print(f"Saved Figure 4 to {self.plots_dir}/figure4_minimum_all_humans.png")
    
    def plot_all_cbfs(self) -> None:
        """Figure 5: Plot all CBFs in one figure."""
        if self.data is None:
            print("No data loaded. Call load_data() first.")
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Find all CBF columns
        cbf_columns = [col for col in self.data.columns if col.startswith('cbf_')]
        
        if not cbf_columns:
            print("No CBF columns found in the dataset.")
            return
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(cbf_columns)))
        
        for i, col in enumerate(cbf_columns):
            ax.plot(self.data['time'], self.data[col], color=colors[i], 
                   linewidth=2, label=col.replace('cbf_', 'CBF '))
        
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('CBF Value', fontsize=12)
        ax.set_title('All Control Barrier Functions vs Time', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/figure5_all_cbfs.png", dpi=600, bbox_inches='tight')
        self._show_or_close()
        print(f"Saved Figure 5 to {self.plots_dir}/figure5_all_cbfs.png")
    
    def plot_individual_subplots(self) -> None:
        """Generate all 6 individual subplot figures separately."""
        print("Generating individual subplot figures...")
        
        self.plot_subplot_distance()
        self.plot_subplot_yielding() 
        self.plot_subplot_speed()
        self.plot_subplot_acceleration()
        self.plot_subplot_cbf_comparison()
        self.plot_subplot_violations()
        
        print("All individual subplot figures generated successfully!")
    
    def plot_subplot_distance(self) -> None:
        """Generate individual subplot for distance barrier functions."""
        if self.data is None:
            print("No data loaded. Call load_data() first.")
            return
        
        num_humans = self._get_human_count()
        if num_humans == 0:
            print("No human data found in the dataset.")
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot main distance barrier function
        ax.plot(self.data['time'], self.data['h_distance'], 'k-', linewidth=3, 
                label='h_distance (main)', alpha=0.8)
        
        # Plot individual human distance barrier functions
        colors = plt.cm.get_cmap('Set1')(np.linspace(0, 1, num_humans))
        for i in range(num_humans):
            col_name = f'h_distance_human{i}'
            if col_name in self.data.columns:
                ax.plot(self.data['time'], self.data[col_name], 
                       color=colors[i], linewidth=2, label=f'Human {i}')
        
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Distance Barrier Function', fontsize=12)
        ax.set_title('Distance Barrier Functions vs Time', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/subplot_distance.png", dpi=600, bbox_inches='tight')
        self._show_or_close()
        print(f"Saved Distance subplot to {self.plots_dir}/subplot_distance.png")
    
    def plot_subplot_yielding(self) -> None:
        """Generate individual subplot for yielding barrier functions."""
        if self.data is None:
            print("No data loaded. Call load_data() first.")
            return
        
        num_humans = self._get_human_count()
        if num_humans == 0:
            print("No human data found in the dataset.")
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot main yielding barrier function
        if 'h_yielding' in self.data.columns:
            ax.plot(self.data['time'], self.data['h_yielding'], 'k-', linewidth=3, 
                    label='h_yielding (main)', alpha=0.8)
        
        # Plot individual human yielding barrier functions
        colors = plt.cm.get_cmap('Set2')(np.linspace(0, 1, num_humans))
        for i in range(num_humans):
            col_name = f'h_yielding_human{i}'
            if col_name in self.data.columns:
                ax.plot(self.data['time'], self.data[col_name], 
                       color=colors[i], linewidth=2, label=f'Human {i}')
        
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Yielding Barrier Function', fontsize=12)
        ax.set_title('Yielding Barrier Functions vs Time', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/subplot_yielding.png", dpi=600, bbox_inches='tight')
        self._show_or_close()
        print(f"Saved Yielding subplot to {self.plots_dir}/subplot_yielding.png")
    
    def plot_subplot_speed(self) -> None:
        """Generate individual subplot for speed barrier functions."""
        if self.data is None:
            print("No data loaded. Call load_data() first.")
            return
        
        num_humans = self._get_human_count()
        if num_humans == 0:
            print("No human data found in the dataset.")
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot main speed barrier function
        ax.plot(self.data['time'], self.data['h_speed'], 'k-', linewidth=3, 
                label='h_speed (main)', alpha=0.8)
        
        # Plot individual human speed barrier functions
        colors = plt.cm.get_cmap('Set3')(np.linspace(0, 1, num_humans))
        for i in range(num_humans):
            col_name = f'h_speed_human{i}'
            if col_name in self.data.columns:
                ax.plot(self.data['time'], self.data[col_name], 
                       color=colors[i], linewidth=2, label=f'Human {i}')
        
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Speed Barrier Function', fontsize=12)
        ax.set_title('Speed Barrier Functions vs Time', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/subplot_speed.png", dpi=600, bbox_inches='tight')
        self._show_or_close()
        print(f"Saved Speed subplot to {self.plots_dir}/subplot_speed.png")
    
    def plot_subplot_acceleration(self) -> None:
        """Generate individual subplot for acceleration barrier functions."""
        if self.data is None:
            print("No data loaded. Call load_data() first.")
            return
        
        num_humans = self._get_human_count()
        if num_humans == 0:
            print("No human data found in the dataset.")
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot main acceleration barrier function
        if 'h_accel' in self.data.columns:
            ax.plot(self.data['time'], self.data['h_accel'], 'k-', linewidth=3, 
                    label='h_accel (main)', alpha=0.8)
        
        # Plot individual human acceleration barrier functions
        colors = plt.cm.get_cmap('Pastel1')(np.linspace(0, 1, num_humans))
        for i in range(num_humans):
            col_name = f'h_accel_human{i}'
            if col_name in self.data.columns:
                ax.plot(self.data['time'], self.data[col_name], 
                       color=colors[i], linewidth=2, label=f'Human {i}')
        
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Acceleration Barrier Function', fontsize=12)
        ax.set_title('Acceleration Barrier Functions vs Time', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/subplot_acceleration.png", dpi=600, bbox_inches='tight')
        self._show_or_close()
        print(f"Saved Acceleration subplot to {self.plots_dir}/subplot_acceleration.png")
    
    def plot_subplot_cbf_comparison(self) -> None:
        """Generate individual subplot comparing all CBF types."""
        if self.data is None:
            print("No data loaded. Call load_data() first.")
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot main CBF functions
        main_cbfs = ['cbf_distance', 'cbf_yielding', 'cbf_speed', 'cbf_accel']
        colors = ['blue', 'green', 'red', 'orange']
        
        for i, cbf_type in enumerate(main_cbfs):
            if cbf_type in self.data.columns:
                ax.plot(self.data['time'], self.data[cbf_type], 
                       color=colors[i], linewidth=3, label=cbf_type.replace('cbf_', 'CBF '))
        
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('CBF Value', fontsize=12)
        ax.set_title('Control Barrier Functions Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/subplot_cbf_comparison.png", dpi=600, bbox_inches='tight')
        self._show_or_close()
        print(f"Saved CBF Comparison subplot to {self.plots_dir}/subplot_cbf_comparison.png")
    
    def plot_subplot_violations(self) -> None:
        """Generate individual subplot for safety violations."""
        if self.data is None:
            print("No data loaded. Call load_data() first.")
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Find violation columns
        violation_cols = [col for col in self.data.columns if col.startswith('violation_')]
        
        if not violation_cols:
            print("No violation columns found in the dataset.")
            return
        
        colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(violation_cols)))
        
        for i, col in enumerate(violation_cols):
            # Convert boolean violations to numerical for plotting
            violations = self.data[col].astype(int)
            ax.plot(self.data['time'], violations, 
                   color=colors[i], linewidth=2, marker='o', markersize=3,
                   label=col.replace('violation_', 'Violation '))
        
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Violation Status (0=Safe, 1=Violation)', fontsize=12)
        ax.set_title('Safety Violations vs Time', fontsize=14, fontweight='bold')
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/subplot_violations.png", dpi=600, bbox_inches='tight')
        self._show_or_close()
        print(f"Saved Violations subplot to {self.plots_dir}/subplot_violations.png")

    def plot_all_figures(self) -> None:
        """Generate all five figures."""
        print("Generating all plots...")
        
        self.plot_distance_and_speed()
        self.plot_all_humans_subplots()
        self.plot_minimum_each_human()
        self.plot_minimum_all_humans()
        self.plot_all_cbfs()
        
        print("All plots generated successfully!")
    
    def get_data_summary(self) -> None:
        """Print a summary of the loaded data."""
        if self.data is None:
            print("No data loaded. Call load_data() first.")
            return
        
        print(f"\n=== Data Summary for run_{self.run_name} ===")
        print(f"Shape: {self.data.shape}")
        print(f"Time range: {self.data['time'].min():.3f} - {self.data['time'].max():.3f} seconds")
        print(f"Number of humans: {self._get_human_count()}")
        
        # Show available columns
        h_columns = [col for col in self.data.columns if col.startswith('h_')]
        cbf_columns = [col for col in self.data.columns if col.startswith('cbf_')]
        
        print(f"\nBarrier function columns ({len(h_columns)}): {h_columns[:5]}{'...' if len(h_columns) > 5 else ''}")
        print(f"CBF columns ({len(cbf_columns)}): {cbf_columns[:5]}{'...' if len(cbf_columns) > 5 else ''}")


def main():
    """Main function to demonstrate the plotter usage."""
    # Initialize plotter
    plotter = SimulationPlotter()
    
    # Load data (will use run_name from config or default to 'test')
    if plotter.load_data():
        # Print data summary
        plotter.get_data_summary()
        
        # Generate all plots
        plotter.plot_all_figures()
    else:
        print("Failed to load data. Please check the file path and try again.")


if __name__ == "__main__":
    main()
