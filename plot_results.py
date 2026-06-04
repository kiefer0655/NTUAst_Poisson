import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

# Create images directory if it doesn't exist
os.makedirs('results/images', exist_ok=True)
sns.set_theme(style="whitegrid")

def load_data():
    dfs = []
    if os.path.exists('results/results_cpu.csv'):
        dfs.append(pd.read_csv('results/results_cpu.csv'))
    if os.path.exists('results/results_gpu.csv'):
        dfs.append(pd.read_csv('results/results_gpu.csv'))
    if os.path.exists('results/results_mpi.csv'):
        dfs.append(pd.read_csv('results/results_mpi.csv'))
    
    if not dfs:
        print("No results found in results/ folder.")
        return None
    return pd.concat(dfs, ignore_index=True)

def plot_algorithmic_scaling(df):
    plt.figure(figsize=(10, 6))
    
    # Filter only for CPU OpenMP to compare pure algorithmic scaling
    df_cpu = df[df['Hardware'] == 'CPU']
    
    methods = ['SOR', 'V-Cycle', 'W-Cycle', 'FMG']
    markers = ['o', 's', '^', 'd']
    
    for method, marker in zip(methods, markers):
        data = df_cpu[df_cpu['Method'] == method].sort_values('N')
        if not data.empty:
            plt.plot(data['N'], data['Time_s'], marker=marker, linewidth=2, label=method)
            
    plt.xscale('log', base=2)
    plt.yscale('log', base=10)
    plt.xlabel('Grid Size (N)', fontsize=12)
    plt.ylabel('Execution Time (Seconds)', fontsize=12)
    plt.title('Algorithmic Complexity (Time vs Grid Size) on Shared Memory CPU', fontsize=14)
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig('results/images/algorithmic_scaling.png', dpi=300)
    print("Saved results/images/algorithmic_scaling.png")

def plot_hardware_comparison(df):
    plt.figure(figsize=(10, 6))
    
    # Let's compare FMG across architectures
    df_fmg = df[df['Method'] == 'FMG'].copy()
    
    # We want CPU, GPU, and the best MPI (64P)
    df_fmg = df_fmg[df_fmg['Hardware'].isin(['CPU', 'GPU', 'MPI-64P'])]
    
    for hw in ['CPU', 'GPU', 'MPI-64P']:
        data = df_fmg[df_fmg['Hardware'] == hw].sort_values('N')
        if not data.empty:
            plt.plot(data['N'], data['Time_s'], marker='o', linewidth=2, label=hw)
            
    plt.xscale('log', base=2)
    plt.yscale('log', base=10)
    plt.xlabel('Grid Size (N)', fontsize=12)
    plt.ylabel('Execution Time (Seconds)', fontsize=12)
    plt.title('Hardware Acceleration: FMG Performance (CPU vs GPU vs MPI-64)', fontsize=14)
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig('results/images/hardware_comparison.png', dpi=300)
    print("Saved results/images/hardware_comparison.png")

def plot_strong_scaling():
    if not os.path.exists('results/scaling_cpu.csv'):
        return
        
    df = pd.read_csv('results/scaling_cpu.csv')
    plt.figure(figsize=(10, 6))
    
    # Extract thread count from Hardware string (e.g., 'CPU-16T' -> 16)
    df['Threads'] = df['Hardware'].str.extract(r'(\d+)').astype(int)
    
    for method in ['V-Cycle', 'FMG']:
        data = df[df['Method'] == method].sort_values('Threads')
        if not data.empty:
            plt.plot(data['Threads'], data['Time_s'], marker='o', linewidth=2, label=method)
            
    plt.xscale('log', base=2)
    plt.xlabel('Number of OpenMP Threads', fontsize=12)
    plt.ylabel('Execution Time (Seconds)', fontsize=12)
    plt.title('Strong Scaling & Amdahl\'s Law on 144-Core CPU (Grid N=2049)', fontsize=14)
    plt.xticks([1, 2, 4, 8, 16, 32, 64, 144], ['1', '2', '4', '8', '16', '32', '64', '144'])
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.6)
    
    # Add annotation for the bottleneck
    plt.annotate('Memory Wall\n(Bandwidth Saturation)', xy=(16, 0.1), xytext=(32, 0.4),
                 arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10)
    plt.annotate('Thread Overhead\nDominates', xy=(144, 0.37), xytext=(64, 0.7),
                 arrowprops=dict(facecolor='red', shrink=0.05), fontsize=10, color='red')
                 
    plt.tight_layout()
    plt.savefig('results/images/strong_scaling.png', dpi=300)
    print("Saved results/images/strong_scaling.png")

if __name__ == "__main__":
    df = load_data()
    if df is not None:
        plot_algorithmic_scaling(df)
        plot_hardware_comparison(df)
    plot_strong_scaling()
    print("All plots generated successfully!")
