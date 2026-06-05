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
        data = df_cpu[df_cpu['Method'] == method]
        if not data.empty:
            data = data.groupby('N').mean(numeric_only=True).reset_index()
            data = data.sort_values('N')
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
        data = df_fmg[df_fmg['Hardware'] == hw]
        if not data.empty:
            data = data.groupby('N').mean(numeric_only=True).reset_index()
            data = data.sort_values('N')
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
        data = df[df['Method'] == method]
        if not data.empty:
            data = data.groupby('Threads').mean(numeric_only=True).reset_index()
            data = data.sort_values('Threads')
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

def plot_error_convergence(df):
    plt.figure(figsize=(10, 6))
    
    # Use CPU data to plot error convergence
    df_cpu = df[df['Hardware'] == 'CPU']
    
    methods = ['V-Cycle', 'FMG', 'W-Cycle']
    markers = ['s', 'd', '^']
    
    for method, marker in zip(methods, markers):
        data = df_cpu[df_cpu['Method'] == method]
        if not data.empty:
            data = data.groupby('N').mean(numeric_only=True).reset_index()
            data = data.sort_values('N')
            plt.plot(data['N'], data['Error'], marker=marker, linewidth=2, label=method)
            
    # Add a theoretical O(h^2) reference line. Error ~ 1/N^2
    # Find a good starting point based on FMG's first N
    if not df_cpu.empty:
        n_vals = sorted(df_cpu['N'].unique())
        if len(n_vals) > 0:
            start_err = 1e-4
            theory_err = [start_err * ((n_vals[0]/n)**2) for n in n_vals]
            plt.plot(n_vals, theory_err, 'k--', linewidth=2, alpha=0.7, label='Theoretical $O(h^2)$')
            
    plt.xscale('log', base=2)
    plt.yscale('log', base=10)
    plt.xlabel('Grid Size (N)', fontsize=12)
    plt.ylabel('L2 Error Norm', fontsize=12)
    plt.title('Numerical Accuracy: Error Convergence vs Grid Size', fontsize=14)
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig('results/images/error_convergence.png', dpi=300)
    print("Saved results/images/error_convergence.png")

def plot_vcycle_vs_fmg(df):
    plt.figure(figsize=(10, 6))
    
    # We will plot CPU FMG vs CPU V-Cycle
    df_cpu = df[df['Hardware'] == 'CPU']
    
    for method, marker in zip(['V-Cycle', 'FMG'], ['s', 'd']):
        data = df_cpu[df_cpu['Method'] == method]
        if not data.empty:
            data = data.groupby('N').mean(numeric_only=True).reset_index()
            data = data.sort_values('N')
            plt.plot(data['N'], data['Time_s'], marker=marker, linewidth=2, label=f'CPU {method}')
            
    plt.xscale('log', base=2)
    plt.yscale('log', base=10)
    plt.xlabel('Grid Size (N)', fontsize=12)
    plt.ylabel('Execution Time (Seconds)', fontsize=12)
    plt.title('V-Cycle (5-10 Iters) vs Full Multigrid (1 Iter)', fontsize=14)
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig('results/images/vcycle_vs_fmg.png', dpi=300)
    print("Saved results/images/vcycle_vs_fmg.png")

def plot_mpi_strong_scaling(df):
    plt.figure(figsize=(10, 6))
    
    # Filter for MPI hardware specifically
    df_mpi = df[df['Hardware'].str.contains('MPI')]
    if df_mpi.empty:
        return
        
    # We will plot strong scaling for N=32769
    target_N = 32769
    df_mpi_N = df_mpi[df_mpi['N'] == target_N].copy()
    if df_mpi_N.empty:
        return
        
    # Extract Process count from string (e.g. 'MPI-64P' -> 64)
    df_mpi_N['Processes'] = df_mpi_N['Hardware'].str.extract(r'(\d+)').astype(int)
    
    for method in ['V-Cycle', 'FMG']:
        data = df_mpi_N[df_mpi_N['Method'] == method]
        if not data.empty:
            data = data.groupby('Processes').mean(numeric_only=True).reset_index()
            data = data.sort_values('Processes')
            plt.plot(data['Processes'], data['Time_s'], marker='o', linewidth=2, label=method)
            
    # Add ideal linear scaling reference line
    fmg_data = df_mpi_N[df_mpi_N['Method'] == 'FMG'].groupby('Processes').mean(numeric_only=True).reset_index()
    if not fmg_data.empty:
        p1 = fmg_data['Processes'].min()
        t1 = fmg_data[fmg_data['Processes'] == p1]['Time_s'].iloc[0]
        p_vals = sorted(fmg_data['Processes'].unique())
        ideal_time = [t1 * (p1/p) for p in p_vals]
        plt.plot(p_vals, ideal_time, 'k--', linewidth=2, alpha=0.7, label='Ideal Linear Scaling')
            
    plt.xscale('log', base=2)
    plt.yscale('log', base=10)
    plt.xlabel('Number of MPI Processes', fontsize=12)
    plt.ylabel('Execution Time (Seconds)', fontsize=12)
    plt.title(f'MPI Strong Scaling on Distributed Memory (Grid N={target_N})', fontsize=14)
    plt.xticks([1, 2, 4, 8, 16, 32, 64], ['1', '2', '4', '8', '16', '32', '64'])
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig('results/images/mpi_strong_scaling.png', dpi=300)
    print("Saved results/images/mpi_strong_scaling.png")

if __name__ == "__main__":
    df = load_data()
    if df is not None:
        plot_algorithmic_scaling(df)
        plot_hardware_comparison(df)
        plot_error_convergence(df)
        plot_vcycle_vs_fmg(df)
        plot_mpi_strong_scaling(df)
    plot_strong_scaling()
    print("All plots generated successfully!")
