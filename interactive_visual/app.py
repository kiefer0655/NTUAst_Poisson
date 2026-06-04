import os
import csv
import struct
import subprocess
from flask import Flask, render_template, request, jsonify
import numpy as np

app = Flask(__name__)

# Ensure binaries are compiled
subprocess.run(['make', 'benchmark_suite'], cwd='..', stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
subprocess.run(['make', 'solve_canvas'], cwd='..', stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/benchmark')
def api_benchmark():
    cpu_csv = '../results_cpu.csv'
    gpu_csv = '../results_gpu.csv'
    
    if not os.path.exists(cpu_csv) and not os.path.exists(gpu_csv):
        return jsonify({"error": "No result CSVs found. Run benchmark_cpu or benchmark_gpu first."}), 404
        
    data = []
    
    for csv_path in [cpu_csv, gpu_csv]:
        if os.path.exists(csv_path):
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    data.append({
                        "N": int(row["N"]),
                        "Method": row["Method"],
                        "Hardware": row["Hardware"],
                        "Iterations": int(row["Iterations"]),
                        "Time_s": float(row["Time_s"]),
                        "Error": float(row["Error"])
                    })
                    
    return jsonify(data)

@app.route('/api/solve_canvas', methods=['POST'])
def api_solve_canvas():
    try:
        req = request.get_json()
        N = req.get('N', 65)
        rho_flat = req.get('rho', [])
        
        if len(rho_flat) != N * N:
            return jsonify({"error": "Invalid rho array size"}), 400

        # Write rho to binary file
        in_file = 'temp_rho.bin'
        out_file = 'temp_u.bin'
        
        with open(in_file, 'wb') as f:
            for val in rho_flat:
                f.write(struct.pack('d', float(val)))
                
        # Call the C++ CUDA binary to solve FMG instantly
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = '3'
        
        result = subprocess.run(
            ['./solve_canvas', str(N), f'interactive_visual/{in_file}', f'interactive_visual/{out_file}'],
            cwd='..', capture_output=True, text=True, env=env
        )
        
        if result.returncode != 0:
            return jsonify({"error": "C++ solver failed", "details": result.stderr}), 500
            
        # Read the solution U
        u_flat = []
        with open(out_file, 'rb') as f:
            while True:
                bytes_read = f.read(8)
                if not bytes_read:
                    break
                val = struct.unpack('d', bytes_read)[0]
                u_flat.append(val)
                
        # Clean up
        if os.path.exists(in_file): os.remove(in_file)
        if os.path.exists(out_file): os.remove(out_file)
        
        return jsonify({"u": u_flat})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8011)
