const N = 257;
let rho = new Float64Array(N * N);
let isDrawing = false;

const rhoCanvas = document.getElementById('rhoCanvas');
const ctxRho = rhoCanvas.getContext('2d');
const uCanvas = document.getElementById('uCanvas');
const ctxU = uCanvas.getContext('2d');

// Canvas Drawing Logic for rho
function drawPoint(x, y) {
    const radius = 5;
    for (let i = -radius; i <= radius; i++) {
        for (let j = -radius; j <= radius; j++) {
            if (i*i + j*j <= radius*radius) {
                let px = Math.floor(x) + i;
                let py = Math.floor(y) + j;
                if (px > 0 && px < N-1 && py > 0 && py < N-1) {
                    rho[py * N + px] = -1000.0; // Negative charge density for visualization
                }
            }
        }
    }
    renderRho();
}

function renderRho() {
    let imgData = ctxRho.createImageData(N, N);
    for (let i = 0; i < N * N; i++) {
        let val = Math.abs(rho[i]) > 0 ? 255 : 0;
        imgData.data[i * 4 + 0] = val; // R
        imgData.data[i * 4 + 1] = val; // G
        imgData.data[i * 4 + 2] = val; // B
        imgData.data[i * 4 + 3] = 255; // A
    }
    ctxRho.putImageData(imgData, 0, 0);
}

rhoCanvas.addEventListener('mousedown', (e) => {
    isDrawing = true;
    drawPoint(e.offsetX, e.offsetY);
});
rhoCanvas.addEventListener('mousemove', (e) => {
    if (isDrawing) drawPoint(e.offsetX, e.offsetY);
});
rhoCanvas.addEventListener('mouseup', () => { isDrawing = false; });
rhoCanvas.addEventListener('mouseleave', () => { isDrawing = false; });

document.getElementById('btnClear').addEventListener('click', () => {
    rho.fill(0);
    renderRho();
    ctxU.clearRect(0, 0, N, N);
    document.getElementById('solveTime').innerText = "Solve Time: 0.000s";
});

// Colormap for potential field (Blue to Red)
function getJetColor(v, vmin, vmax) {
    let c = {r: 0, g: 0, b: 0};
    if (vmax <= vmin) return c;
    let dv = vmax - vmin;
    if (v < vmin) v = vmin;
    if (v > vmax) v = vmax;
    let result = (v - vmin) / dv;
    
    // Smooth transition
    if (result < 0.25) {
        c.b = 255;
        c.g = Math.floor(255 * (4 * result));
    } else if (result < 0.5) {
        c.b = Math.floor(255 * (1 - 4 * (result - 0.25)));
        c.g = 255;
    } else if (result < 0.75) {
        c.r = Math.floor(255 * (4 * (result - 0.5)));
        c.g = 255;
    } else {
        c.r = 255;
        c.g = Math.floor(255 * (1 - 4 * (result - 0.75)));
    }
    return c;
}

document.getElementById('btnSolve').addEventListener('click', async () => {
    const btn = document.getElementById('btnSolve');
    btn.innerText = "Solving...";
    btn.disabled = true;
    
    const startTime = performance.now();
    try {
        const response = await fetch('/api/solve_canvas', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ N: N, rho: Array.from(rho) })
        });
        const data = await response.json();
        
        if (data.error) {
            alert(data.error);
        } else {
            const endTime = performance.now();
            document.getElementById('solveTime').innerText = `Solve Time: ${((endTime - startTime)/1000).toFixed(3)}s (Round-trip)`;
            
            const u = data.u;
            
            // Find min/max for colormap
            let vmin = Math.min(...u);
            let vmax = Math.max(...u);
            // Add a tiny offset to avoid division by zero
            if (vmax === vmin) vmax += 1e-9;
            
            let imgData = ctxU.createImageData(N, N);
            for (let i = 0; i < N * N; i++) {
                let color = getJetColor(u[i], vmin, vmax);
                imgData.data[i * 4 + 0] = color.r;
                imgData.data[i * 4 + 1] = color.g;
                imgData.data[i * 4 + 2] = color.b;
                imgData.data[i * 4 + 3] = 255;
            }
            ctxU.putImageData(imgData, 0, 0);
        }
    } catch (err) {
        console.error(err);
        alert("Failed to solve. Check console.");
    }
    
    btn.innerText = "Solve on GPU";
    btn.disabled = false;
});

// Chart.js Benchmarks
let benchmarkChart = null;

document.getElementById('btnLoadBench').addEventListener('click', async () => {
    try {
        const res = await fetch('/api/benchmark');
        if (!res.ok) {
            const err = await res.json();
            alert(err.error || "Failed to fetch benchmark data");
            return;
        }
        const data = await res.json();
        
        // Group by hardware and method
        // We will plot N on X axis, Time on Y axis (log scale)
        const datasets = [];
        const colors = {
            'CPU-SOR': '#ef4444',
            'CPU-V-Cycle': '#f97316',
            'CPU-W-Cycle': '#eab308',
            'CPU-FMG': '#22c55e',
            'GPU-V-Cycle': '#06b6d4',
            'GPU-FMG': '#8b5cf6'
        };
        
        // Extract unique N's
        const N_labels = [...new Set(data.map(d => d.N))].sort((a,b) => a-b);
        
        const grouped = {};
        data.forEach(d => {
            const key = `${d.Hardware}-${d.Method}`;
            if (!grouped[key]) grouped[key] = { label: key, data: {}, borderColor: colors[key] || '#ffffff' };
            grouped[key].data[d.N] = d.Time_s / d.Iterations; // Time per iteration
        });
        
        for (let key in grouped) {
            const points = N_labels.map(n => grouped[key].data[n] || null);
            datasets.push({
                label: key,
                data: points,
                borderColor: grouped[key].borderColor,
                backgroundColor: grouped[key].borderColor,
                tension: 0.1,
                fill: false,
                borderWidth: 2,
                pointRadius: 4
            });
        }
        
        const ctx = document.getElementById('benchmarkChart').getContext('2d');
        if (benchmarkChart) benchmarkChart.destroy();
        
        benchmarkChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: N_labels.map(n => `N=${n}`),
                datasets: datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                color: '#cbd5e1',
                plugins: {
                    legend: { labels: { color: '#cbd5e1' } },
                    title: { display: true, text: 'Time per Solution Cycle (Seconds)', color: '#fff' }
                },
                scales: {
                    y: {
                        type: 'logarithmic',
                        title: { display: true, text: 'Time (s) [Log Scale]', color: '#94a3b8' },
                        grid: { color: 'rgba(255,255,255,0.1)' },
                        ticks: { color: '#cbd5e1' }
                    },
                    x: {
                        grid: { color: 'rgba(255,255,255,0.1)' },
                        ticks: { color: '#cbd5e1' }
                    }
                }
            }
        });
        
    } catch (err) {
        console.error(err);
    }
});

// Initialize empty Rho canvas
renderRho();
