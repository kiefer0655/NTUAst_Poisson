const N = 257; // Grid size
let solver = null;
let isDrawing = false;
let audioCtx = null;
let oscillator = null;
let gainNode = null;
let isAudioEnabled = false;

// Color mapping (Heatmap: Blue -> Black -> Red)
function valueToRGB(val) {
    // val is typical around -1 to 1 for this problem, but could be larger
    // Map val to a smooth color
    const norm = Math.max(-1, Math.min(1, val / 100.0)); // Normalize roughly
    
    let r = 0, g = 0, b = 0;
    if (norm > 0) {
        r = Math.floor(norm * 255);
        b = Math.floor((1 - norm) * 50);
    } else {
        b = Math.floor(Math.abs(norm) * 255);
        r = Math.floor((1 - Math.abs(norm)) * 50);
    }
    return [r, g, b, 255];
}

async function initApp() {
    // 1. Load WebAssembly Module
    const poisson_module = await Module();
    solver = new poisson_module.PoissonSolver(N);
    
    // 2. Setup Canvas
    const canvas = document.getElementById('gridCanvas');
    const ctx = canvas.getContext('2d');
    const imgData = ctx.createImageData(N, N);
    
    // 3. UI Elements
    const residualDisplay = document.getElementById('residualDisplay');
    const brushSizeSlider = document.getElementById('brushSize');
    const chargeSlider = document.getElementById('chargeAmount');
    
    // Mouse Interaction
    function getCanvasPos(evt) {
        const rect = canvas.getBoundingClientRect();
        // Canvas visually is 514x514 but logically 257x257 (scale by 0.5)
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        return {
            x: Math.floor((evt.clientX - rect.left) * scaleX / 2),
            y: Math.floor((evt.clientY - rect.top) * scaleY / 2)
        };
    }

    function drawCharge(evt) {
        if (!isDrawing || !solver) return;
        const pos = getCanvasPos(evt);
        const radius = parseInt(brushSizeSlider.value);
        const amount = parseFloat(chargeSlider.value);
        
        // Ensure within bounds
        if (pos.x > 0 && pos.x < N && pos.y > 0 && pos.y < N) {
            // Note: C++ uses i, j. For row-major, i is y, j is x.
            solver.add_charge(pos.y, pos.x, radius, amount);
        }
    }

    canvas.addEventListener('mousedown', (e) => { isDrawing = true; drawCharge(e); });
    canvas.addEventListener('mousemove', drawCharge);
    window.addEventListener('mouseup', () => { isDrawing = false; });

    // Buttons
    document.getElementById('btnReset').addEventListener('click', () => {
        solver.reset();
    });

    document.getElementById('toggleAudio').addEventListener('click', (e) => {
        if (!audioCtx) {
            initAudio();
        }
        isAudioEnabled = !isAudioEnabled;
        if (isAudioEnabled) {
            audioCtx.resume();
            gainNode.gain.value = 0.1;
            e.target.innerText = "Disable Sonification";
            e.target.style.background = "linear-gradient(135deg, #00b3ff 0%, #00ffcc 100%)";
        } else {
            gainNode.gain.value = 0;
            e.target.innerText = "Enable Sonification";
            e.target.style.background = "";
        }
    });

    // 4. Main Simulation Loop
    function renderLoop() {
        if (solver) {
            // Compute 2 V-Cycles per frame for faster visual solving
            solver.step();
            solver.step();
            
            // Sonification
            const residual = solver.get_residual_norm();
            residualDisplay.innerText = residual.toExponential(4);
            
            if (isAudioEnabled && oscillator) {
                // Map residual to pitch: higher residual = higher pitch
                // Clamp residual to avoid exploding sound
                const clampedRes = Math.min(Math.max(residual, 1e-10), 1e5);
                // Map to 100Hz - 1000Hz logarithmically
                const minFreq = 100;
                const maxFreq = 1000;
                // Simple mapping: frequency drops as residual approaches 0
                const freq = minFreq + (Math.log10(clampedRes + 1) * 100);
                oscillator.frequency.setTargetAtTime(Math.min(freq, maxFreq), audioCtx.currentTime, 0.05);
            }

            // Visualization
            const uView = solver.get_u(); // Float64Array view from WASM memory
            
            for (let i = 0; i < N; i++) {
                for (let j = 0; j < N; j++) {
                    const idx = i * N + j;
                    const val = uView[idx];
                    const [r, g, b, a] = valueToRGB(val);
                    
                    const pIdx = idx * 4;
                    imgData.data[pIdx] = r;
                    imgData.data[pIdx + 1] = g;
                    imgData.data[pIdx + 2] = b;
                    imgData.data[pIdx + 3] = a;
                }
            }
            
            // Draw to a temporary canvas of size NxN, then scale it up to the main canvas
            const tmpCanvas = document.createElement('canvas');
            tmpCanvas.width = N;
            tmpCanvas.height = N;
            tmpCanvas.getContext('2d').putImageData(imgData, 0, 0);
            
            ctx.imageSmoothingEnabled = false; // keep pixels sharp
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(tmpCanvas, 0, 0, N, N, 0, 0, canvas.width, canvas.height);
        }
        
        requestAnimationFrame(renderLoop);
    }
    
    // Start Loop
    requestAnimationFrame(renderLoop);
}

function initAudio() {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    oscillator = audioCtx.createOscillator();
    gainNode = audioCtx.createGain();
    
    oscillator.type = 'sine';
    oscillator.frequency.value = 440;
    
    gainNode.gain.value = 0; // start muted
    
    oscillator.connect(gainNode);
    gainNode.connect(audioCtx.destination);
    
    oscillator.start();
}

// Start everything when DOM loads
window.onload = initApp;
