import wave
import struct
import math

# Simulate a multigrid convergence error log
# V-cycle usually reduces error by a constant factor every iteration
iterations = 20
initial_error = 1.0
error_reduction_factor = 0.5 # Error halves every iteration (typical for Poisson V-cycle)

# We map error (1.0 to 1e-6) to frequency (1000Hz down to 100Hz)
def map_error_to_freq(error):
    if error < 1e-10: error = 1e-10
    log_err = math.log10(error) # 0 to -10
    # map log_err 0 -> 1000 Hz, -10 -> 100 Hz
    freq = 1000 + (log_err / -10.0) * (100 - 1000)
    return max(100, min(1000, freq))

# Audio parameters
sample_rate = 44100
duration_per_iter = 0.25 # seconds per iteration

print("Generating Multigrid Convergence Sonification...")
obj = wave.open('convergence.wav', 'w')
obj.setnchannels(1) # mono
obj.setsampwidth(2) # 16-bit
obj.setframerate(sample_rate)

error = initial_error
for i in range(iterations):
    freq = map_error_to_freq(error)
    print(f"Iter {i}: Error = {error:.2e} | Tone = {freq:.1f} Hz")
    
    num_samples = int(sample_rate * duration_per_iter)
    for j in range(num_samples):
        t = float(j) / sample_rate
        # Sine wave with volume envelope (slight fade out at the end of the tone)
        volume = 0.5 * (1.0 - j/num_samples)
        value = int(32767.0 * volume * math.sin(2.0 * math.pi * freq * t))
        data = struct.pack('<h', value)
        obj.writeframesraw(data)
        
    error *= error_reduction_factor

obj.close()
print("\nSuccess! Saved sonification to 'convergence.wav'")
print("Play this file to literally HEAR the high-frequency error being smoothed out of the grid over time!")
