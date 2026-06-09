## 1. Parallelization Methods

1. **OpenMP**：single node、multi core CPU, shared memory
2. **MPI**：multi node、Distributed Memory
3. **CUDA**：GPU

## 2. Implementation and Difficulties

### 2.1 OpenMP (Shared Memory 平行化)

* **實作方法**：
    在原本的 C++ 迴圈前加上 `#pragma omp parallel for`，讓編譯器自動將迴圈的工作分配給多個 CPU Threads。主要應用在 SOR Smoothing、殘差 (Residual) 計算、Restriction 與 Prolongation 這些資料不互相關聯的網格操作上。
* **遇到的困難與挑戰：Overhead 問題**
    在 Multigrid 中，V-Cycle 會將網格一路降維到非常小的 Coarse Grid (例如 $3 \times 3$)。這時候如果仍強制使用 144 個 Threads 去切分 9 個格子，**Thread 喚醒與同步的成本 (Overhead)** 會遠遠超過實際的數學計算，導致極嚴重的效能倒退 (Amdahl's Law)。

### 2.2 MPI (Distributed Memory 平行化)

* **實作方法**：
    我們採用了**領域分割 (Domain Decomposition)** 的策略。將一個超巨大的 1D 陣列 (例如 1 Billion Cells) 垂直/水平切分成多個子區塊，分配給不同的 MPI Process。每個 Process 有自己獨立的記憶體。
* **遇到的困難與挑戰：Ghost Cells 與通訊瓶頸**
    因為計算 Laplacian 需要相鄰的點 (Stencils)，我們必須實作 **Halo Exchange (Ghost Cells)**。每次迭代前，Process 必須使用 `MPI_Sendrecv` 和相鄰的 Process 交換邊界資料。
    最大的困難在於 **W-Cycle**：在極大的網格下 ($N \ge 4097$)，W-Cycle 頻繁地在極小的底層 Coarse Grid 之間來回跳躍。

### 2.3 CUDA (GPU 平行化)

* **實作方法**：
    我們將 CPU 的 `for` 迴圈改寫為 CUDA Kernels (`__global__`)，讓 GPU 的數千個核心同時處理每一格的計算。
* **遇到的困難與挑戰：Data Dependency (Race Condition)**
    傳統的 Gauss-Seidel 或是 SOR 演算法具有嚴格的資料依賴性 (Data Dependency)，這格算完才能算下一格，這在 GPU 上是無法平行化的。
    **解決方案**：我們實作了 **Red-Black SOR (紅黑排序法)**。將網格像西洋棋盤一樣標上紅黑兩色。因為紅格的鄰居只有黑格，所以我們可以先「完全平行」地更新所有紅格，再「完全平行」地更新所有黑格，成功釋放了 GPU 的算力！

## 3. 實驗結果與數據圖表解析

### 圖一：不同硬體的效能對比 (Hardware Comparison)

![Hardware Comparison](../results/images/hardware_comparison.png)

* **解析**：可以先跳過

### 圖二：OpenMP 強擴展性測試 (Strong Scaling - CPU Shared Memory)

![OpenMP Strong Scaling](../results/images/strong_scaling.png)

* **解析**：可以先跳過

### 圖三：MPI 強擴展性測試 (Strong Scaling - Distributed Memory)

![MPI Strong Scaling](../results/images/mpi_strong_scaling.png)

* **解析**：可以先跳過

### 平行化總結

* **解析**：可以先跳過
