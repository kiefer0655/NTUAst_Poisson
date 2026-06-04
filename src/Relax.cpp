#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <omp.h>

#define loging false

class Relax{
public:
    int     N;
    double  h;
    double  h2;
    double  epsilon;
    int     max_iter;
    double  w;

    std::vector<double> U;
    std::vector<double> U_new;
    std::vector<double> U_ref;
    std::vector<double> Rho;


    void (Relax::*update_method)();

    Relax(int N_, std::string algorithm_, double epsilon_, int max_iter_, double w_ = 1.0)
        : N(N_), epsilon(epsilon_), max_iter(max_iter_), w(w_)
    {
        h = 1.0/(N-1);
        h2 = h*h;
        U.assign(N * N, 0.0);
        U_new.assign(N * N, 0.0);
        U_ref.assign(N * N, 0.0);
        Rho.assign(N * N, 0.0);

        if (algorithm_ == "SOR") {
            update_method = &Relax::sor_update;
        }
        else if (algorithm_ == "OMP_SOR"){
            update_method = &Relax::omp_sor_update;
        }
        
        else {
            std::cerr << "Unknown algorithm\n";
            exit(1);
        }

        initialize();
    }

    int idx(int i, int j) const{
        return i * N + j;
    }

    void initialize(){
        for(int i = 0; i < N; i++){
            double x = i*h;
            for(int j = 0; j < N; j++){
                double y = j*h;

                double ref = std::sin(M_PI * x) * std::sin(M_PI * y);

                U_ref[idx(i,j)] = ref;
                Rho[idx(i,j)]   = -2.0 * M_PI * M_PI * ref;
            }
        }
    }

    void apply_bc(){
        #pragma omp parallel for schedule(static)
        for(int i = 0; i<N; i++){
            U[idx(i,0)] = 0.0;
            U[idx(i,N-1)] = 0.0;
            U[idx(0,i)] = 0.0;
            U[idx(N-1,i)] = 0.0;
        }
    }

    double get_error() const{
        double sum = 0.0;
        for(int i=0;i<N*N;i++){
            sum += std::abs(U[i]-U_ref[i]);
        }
        return sum/(N*N);
    }

    void sor_update(){
        for(int i=1; i<N-1; i++){
            for (int j = 1; j < N-1; j++)
            {
                double gs_value = 0.25 * (
                    U[idx(i - 1, j)] +
                    U[idx(i + 1, j)] +
                    U[idx(i, j - 1)] +
                    U[idx(i, j + 1)] -
                    h2 * Rho[idx(i, j)]);

                U[idx(i, j)] = (1.0 - w) * U[idx(i, j)] + w * gs_value;
            }
        }
    }

    void omp_sor_update(){

        for(int color=0; color<2;color ++){
            // color = 0 orange, color = 1 blue
            
            #pragma omp parallel for schedule(static)
            for(int i=1; i<N-1; i++){
            
                int j_start = ((i + 1) % 2 == color) ? 1 : 2;


                for (int j = j_start; j < N-1; j+=2)
                {
                    int id = idx(i, j);
                    double gs_value = 0.25 * (
                        U[idx(i - 1, j)] +
                        U[idx(i + 1, j)] +
                        U[idx(i, j - 1)] +
                        U[idx(i, j + 1)] -
                        h2 * Rho[id]);
                    U[id] = (1.0 - w) * U[id] + w * gs_value;
                }
            }
            }
    }

    void update(){
        // apply_bc();
        (this->*update_method)();
        // apply_bc();
    }

    int chill(){
        for (int iter = 0; iter < max_iter; iter++)
        {
            std::vector<double> U_old = U;
            update();

            double r =0.0;
            # pragma omp parallel for reduction(max:r)
            for (int k = 0; k < N*N; k++)
            {
                r = std::max(r,std::abs(U[k]-U_old[k]));
            }
            
            if(iter%1000 == 0){
                if( loging ){
                std::cout << "r = " << r << std::endl;
                }
            }
            
            if(r < epsilon){
                if( loging){
                    std::cout << "done chilling total iter = " << iter << std::endl;
                }
                    return iter;
            }
        
        }
        std::cout << "ACHTUNG!: Max Iteration Reached, can't converge, total iter = "
                  << max_iter << std::endl;
        return 0;
    }


};

void generate_time_CSV(  const   std::vector<int>& Ns,
                        const   std::vector<int>& Threads
                    )
{
    std::cout << "N,threads,runtime,iterations,error" << std::endl;
    

    for(int Thread: Threads){
        
    }
}


int main(int argc, char* argv[]){

    double epsilon  = 1e-8;
    int maxiter     = 100000;
    double w        = 1.8;

    int N           = 101;
    int Thread      = 1;

    if(argc >= 2){
        N = std::atoi(argv[1]);
    }

    if(argc >= 3){
        Thread = std::atoi(argv[2]);
    }
    
    omp_set_num_threads(Thread);
        
    Relax solver(N, "OMP_SOR", epsilon, maxiter, w);
    double start_time = omp_get_wtime();
    solver.chill();
    double end_time = omp_get_wtime();
    std::cout   << N << ","
                << Thread << ","
                << end_time - start_time<< std::endl;

    return 0;
}