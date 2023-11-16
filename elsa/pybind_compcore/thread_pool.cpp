#include <chrono>
#include <atomic>
#include"compcore.hpp"
#include <unistd.h>
#include <execution>

using namespace std;

thread_local int rx;
thread_local int ry;
thread_local float* ptr_rx;
thread_local float* ptr_ry;

class ThreadPool
{
    public:
        ThreadPool(size_t numThreads, long long n_01, long long n_02, int s, long long col, float* x, float* y,float* score_data,int* x_0,int* x_1,int* y_0,int* y_1)
        {
            num_01 = n_01;
            num_02 = n_02;
            shift = s;
            ROW = 2 * shift + 1;
            COL = col;

            x_ptr = x;
            y_ptr = y;

            score = score_data;
            x_0_data=x_0;
            x_1_data=x_1;
            y_0_data=y_0;
            y_1_data=y_1;

            pool_size = num_01 * num_02;
            threads_num = numThreads;
            num_r = 0;
            num_end = 0;

            for (size_t i = 0; i < numThreads; ++i)
                workers.emplace_back(&ThreadPool::workerFunction, this);
        }

        ~ThreadPool()
        {
            while (num_end != threads_num);

            for (thread& worker : workers)
                worker.join();
        }

    private:
        long long num_01;
        long long num_02;
        int shift;
        int ROW;
        long long COL;
        float* x_ptr;
        float* y_ptr;
        float* score;
        int* x_0_data;
        int* x_1_data;
        int* y_0_data;
        int* y_1_data;

        long long pool_size;
        int threads_num;
        atomic<long long> num_r;
        atomic<long long> num_end;

        vector<thread> workers;
        bool stop = false;

        void workerFunction()
        {
            long long num_x = num_r.fetch_add(1, memory_order_relaxed);

            ptr_rx = new float[COL];
            ptr_ry = new float[COL];

            while (true)
            {
                if (num_x < pool_size)
                {
                    rx = num_x / num_02;
                    ry = num_x % num_02;
                }
                else
                {
                    ++num_end;
                    return;
                }

                copy(y_ptr + ry * COL, y_ptr + ry * COL + COL, ptr_ry);
                copy(x_ptr + rx * COL, x_ptr + rx * COL + COL, ptr_rx);
                
                vector<float>matr_ptr(ROW * COL);

                for (int i = 0; i < ROW; ++i)
                {
                    if (i <= shift)
                        for (int j = 0; j < COL - shift + i; ++j)
                            matr_ptr[i * COL + j] = ptr_rx[shift - i + j] * ptr_ry[j];
                    else
                        for (int j = 0; j < COL + shift - i; ++j)
                            matr_ptr[i * COL + j] = ptr_rx[j] * ptr_ry[i - shift + j];
                }

                int r1 = 0;
                float p1 = 0;
                int L[4] = { 0 };

                int I = 0;
                int P = 0;
                int J = 0;

                for (int i = 0; i < ROW; ++i)
                {
                    vector<float> a_num(matr_ptr.begin() + i * COL, matr_ptr.begin() + (i + 1) * COL);
                    int p = 0;
                    int n = 0; 
                    float p_s = 0;
                    float n_s = 0;
                    float max_p = 0;
                    float max_n = 0;
                    for (int j = 0; j < COL; ++j)
                    {
                        p_s = max(p_s + a_num[j], (float)0.0); 
                        n_s = max(n_s - a_num[j], (float)0.0); 
                        if (p_s == 0) p = j + 1;
                        else  if (p_s > max_p) max_p = p_s;
                        if (n_s == 0) n = j + 1;
                        else  if (n_s > max_n) max_n = n_s;

                        if (max_p > max_n) { if (max_p > p1) { p1 = max_p; r1 =  1; I = i, P = p, J = j; } }
                        else               { if (max_n > p1) { p1 = max_n; r1 = -1; I = i, P = n, J = j; } }
                
                    }
                }
                if (I <= shift)
                    {
                        x_0_data[num_x] = shift - I + P + 1; y_0_data[num_x] = P + 1; x_1_data[num_x] = shift - I + J + 1;  y_1_data[num_x] = J + 1;
                    }
                else
                    {
                        x_0_data[num_x] = P + 1; y_0_data[num_x] = I - shift + P + 1; x_1_data[num_x] = J + 1; y_1_data[num_x] = I - shift + J + 1;  
                    }

                score[num_x] = r1 * p1 / COL;
                num_x += threads_num;
            }

        }
};

void thread_pool_compcore(float* x, float* y, long long num_01, long long num_02, int shift, long long COL, float* score, int* x_0,int* x_1,int* y_0,int* y_1)
{
    unsigned int logicalCoreCount = sysconf(_SC_NPROCESSORS_ONLN);
    unsigned int thread_num = 4*logicalCoreCount;
    // cout<<"thread_num:\t"<<thread_num<<endl;
    ThreadPool pool(thread_num ,num_01,num_02,shift,COL,x,y,score,x_0,x_1,y_0,y_1);
}
