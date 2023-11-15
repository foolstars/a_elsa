#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <thread>
#include <atomic>
#include <chrono>

using namespace std;
namespace py = pybind11;

#define NN 32ll

void thread_pool_compcore(float* x, float* y, long long num_01, long long num_02, int shift, long long COL, float* score,int* x_0,int* x_1,int* y_0,int* y_1);

class	LSA
{
	public:
		vector<float>X;
		vector<float>Y;
		long long num_01;
		long long num_02;
		int shift;
		long long COL;

		vector<float>score;
		vector<int>x_0;
		vector<int>x_1;
		vector<int>y_0;
		vector<int>y_1;

	public:
		LSA(size_t total_num_01,size_t total_num_02) :num_01(NN), num_02(NN), shift(10), COL(50)//参数化列表
		{
			X.resize(num_01 * COL);
			Y.resize(num_02 * COL);

			X.insert(X.begin(), num_01 * COL, 0.1);
			Y.insert(Y.begin(), num_02 * COL, 0.1);
		}

		~LSA()
		{
			thread thread1([this]() { score.clear(); });

			thread thread2([this]() { x_0.clear(); });
			thread thread3([this]() { x_1.clear(); });
			thread thread4([this]() { y_0.clear(); });
			thread thread5([this]() { y_1.clear(); });

			thread1.join();
			thread2.join();
			thread3.join();
			thread4.join();
			thread5.join();
		}

	public:
		void assign(long long n_01, long long n_02, int s, long long col, vector<float>x, vector<float>y)
		{
			num_01 = n_01;
			num_02 = n_02;
			shift = s;
			COL = col;

			X.resize(num_01 * COL);
			Y.resize(num_02 * COL);

			X.insert(X.begin(), x.begin(), x.end());
			Y.insert(Y.begin(), y.begin(), y.end());
		}

		void dp_lsa()
		{
			float* ptr_x = X.data();
			float* ptr_y = Y.data();

			thread thread1([this]() { score.resize(num_01 * num_02); });
			thread thread2([this]() { x_0.resize(num_01 * num_02); });
			thread thread3([this]() { x_1.resize(num_01 * num_02); });
			thread thread4([this]() { y_0.resize(num_01 * num_02); });
			thread thread5([this]() { y_1.resize(num_01 * num_02); });

			thread1.join();
			thread2.join();
			thread3.join();
			thread4.join();
			thread5.join();

			float *score_data=score.data();

			int *x_0_data=x_0.data();
			int *x_1_data=x_1.data();
			int *y_0_data=y_0.data();
			int *y_1_data=y_1.data();

			thread_pool_compcore(ptr_x, ptr_y, num_01, num_02, shift, COL, score_data, x_0_data, x_1_data, y_0_data, y_1_data);
		}

		void lsa_clean() 
		{
			thread thread1([this]() { score.clear(); });

			thread thread2([this]() { x_0.clear(); });
			thread thread3([this]() { x_1.clear(); });
			thread thread4([this]() { y_0.clear(); });
			thread thread5([this]() { y_1.clear(); });

			thread1.join();
			thread2.join();
			thread3.join();
			thread4.join();
			thread5.join();
		}

};
