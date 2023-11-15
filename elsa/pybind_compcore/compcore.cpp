#include"compcore.hpp"

PYBIND11_MODULE(compcore, m)                     // 定义 Python 模块
{
	m.doc() = "pybind11 module of cpp compcore!";

	py::class_<LSA>(m, "LSA")                 	 // 定义 Python 类 LSA_Data
		.def(py::init<size_t, size_t>())		 // 添加构造函数 __init__()
		.def("assign", &LSA::assign)               // 添加成员函数 assign() 
		.def("lsa_clean", &LSA::lsa_clean)		   //添加成员函数 lsa_clean()

		.def_readwrite("num_01", &LSA::num_01)     // 添加成员变量 num_01，并提供读写权限
		.def_readwrite("num_02", &LSA::num_02)     // 添加成员变量 num_02，并提供读写权限
		.def_readwrite("shift", &LSA::shift)       // 添加成员变量 shift，并提供读写权限
		.def_readwrite("COL", &LSA::COL)           // 添加成员变量 COL，并提供读写权限
		.def_readwrite("X", &LSA::X)               // 添加成员变量 X，并提供读写权限
		.def_readwrite("Y", &LSA::Y)	           // 添加成员变量 Y，并提供读写权限
		.def_readwrite("score", &LSA::score)	           // 添加成员变量 Y，并提供读写权限
		.def_readwrite("x_0", &LSA::x_0)	           // 添加成员变量 Y，并提供读写权限
		.def_readwrite("x_1", &LSA::x_1)	           // 添加成员变量 Y，并提供读写权限
		.def_readwrite("y_0", &LSA::y_0)	           // 添加成员变量 Y，并提供读写权限
		.def_readwrite("y_1", &LSA::y_1)	           // 添加成员变量 Y，并提供读写权限
			
		.def("dp_lsa", &LSA::dp_lsa);						 // 定义函数 DP_lsa 并绑定到模块
}
