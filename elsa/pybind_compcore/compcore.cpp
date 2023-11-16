#include"compcore.hpp"

PYBIND11_MODULE(compcore, m)
{
	m.doc() = "pybind11 module of cpp compcore!";

	py::class_<LSA>(m, "LSA")
		.def(py::init<size_t, size_t>())
		.def("assign", &LSA::assign)
		.def("lsa_clean", &LSA::lsa_clean)

		.def_readwrite("num_01", &LSA::num_01)
		.def_readwrite("num_02", &LSA::num_02)
		.def_readwrite("shift", &LSA::shift)
		.def_readwrite("COL", &LSA::COL)
		.def_readwrite("X", &LSA::X)
		.def_readwrite("Y", &LSA::Y)
		.def_readwrite("score", &LSA::score)
		.def_readwrite("x_0", &LSA::x_0)
		.def_readwrite("x_1", &LSA::x_1)
		.def_readwrite("y_0", &LSA::y_0)
		.def_readwrite("y_1", &LSA::y_1)
			
		.def("dp_lsa", &LSA::dp_lsa);
}
