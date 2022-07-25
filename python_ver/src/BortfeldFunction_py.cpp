#include <cmath>
#include "parabolic_cylinder_function.h"
#include "bp.h"
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
using namespace pybind11::literals;
class Bortfeld {
public:
    Bortfeld() {}
    void bf_dose(double* z, double* idd_o, double* para, int data_size, int para_size)
    {
        // get dose
        BP::IDD_array_new(z, idd_o, para, data_size, para_size);
    }
    void bf_grad(double* z, double* grad_o, double* para, int data_size, int para_size)
    {
        // get Jacobian
        BP::get_jacobian(z, grad_o, para, data_size, para_size);
    }
};


PYBIND11_MODULE(Bortfeld, m) {
    m.doc() = "Bortfeld function module";
    py::class_<Bortfeld>(m, "Bortfeld")
        .def(py::init<>())
        .def("bf_dose", [](Bortfeld &m, std::vector<double> z, std::vector<double> para) {
            std::vector<double> idd_o(z.size());
            m.bf_dose(z.data(), idd_o.data(), para.data(), z.size(), para.size());
            return idd_o;
        })
        .def("bf_grad", [](Bortfeld &m, std::vector<double> z, std::vector<double> para) {
            std::vector<double> grad_o(z.size()*para.size());
            m.bf_grad(z.data(), grad_o.data(), para.data(), z.size(), para.size());
            return grad_o;
        });
}
