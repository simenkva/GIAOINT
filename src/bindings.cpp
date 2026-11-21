#include "giao_integrals/boys.hpp"
#include "cholesky.hpp"
#include "assembly.hpp"
#include "giao_integrals/derivatives.hpp"
#include "giao_integrals/eri.hpp"
#include "giao_integrals/nuclear.hpp"
#include "giao_integrals/overlap.hpp"
#include "giao_integrals/property.hpp"
#include "giao_integrals/types.hpp"
#include "giao_integrals/version.hpp"

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <cmath>
#include <memory>
#ifdef GIAO_HAS_OPENMP
#include <omp.h>
#if defined(__unix__) || defined(__APPLE__)
#include <dlfcn.h>
#endif
#endif
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

std::size_t threads_from_python(const py::object& value) {
    return py::module_::import("giao_integrals._threads")
        .attr("resolve_num_threads")(value).cast<std::size_t>();
}

giao::Vec3 vec3_from_python(const py::handle& value, const char* name) {
    py::array array = py::array::ensure(value);
    if (!array || array.ndim() != 1 || array.shape(0) != 3) {
        throw py::value_error(std::string(name) +
                              " must be a one-dimensional array with length 3");
    }
    py::array_t<double, py::array::c_style | py::array::forcecast> converted(array);
    const auto values = converted.unchecked<1>();
    return giao::Vec3{values(0), values(1), values(2)};
}
giao::CartesianExponent angular_from_python(const py::handle& value) {
    if (!PySequence_Check(value.ptr()) || PyUnicode_Check(value.ptr()) ||
        PyBytes_Check(value.ptr())) {
        throw py::value_error("angular momentum must contain exactly three integers");
    }
    py::sequence sequence = py::reinterpret_borrow<py::sequence>(value);
    if (py::len(sequence) != 3) {
        throw py::value_error("angular momentum must contain exactly three integers");
    }
    std::uint16_t result[3]{};
    for (py::ssize_t axis = 0; axis < 3; ++axis) {
        const auto item = sequence[axis];
        if (PyBool_Check(item.ptr()) || !PyLong_Check(item.ptr())) {
            throw py::value_error("angular momentum values must be integers");
        }
        const long value_as_long = PyLong_AsLong(item.ptr());
        if (value_as_long < 0 ||
            value_as_long > std::numeric_limits<std::uint16_t>::max()) {
            throw py::value_error(
                "angular momentum values must fit non-negative uint16");
        }
        result[axis] = static_cast<std::uint16_t>(value_as_long);
    }
    return {result[0], result[1], result[2]};
}

bool bool_from_python(const py::handle& value, const char* name) {
    if (!PyBool_Check(value.ptr())) {
        throw py::value_error(std::string(name) + " must be a bool");
    }
    return value.ptr() == Py_True;
}

std::uint16_t total_angular_from_python(const py::handle& value) {
    if (PyBool_Check(value.ptr()) || !PyLong_Check(value.ptr())) {
        throw py::value_error("shell angular momentum must be a non-negative integer");
    }
    const long result = PyLong_AsLong(value.ptr());
    if (result < 0 || result > std::numeric_limits<std::uint16_t>::max()) {
        throw py::value_error("shell angular momentum must fit non-negative uint16");
    }
    return static_cast<std::uint16_t>(result);
}

std::vector<double> vector_from_python(const py::handle& value, const char* name) {
    py::array array = py::array::ensure(value);
    if (!array || array.ndim() != 1) {
        throw py::value_error(std::string(name) + " must be one-dimensional");
    }
    py::array_t<double, py::array::c_style | py::array::forcecast> converted(array);
    const auto values = converted.unchecked<1>();
    std::vector<double> result(static_cast<std::size_t>(values.shape(0)));
    for (py::ssize_t index = 0; index < values.shape(0); ++index) {
        result[static_cast<std::size_t>(index)] = values(index);
    }
    return result;
}

giao::Shell shell_from_python(const py::handle& center,
                              const py::handle& angular_momentum,
                              const py::handle& exponents,
                              const py::handle& coefficients,
                              const py::handle& normalize) {
    auto exponent_values = vector_from_python(exponents, "exponents");
    py::array coefficient_array = py::array::ensure(coefficients);
    if (!coefficient_array ||
        (coefficient_array.ndim() != 1 && coefficient_array.ndim() != 2)) {
        throw py::value_error("coefficients must be one- or two-dimensional");
    }
    py::array_t<double, py::array::c_style | py::array::forcecast> converted(
        coefficient_array);
    const std::size_t contraction_count =
        converted.ndim() == 1 ? 1U
                              : static_cast<std::size_t>(converted.shape(0));
    const std::size_t primitive_count = static_cast<std::size_t>(
        converted.ndim() == 1 ? converted.shape(0) : converted.shape(1));
    if (primitive_count != exponent_values.size()) {
        throw py::value_error(
            "the final coefficient dimension must equal the exponent count");
    }
    std::vector<double> coefficient_values(contraction_count * primitive_count);
    if (converted.ndim() == 1) {
        const auto values = converted.unchecked<1>();
        for (std::size_t primitive = 0; primitive < primitive_count; ++primitive) {
            coefficient_values[primitive] =
                values(static_cast<py::ssize_t>(primitive));
        }
    } else {
        const auto values = converted.unchecked<2>();
        for (std::size_t contraction = 0; contraction < contraction_count;
             ++contraction) {
            for (std::size_t primitive = 0; primitive < primitive_count;
                 ++primitive) {
                coefficient_values[contraction * primitive_count + primitive] =
                    values(static_cast<py::ssize_t>(contraction),
                           static_cast<py::ssize_t>(primitive));
            }
        }
    }
    return giao::Shell(
        vec3_from_python(center, "center"),
        total_angular_from_python(angular_momentum), std::move(exponent_values),
        std::move(coefficient_values), contraction_count,
        bool_from_python(normalize, "normalize")
            ? giao::ContractionNormalization::normalize
            : giao::ContractionNormalization::as_provided);
}

giao::MagneticField field_or_zero(const py::object& field) {
    return field.is_none() ? giao::MagneticField{} : field.cast<giao::MagneticField>();
}

py::array validate_or_create_output(py::object output,
                                    const std::vector<py::ssize_t>& shape) {
    if (output.is_none()) {
        return py::array_t<giao::Complex>(shape);
    }
    if (!py::isinstance<py::array>(output)) {
        throw py::value_error("out must be a NumPy array");
    }
    py::array array = py::reinterpret_borrow<py::array>(output);
    if (!array.dtype().is(py::dtype::of<giao::Complex>())) {
        throw py::value_error("out must have dtype complex128");
    }
    if (array.ndim() != static_cast<py::ssize_t>(shape.size())) {
        throw py::value_error("out has an incorrect number of dimensions");
    }
    for (py::ssize_t axis = 0; axis < array.ndim(); ++axis) {
        if (array.shape(axis) != shape[static_cast<std::size_t>(axis)]) {
            throw py::value_error("out has an incorrect shape");
        }
    }
    if ((array.flags() & py::array::c_style) == 0) {
        throw py::value_error("out must be C-contiguous");
    }
    if (!array.writeable()) {
        throw py::value_error("out must be writable");
    }
    return array;
}

giao::Axis axis_from_python(const std::string& value) {
    if (value == "x") {
        return giao::Axis::x;
    }
    if (value == "y") {
        return giao::Axis::y;
    }
    if (value == "z") {
        return giao::Axis::z;
    }
    throw py::value_error("component must be 'x', 'y', or 'z'");
}

giao::CartesianMoment moment_from_python(const py::handle& powers,
                                         const py::handle& origin) {
    return giao::CartesianMoment(vec3_from_python(origin, "origin"),
                                 angular_from_python(powers));
}

template <typename Compute>
py::array shell_array(const giao::Shell& a, const giao::Shell& b,
                      py::object output, Compute&& compute) {
    py::array result = validate_or_create_output(
        output, {static_cast<py::ssize_t>(a.ao_count()),
                 static_cast<py::ssize_t>(b.ao_count())});
    auto* data = static_cast<giao::Complex*>(result.mutable_data());
    {
        py::gil_scoped_release release;
        giao::IntegralWorkspace workspace;
        compute(std::span<giao::Complex>(data, giao::shell_pair_size(a, b)),
                workspace);
    }
    return result;
}

template <typename Compute>
py::array basis_array(const giao::Basis& basis, py::object output,
                      Compute&& compute) {
    const auto count = basis.ao_count();
    py::array result = validate_or_create_output(
        output, {static_cast<py::ssize_t>(count),
                 static_cast<py::ssize_t>(count)});
    auto* data = static_cast<giao::Complex*>(result.mutable_data());
    {
        py::gil_scoped_release release;
        compute(std::span<giao::Complex>(data, count * count));
    }
    return result;
}

template <std::size_t count>
py::array derivative_values(const std::array<giao::Complex, count>& values,
                            const std::vector<py::ssize_t>& shape) {
    py::array_t<giao::Complex> result(shape);
    std::copy(values.begin(), values.end(), result.mutable_data());
    return result;
}

template <typename Compute>
py::array derivative_array(py::object output, const std::vector<py::ssize_t>& shape,
                           Compute&& compute) {
    py::array result = validate_or_create_output(std::move(output), shape);
    auto* data = static_cast<giao::Complex*>(result.mutable_data());
    std::size_t size = 1U;
    for (const auto extent : shape) {
        size *= static_cast<std::size_t>(extent);
    }
    {
        py::gil_scoped_release release;
        compute(std::span<giao::Complex>(data, size));
    }
    return result;
}

struct PackedEriData {
    std::vector<std::array<std::uint32_t, 4>> quartets;
    std::vector<std::array<std::size_t, 4>> shapes;
    std::vector<std::size_t> offsets{0U};
    std::vector<giao::Complex> values;
};

void append_eri_block(const giao::ShellQuartetBlockView& block,
                      void* user_data) {
    auto& packed = *static_cast<PackedEriData*>(user_data);
    packed.quartets.push_back(block.shells.as_array());
    packed.shapes.push_back(block.shape);
    packed.values.insert(packed.values.end(), block.values.begin(),
                         block.values.end());
    packed.offsets.push_back(packed.values.size());
}

}  // namespace

PYBIND11_MODULE(_giao_integrals, module) {
    module.doc() = "C++20 Cartesian GIAO/London Gaussian integrals";
    module.attr("__version__") = std::string(giao::version);

    py::register_exception<giao::BoysNumericalError>(module,
                                                      "BoysNumericalError");

    py::enum_<giao::BoysRegion>(module, "BoysRegion")
        .value("POWER_SERIES", giao::BoysRegion::power_series)
        .value("ADAPTIVE_QUADRATURE", giao::BoysRegion::adaptive_quadrature)
        .value("SCALED_QUADRATURE", giao::BoysRegion::scaled_quadrature)
        .value("POSITIVE_ASYMPTOTIC", giao::BoysRegion::positive_asymptotic);

    py::class_<giao::CartesianExponent>(module, "CartesianExponent")
        .def(py::init([](const py::object& values) {
            return angular_from_python(values);
        }))
        .def_property_readonly("x", [](const giao::CartesianExponent& value) {
            return value.x;
        })
        .def_property_readonly("y", [](const giao::CartesianExponent& value) {
            return value.y;
        })
        .def_property_readonly("z", [](const giao::CartesianExponent& value) {
            return value.z;
        })
        .def_property_readonly("total", &giao::CartesianExponent::total)
        .def("__iter__", [](const giao::CartesianExponent& value) {
            return py::iter(py::make_tuple(value.x, value.y, value.z));
        });

    py::class_<giao::MagneticField>(module, "MagneticField")
        .def(py::init([](const py::object& B, const py::object& gauge_origin) {
                 return giao::MagneticField(vec3_from_python(B, "B"),
                                            vec3_from_python(gauge_origin,
                                                             "gauge_origin"));
             }),
             py::arg("B") = py::make_tuple(0.0, 0.0, 0.0),
             py::arg("gauge_origin") = py::make_tuple(0.0, 0.0, 0.0))
        .def_property_readonly("B", [](const giao::MagneticField& field) {
            return py::make_tuple(field.B.x, field.B.y, field.B.z);
        })
        .def_property_readonly("gauge_origin", [](const giao::MagneticField& field) {
            return py::make_tuple(field.gauge_origin.x, field.gauge_origin.y,
                                  field.gauge_origin.z);
        })
        .def("london_wave_vector", [](const giao::MagneticField& field,
                                      const py::object& center) {
            const auto value = field.london_wave_vector(
                vec3_from_python(center, "center"));
            return py::make_tuple(value.x, value.y, value.z);
        });

    py::class_<giao::Nucleus>(module, "Nucleus")
        .def(py::init([](double charge, const py::object& center) {
                 return giao::Nucleus(charge,
                                      vec3_from_python(center, "center"));
             }),
             py::arg("charge"), py::arg("center"))
        .def_readonly("charge", &giao::Nucleus::charge)
        .def_property_readonly("center", [](const giao::Nucleus& nucleus) {
            return py::make_tuple(nucleus.center.x, nucleus.center.y,
                                  nucleus.center.z);
        });

    py::class_<giao::PrimitiveGaussian>(module, "PrimitiveGaussian")
        .def(py::init([](double exponent, const py::object& center,
                         const py::object& angular, double coefficient,
                         const py::object& normalized) {
                 return giao::PrimitiveGaussian(
                     exponent, vec3_from_python(center, "center"),
                     angular_from_python(angular), coefficient,
                     bool_from_python(normalized, "normalized"));
             }),
             py::arg("exponent"), py::arg("center"),
             py::arg("angular") = py::make_tuple(0, 0, 0),
             py::arg("coefficient") = 1.0, py::arg("normalized") = true)
        .def_readonly("exponent", &giao::PrimitiveGaussian::exponent)
        .def_readonly("coefficient", &giao::PrimitiveGaussian::coefficient)
        .def_property_readonly("center", [](const giao::PrimitiveGaussian& primitive) {
            return py::make_tuple(primitive.center.x, primitive.center.y,
                                  primitive.center.z);
        })
        .def_property_readonly("angular", [](const giao::PrimitiveGaussian& primitive) {
            return py::make_tuple(primitive.angular.x, primitive.angular.y,
                                  primitive.angular.z);
        })
        .def_readonly("normalized", &giao::PrimitiveGaussian::normalized);

    py::class_<giao::Shell>(module, "Shell")
        .def(py::init(&shell_from_python), py::arg("center"),
             py::arg("angular_momentum"), py::arg("exponents"),
             py::arg("coefficients"), py::kw_only(), py::arg("normalize") = true)
        .def_property_readonly("center", [](const giao::Shell& shell) {
            const auto value = shell.center();
            return py::make_tuple(value.x, value.y, value.z);
        })
        .def_property_readonly("angular_momentum", &giao::Shell::angular_momentum)
        .def_property_readonly("primitive_count", &giao::Shell::primitive_count)
        .def_property_readonly("contraction_count", &giao::Shell::contraction_count)
        .def_property_readonly("cartesian_count", &giao::Shell::cartesian_count)
        .def_property_readonly("ao_count", &giao::Shell::ao_count)
        .def_property_readonly("components", [](const giao::Shell& shell) {
            py::tuple result(shell.cartesian_count());
            for (std::size_t index = 0; index < shell.cartesian_count(); ++index) {
                const auto value = shell.components()[index];
                result[index] = py::make_tuple(value.x, value.y, value.z);
            }
            return result;
        });

    py::class_<giao::Basis>(module, "Basis")
        .def(py::init<std::vector<giao::Shell>>(), py::arg("shells"))
        .def_property_readonly("shells", [](const giao::Basis& basis) {
            return std::vector<giao::Shell>(basis.shells().begin(),
                                            basis.shells().end());
        })
        .def_property_readonly("ao_offsets", [](const giao::Basis& basis) {
            return py::tuple(py::cast(std::vector<std::size_t>(
                basis.ao_offsets().begin(), basis.ao_offsets().end())));
        })
        .def_property_readonly("ao_count", &giao::Basis::ao_count);

    py::class_<giao::EriSchwarzBounds>(module, "_EriSchwarzBounds")
        .def(py::init([](const giao::Basis& basis, const py::object& field) {
                 const auto field_value = field_or_zero(field);
                 py::gil_scoped_release release;
                 return giao::EriSchwarzBounds(basis, field_value);
             }),
             py::arg("basis"), py::kw_only(), py::arg("field") = py::none())
        .def("bound", &giao::EriSchwarzBounds::operator())
        .def_property_readonly("values", [](const giao::EriSchwarzBounds& bounds) {
            const auto count = bounds.shell_count();
            py::array_t<double> result(
                {static_cast<py::ssize_t>(count),
                 static_cast<py::ssize_t>(count)});
            std::copy(bounds.values().begin(), bounds.values().end(),
                      result.mutable_data());
            return result;
        });

    module.def("cartesian_components", [](const py::object& angular_momentum) {
        const auto values = giao::cartesian_components(
            total_angular_from_python(angular_momentum));
        py::tuple result(values.size());
        for (std::size_t index = 0; index < values.size(); ++index) {
            result[index] =
                py::make_tuple(values[index].x, values[index].y, values[index].z);
        }
        return result;
    });

    module.def("primitive_normalization",
               [](double exponent, const py::object& angular) {
                   return giao::primitive_normalization(
                       exponent, angular_from_python(angular));
               });

    module.def(
        "boys",
        [](giao::Complex argument, std::size_t maximum_order, bool scaled) {
            py::array_t<giao::Complex> result(maximum_order + 1U);
            auto* data = result.mutable_data();
            {
                py::gil_scoped_release release;
                [[maybe_unused]] const auto diagnostics = giao::compute_boys(
                    argument,
                    std::span<giao::Complex>(data, maximum_order + 1U),
                    scaled ? giao::BoysScaling::exp_z
                           : giao::BoysScaling::unscaled);
            }
            return result;
        },
        py::arg("argument"), py::arg("maximum_order"), py::kw_only(),
        py::arg("scaled") = false);

    module.def(
        "boys_with_diagnostics",
        [](giao::Complex argument, std::size_t maximum_order, bool scaled) {
            py::array_t<giao::Complex> result(maximum_order + 1U);
            giao::BoysDiagnostics diagnostics;
            {
                py::gil_scoped_release release;
                diagnostics = giao::compute_boys(
                    argument,
                    std::span<giao::Complex>(result.mutable_data(),
                                             maximum_order + 1U),
                    scaled ? giao::BoysScaling::exp_z
                           : giao::BoysScaling::unscaled);
            }
            py::dict details;
            details["region"] = py::cast(diagnostics.region);
            details["estimated_absolute_error"] =
                diagnostics.estimated_absolute_error;
            details["quadrature_segments"] = diagnostics.quadrature_segments;
            details["scaled"] = scaled;
            return py::make_tuple(std::move(result), std::move(details));
        },
        py::arg("argument"), py::arg("maximum_order"), py::kw_only(),
        py::arg("scaled") = false);

    module.def(
        "primitive_overlap",
        [](const giao::PrimitiveGaussian& bra,
           const giao::PrimitiveGaussian& ket, const py::object& field) {
            const auto field_value = field_or_zero(field);
            py::gil_scoped_release release;
            return giao::primitive_overlap(bra, ket, field_value);
        },
        py::arg("bra"), py::arg("ket"), py::kw_only(),
        py::arg("field") = py::none());

    module.def(
        "overlap_shell",
        [](const giao::Shell& a, const giao::Shell& b, const py::object& field,
           py::object output) {
            py::array result = validate_or_create_output(
                output, {static_cast<py::ssize_t>(a.ao_count()),
                         static_cast<py::ssize_t>(b.ao_count())});
            auto* data = static_cast<giao::Complex*>(result.mutable_data());
            const auto field_value = field_or_zero(field);
            {
                py::gil_scoped_release release;
                giao::IntegralWorkspace workspace;
                giao::compute_overlap(
                    a, b, field_value,
                    std::span<giao::Complex>(data, giao::shell_pair_size(a, b)),
                    workspace);
            }
            return result;
        },
        py::arg("a"), py::arg("b"), py::kw_only(),
        py::arg("field") = py::none(), py::arg("out") = py::none());

    module.def(
        "overlap",
        [](const giao::Basis& basis, const py::object& field, py::object output) {
            const auto count = basis.ao_count();
            py::array result = validate_or_create_output(
                output, {static_cast<py::ssize_t>(count),
                         static_cast<py::ssize_t>(count)});
            auto* data = static_cast<giao::Complex*>(result.mutable_data());
            const auto field_value = field_or_zero(field);
            {
                py::gil_scoped_release release;
                giao::compute_overlap_matrix(
                    basis, field_value,
                    std::span<giao::Complex>(data, count * count));
            }
            return result;
        },
        py::arg("basis"), py::kw_only(), py::arg("field") = py::none(),
        py::arg("out") = py::none());

    module.def(
        "primitive_nuclear_attraction",
        [](const giao::PrimitiveGaussian& bra,
           const giao::PrimitiveGaussian& ket,
           const std::vector<giao::Nucleus>& nuclei,
           const py::object& field) {
            const auto field_value = field_or_zero(field);
            py::gil_scoped_release release;
            return giao::primitive_nuclear_attraction(bra, ket, nuclei,
                                                       field_value);
        },
        py::arg("bra"), py::arg("ket"), py::arg("nuclei"), py::kw_only(),
        py::arg("field") = py::none());

    module.def(
        "nuclear_attraction_shell",
        [](const giao::Shell& a, const giao::Shell& b,
           const std::vector<giao::Nucleus>& nuclei,
           const py::object& field, py::object output) {
            py::array result = validate_or_create_output(
                output, {static_cast<py::ssize_t>(a.ao_count()),
                         static_cast<py::ssize_t>(b.ao_count())});
            auto* data = static_cast<giao::Complex*>(result.mutable_data());
            const auto field_value = field_or_zero(field);
            {
                py::gil_scoped_release release;
                giao::NuclearAttractionWorkspace workspace;
                giao::compute_nuclear_attraction(
                    a, b, nuclei, field_value,
                    std::span<giao::Complex>(data, giao::shell_pair_size(a, b)),
                    workspace);
            }
            return result;
        },
        py::arg("a"), py::arg("b"), py::arg("nuclei"), py::kw_only(),
        py::arg("field") = py::none(), py::arg("out") = py::none());

    module.def(
        "nuclear_attraction",
        [](const giao::Basis& basis,
           const std::vector<giao::Nucleus>& nuclei,
           const py::object& field, py::object output) {
            const auto count = basis.ao_count();
            py::array result = validate_or_create_output(
                output, {static_cast<py::ssize_t>(count),
                         static_cast<py::ssize_t>(count)});
            auto* data = static_cast<giao::Complex*>(result.mutable_data());
            const auto field_value = field_or_zero(field);
            {
                py::gil_scoped_release release;
                giao::compute_nuclear_attraction_matrix(
                    basis, nuclei, field_value,
                    std::span<giao::Complex>(data, count * count));
            }
            return result;
        },
        py::arg("basis"), py::arg("nuclei"), py::kw_only(),
        py::arg("field") = py::none(), py::arg("out") = py::none());

    module.def(
        "primitive_eri",
        [](const giao::PrimitiveGaussian& a,
           const giao::PrimitiveGaussian& b,
           const giao::PrimitiveGaussian& c,
           const giao::PrimitiveGaussian& d, const py::object& field) {
            const auto field_value = field_or_zero(field);
            py::gil_scoped_release release;
            return giao::primitive_eri(a, b, c, d, field_value);
        },
        py::arg("a"), py::arg("b"), py::arg("c"), py::arg("d"),
        py::kw_only(), py::arg("field") = py::none());

    module.def(
        "eri_shell",
        [](const giao::Shell& a, const giao::Shell& b,
           const giao::Shell& c, const giao::Shell& d,
           const py::object& field, py::object output) {
            const std::vector<py::ssize_t> shape{
                static_cast<py::ssize_t>(a.ao_count()),
                static_cast<py::ssize_t>(b.ao_count()),
                static_cast<py::ssize_t>(c.ao_count()),
                static_cast<py::ssize_t>(d.ao_count())};
            py::array result = validate_or_create_output(output, shape);
            auto* data = static_cast<giao::Complex*>(result.mutable_data());
            const auto field_value = field_or_zero(field);
            {
                py::gil_scoped_release release;
                giao::EriWorkspace workspace;
                giao::compute_eri(
                    a, b, c, d, field_value,
                    std::span<giao::Complex>(
                        data, giao::shell_quartet_size(a, b, c, d)),
                    workspace);
            }
            return result;
        },
        py::arg("a"), py::arg("b"), py::arg("c"), py::arg("d"),
        py::kw_only(), py::arg("field") = py::none(),
        py::arg("out") = py::none());

    module.def(
        "_eri_full",
        [](const giao::Basis& basis, const py::object& field,
           double screening_threshold, const py::object& thread_request) {
            const auto threads = threads_from_python(thread_request);
            const auto count = basis.ao_count();
            py::array_t<giao::Complex> result(
                {static_cast<py::ssize_t>(count),
                 static_cast<py::ssize_t>(count),
                 static_cast<py::ssize_t>(count),
                 static_cast<py::ssize_t>(count)});
            const auto field_value = field_or_zero(field);
            {
                py::gil_scoped_release release;
                const std::optional<giao::EriSchwarzBounds> bounds =
                    screening_threshold > 0.0
                        ? std::optional<giao::EriSchwarzBounds>(
                              std::in_place, basis, field_value)
                        : std::nullopt;
                [[maybe_unused]] const auto statistics =
                    giao::compute_eri_tensor(
                        basis, field_value,
                        {screening_threshold, threads},
                        bounds ? &*bounds : nullptr,
                        std::span<giao::Complex>(result.mutable_data(),
                                                 result.size()));
            }
            return result;
        },
        py::arg("basis"), py::kw_only(), py::arg("field") = py::none(),
        py::arg("screening_threshold") = 0.0, py::arg("threads") = py::none());

    module.def(
        "_eri_batch",
        [](const giao::Basis& basis,
           const std::vector<std::array<std::uint32_t, 4>>& quartet_arrays,
           const py::object& field, double screening_threshold,
           const py::object& thread_request, const giao::EriSchwarzBounds* bounds) {
            const auto threads = threads_from_python(thread_request);
            std::vector<giao::ShellQuartetIndex> quartets;
            quartets.reserve(quartet_arrays.size());
            for (const auto& value : quartet_arrays) {
                quartets.push_back({value[0], value[1], value[2], value[3]});
            }
            PackedEriData packed;
            giao::EriStatistics statistics;
            const auto field_value = field_or_zero(field);
            {
                py::gil_scoped_release release;
                statistics = giao::evaluate_eri_shell_quartets(
                    basis, quartets, field_value,
                    {screening_threshold, threads}, bounds, append_eri_block,
                    &packed);
            }
            py::array_t<std::uint32_t> quartet_result(
                {static_cast<py::ssize_t>(packed.quartets.size()),
                 py::ssize_t{4}});
            py::array_t<std::int64_t> shape_result(
                {static_cast<py::ssize_t>(packed.shapes.size()),
                 py::ssize_t{4}});
            py::array_t<std::int64_t> offset_result(packed.offsets.size());
            py::array_t<giao::Complex> value_result(packed.values.size());
            auto quartet_view = quartet_result.mutable_unchecked<2>();
            auto shape_view = shape_result.mutable_unchecked<2>();
            for (std::size_t row = 0; row < packed.quartets.size(); ++row) {
                for (std::size_t axis = 0; axis < 4; ++axis) {
                    quartet_view(static_cast<py::ssize_t>(row),
                                  static_cast<py::ssize_t>(axis)) =
                        packed.quartets[row][axis];
                    shape_view(static_cast<py::ssize_t>(row),
                               static_cast<py::ssize_t>(axis)) =
                        static_cast<std::int64_t>(packed.shapes[row][axis]);
                }
            }
            auto* offset_data = offset_result.mutable_data();
            for (std::size_t index = 0; index < packed.offsets.size(); ++index) {
                offset_data[index] =
                    static_cast<std::int64_t>(packed.offsets[index]);
            }
            std::copy(packed.values.begin(), packed.values.end(),
                      value_result.mutable_data());
            return py::make_tuple(std::move(quartet_result),
                                  std::move(shape_result),
                                  std::move(offset_result),
                                  std::move(value_result),
                                  statistics.requested_quartets,
                                  statistics.screened_quartets);
        },
        py::arg("basis"), py::arg("quartets"), py::kw_only(),
        py::arg("field") = py::none(),
        py::arg("screening_threshold") = 0.0, py::arg("threads") = py::none(),
        py::arg("bounds") = py::none());

    module.def(
        "_canonical_shell_quartet",
        [](std::uint32_t a, std::uint32_t b, std::uint32_t c,
           std::uint32_t d) {
            const auto canonical = giao::canonicalize_shell_quartet({a, b, c, d});
            const auto indices = canonical.shells.as_array();
            return py::make_tuple(
                py::make_tuple(indices[0], indices[1], indices[2], indices[3]),
                canonical.conjugate);
        },
        py::arg("a"), py::arg("b"), py::arg("c"), py::arg("d"));

    module.def("_build_info", []() {
        py::dict result;
        result["version"] = giao::version;
        result["compiler"] = GIAO_COMPILER;
        result["openmp_enabled"] = giao::openmp_enabled();
        result["openmp_runtime_path"] = py::none();
#ifdef GIAO_HAS_OPENMP
#if defined(__unix__) || defined(__APPLE__)
        Dl_info info{};
        if (dladdr(reinterpret_cast<const void*>(&omp_get_max_threads), &info)
            != 0 && info.dli_fname != nullptr) {
            result["openmp_runtime_path"] = std::string(info.dli_fname);
        }
#endif
#endif
        return result;
    });

    module.def("openmp_enabled", &giao::openmp_enabled);
    module.def("openmp_max_threads", &giao::openmp_max_threads);

    module.def(
        "primitive_moment",
        [](const giao::PrimitiveGaussian& bra,
           const giao::PrimitiveGaussian& ket, const py::object& powers,
           const py::object& origin, const py::object& field) {
            const auto moment = moment_from_python(powers, origin);
            const auto field_value = field_or_zero(field);
            py::gil_scoped_release release;
            giao::IntegralWorkspace workspace;
            return giao::primitive_moment(bra, ket, moment, field_value,
                                          workspace);
        },
        py::arg("bra"), py::arg("ket"), py::arg("powers"), py::kw_only(),
        py::arg("origin") = py::make_tuple(0.0, 0.0, 0.0),
        py::arg("field") = py::none());

    module.def(
        "primitive_gradient",
        [](const giao::PrimitiveGaussian& bra,
           const giao::PrimitiveGaussian& ket, const std::string& component,
           const py::object& field) {
            const auto axis = axis_from_python(component);
            const auto field_value = field_or_zero(field);
            py::gil_scoped_release release;
            giao::IntegralWorkspace workspace;
            return giao::primitive_gradient(bra, ket, axis, field_value,
                                            workspace);
        },
        py::arg("bra"), py::arg("ket"), py::arg("component"), py::kw_only(),
        py::arg("field") = py::none());

    module.def(
        "primitive_momentum",
        [](const giao::PrimitiveGaussian& bra,
           const giao::PrimitiveGaussian& ket, const std::string& component,
           const py::object& field) {
            const auto axis = axis_from_python(component);
            const auto field_value = field_or_zero(field);
            py::gil_scoped_release release;
            giao::IntegralWorkspace workspace;
            return giao::primitive_momentum(bra, ket, axis, field_value,
                                            workspace);
        },
        py::arg("bra"), py::arg("ket"), py::arg("component"), py::kw_only(),
        py::arg("field") = py::none());

    module.def(
        "primitive_kinetic",
        [](const giao::PrimitiveGaussian& bra,
           const giao::PrimitiveGaussian& ket, const py::object& field) {
            const auto field_value = field_or_zero(field);
            py::gil_scoped_release release;
            giao::IntegralWorkspace workspace;
            return giao::primitive_kinetic(bra, ket, field_value, workspace);
        },
        py::arg("bra"), py::arg("ket"), py::kw_only(),
        py::arg("field") = py::none());

    module.def(
        "primitive_magnetic_kinetic",
        [](const giao::PrimitiveGaussian& bra,
           const giao::PrimitiveGaussian& ket, const py::object& field) {
            const auto field_value = field_or_zero(field);
            py::gil_scoped_release release;
            giao::IntegralWorkspace workspace;
            return giao::primitive_magnetic_kinetic(bra, ket, field_value,
                                                     workspace);
        },
        py::arg("bra"), py::arg("ket"), py::kw_only(),
        py::arg("field") = py::none());

    module.def(
        "primitive_overlap_center_derivatives",
        [](const giao::PrimitiveGaussian& bra, const giao::PrimitiveGaussian& ket,
           const py::object& field) {
            const auto field_value = field_or_zero(field);
            std::array<giao::Complex, 6> values;
            {
                py::gil_scoped_release release;
                values =
                    giao::primitive_overlap_center_derivatives(bra, ket, field_value);
            }
            return derivative_values(values, {2, 3});
        },
        py::arg("bra"), py::arg("ket"), py::kw_only(), py::arg("field") = py::none());

    module.def(
        "primitive_overlap_magnetic_derivatives",
        [](const giao::PrimitiveGaussian& bra, const giao::PrimitiveGaussian& ket,
           const py::object& field) {
            const auto field_value = field_or_zero(field);
            std::array<giao::Complex, 3> values;
            {
                py::gil_scoped_release release;
                values =
                    giao::primitive_overlap_magnetic_derivatives(bra, ket, field_value);
            }
            return derivative_values(values, {3});
        },
        py::arg("bra"), py::arg("ket"), py::kw_only(), py::arg("field") = py::none());

    module.def(
        "primitive_kinetic_center_derivatives",
        [](const giao::PrimitiveGaussian& bra, const giao::PrimitiveGaussian& ket,
           const py::object& field) {
            const auto field_value = field_or_zero(field);
            std::array<giao::Complex, 6> values;
            {
                py::gil_scoped_release release;
                values =
                    giao::primitive_kinetic_center_derivatives(bra, ket, field_value);
            }
            return derivative_values(values, {2, 3});
        },
        py::arg("bra"), py::arg("ket"), py::kw_only(), py::arg("field") = py::none());

    module.def(
        "primitive_kinetic_magnetic_derivatives",
        [](const giao::PrimitiveGaussian& bra, const giao::PrimitiveGaussian& ket,
           const py::object& field) {
            const auto field_value = field_or_zero(field);
            std::array<giao::Complex, 3> values;
            {
                py::gil_scoped_release release;
                values =
                    giao::primitive_kinetic_magnetic_derivatives(bra, ket, field_value);
            }
            return derivative_values(values, {3});
        },
        py::arg("bra"), py::arg("ket"), py::kw_only(), py::arg("field") = py::none());

    module.def(
        "primitive_magnetic_kinetic_center_derivatives",
        [](const giao::PrimitiveGaussian& bra, const giao::PrimitiveGaussian& ket,
           const py::object& field) {
            const auto field_value = field_or_zero(field);
            std::array<giao::Complex, 6> values;
            {
                py::gil_scoped_release release;
                values = giao::primitive_magnetic_kinetic_center_derivatives(
                    bra, ket, field_value);
            }
            return derivative_values(values, {2, 3});
        },
        py::arg("bra"), py::arg("ket"), py::kw_only(), py::arg("field") = py::none());

    module.def(
        "primitive_magnetic_kinetic_magnetic_derivatives",
        [](const giao::PrimitiveGaussian& bra, const giao::PrimitiveGaussian& ket,
           const py::object& field) {
            const auto field_value = field_or_zero(field);
            std::array<giao::Complex, 3> values;
            {
                py::gil_scoped_release release;
                values = giao::primitive_magnetic_kinetic_magnetic_derivatives(
                    bra, ket, field_value);
            }
            return derivative_values(values, {3});
        },
        py::arg("bra"), py::arg("ket"), py::kw_only(), py::arg("field") = py::none());

    module.def(
        "primitive_nuclear_attraction_center_derivatives",
        [](const giao::PrimitiveGaussian& bra, const giao::PrimitiveGaussian& ket,
           const std::vector<giao::Nucleus>& nuclei, const py::object& field) {
            const auto field_value = field_or_zero(field);
            std::array<giao::Complex, 6> values;
            {
                py::gil_scoped_release release;
                values = giao::primitive_nuclear_attraction_center_derivatives(
                    bra, ket, nuclei, field_value);
            }
            return derivative_values(values, {2, 3});
        },
        py::arg("bra"), py::arg("ket"), py::arg("nuclei"), py::kw_only(),
        py::arg("field") = py::none());

    module.def(
        "primitive_nuclear_attraction_magnetic_derivatives",
        [](const giao::PrimitiveGaussian& bra, const giao::PrimitiveGaussian& ket,
           const std::vector<giao::Nucleus>& nuclei, const py::object& field) {
            const auto field_value = field_or_zero(field);
            std::array<giao::Complex, 3> values;
            {
                py::gil_scoped_release release;
                values = giao::primitive_nuclear_attraction_magnetic_derivatives(
                    bra, ket, nuclei, field_value);
            }
            return derivative_values(values, {3});
        },
        py::arg("bra"), py::arg("ket"), py::arg("nuclei"), py::kw_only(),
        py::arg("field") = py::none());

    module.def(
        "primitive_nuclear_attraction_nucleus_derivatives",
        [](const giao::PrimitiveGaussian& bra, const giao::PrimitiveGaussian& ket,
           const std::vector<giao::Nucleus>& nuclei, const py::object& field) {
            const auto field_value = field_or_zero(field);
            py::array_t<giao::Complex> result(
                {static_cast<py::ssize_t>(nuclei.size()), py::ssize_t{3}});
            {
                py::gil_scoped_release release;
                giao::primitive_nuclear_attraction_nucleus_derivatives(
                    bra, ket, nuclei, field_value,
                    std::span<giao::Complex>(result.mutable_data(),
                                             nuclei.size() * 3U));
            }
            return result;
        },
        py::arg("bra"), py::arg("ket"), py::arg("nuclei"), py::kw_only(),
        py::arg("field") = py::none());

    module.def(
        "primitive_eri_center_derivatives",
        [](const giao::PrimitiveGaussian& a, const giao::PrimitiveGaussian& b,
           const giao::PrimitiveGaussian& c, const giao::PrimitiveGaussian& d,
           const py::object& field) {
            const auto field_value = field_or_zero(field);
            std::array<giao::Complex, 12> values;
            {
                py::gil_scoped_release release;
                values =
                    giao::primitive_eri_center_derivatives(a, b, c, d, field_value);
            }
            return derivative_values(values, {4, 3});
        },
        py::arg("a"), py::arg("b"), py::arg("c"), py::arg("d"), py::kw_only(),
        py::arg("field") = py::none());

    module.def(
        "primitive_eri_magnetic_derivatives",
        [](const giao::PrimitiveGaussian& a, const giao::PrimitiveGaussian& b,
           const giao::PrimitiveGaussian& c, const giao::PrimitiveGaussian& d,
           const py::object& field) {
            const auto field_value = field_or_zero(field);
            std::array<giao::Complex, 3> values;
            {
                py::gil_scoped_release release;
                values =
                    giao::primitive_eri_magnetic_derivatives(a, b, c, d, field_value);
            }
            return derivative_values(values, {3});
        },
        py::arg("a"), py::arg("b"), py::arg("c"), py::arg("d"), py::kw_only(),
        py::arg("field") = py::none());

    module.def(
        "moment_shell",
        [](const giao::Shell& a, const giao::Shell& b,
           const py::object& powers, const py::object& origin,
           const py::object& field, py::object output) {
            const auto moment = moment_from_python(powers, origin);
            const auto field_value = field_or_zero(field);
            return shell_array(a, b, std::move(output),
                               [&](auto values, auto& workspace) {
                                   giao::compute_moment(a, b, moment, field_value,
                                                        values, workspace);
                               });
        },
        py::arg("a"), py::arg("b"), py::arg("powers"), py::kw_only(),
        py::arg("origin") = py::make_tuple(0.0, 0.0, 0.0),
        py::arg("field") = py::none(), py::arg("out") = py::none());

    module.def(
        "moment",
        [](const giao::Basis& basis, const py::object& powers,
           const py::object& origin, const py::object& field,
           py::object output) {
            const auto moment = moment_from_python(powers, origin);
            const auto field_value = field_or_zero(field);
            return basis_array(basis, std::move(output), [&](auto values) {
                giao::compute_moment_matrix(basis, moment, field_value, values);
            });
        },
        py::arg("basis"), py::arg("powers"), py::kw_only(),
        py::arg("origin") = py::make_tuple(0.0, 0.0, 0.0),
        py::arg("field") = py::none(), py::arg("out") = py::none());

    module.def(
        "gradient_shell",
        [](const giao::Shell& a, const giao::Shell& b,
           const std::string& component, const py::object& field,
           py::object output) {
            const auto axis = axis_from_python(component);
            const auto field_value = field_or_zero(field);
            return shell_array(a, b, std::move(output),
                               [&](auto values, auto& workspace) {
                                   giao::compute_gradient(a, b, axis, field_value,
                                                          values, workspace);
                               });
        },
        py::arg("a"), py::arg("b"), py::arg("component"), py::kw_only(),
        py::arg("field") = py::none(), py::arg("out") = py::none());

    module.def(
        "gradient",
        [](const giao::Basis& basis, const std::string& component,
           const py::object& field, py::object output) {
            const auto axis = axis_from_python(component);
            const auto field_value = field_or_zero(field);
            return basis_array(basis, std::move(output), [&](auto values) {
                giao::compute_gradient_matrix(basis, axis, field_value, values);
            });
        },
        py::arg("basis"), py::arg("component"), py::kw_only(),
        py::arg("field") = py::none(), py::arg("out") = py::none());

    module.def(
        "momentum_shell",
        [](const giao::Shell& a, const giao::Shell& b,
           const std::string& component, const py::object& field,
           py::object output) {
            const auto axis = axis_from_python(component);
            const auto field_value = field_or_zero(field);
            return shell_array(a, b, std::move(output),
                               [&](auto values, auto& workspace) {
                                   giao::compute_momentum(a, b, axis, field_value,
                                                          values, workspace);
                               });
        },
        py::arg("a"), py::arg("b"), py::arg("component"), py::kw_only(),
        py::arg("field") = py::none(), py::arg("out") = py::none());

    module.def(
        "momentum",
        [](const giao::Basis& basis, const std::string& component,
           const py::object& field, py::object output) {
            const auto axis = axis_from_python(component);
            const auto field_value = field_or_zero(field);
            return basis_array(basis, std::move(output), [&](auto values) {
                giao::compute_momentum_matrix(basis, axis, field_value, values);
            });
        },
        py::arg("basis"), py::arg("component"), py::kw_only(),
        py::arg("field") = py::none(), py::arg("out") = py::none());

    module.def(
        "kinetic_shell",
        [](const giao::Shell& a, const giao::Shell& b,
           const py::object& field, py::object output) {
            const auto field_value = field_or_zero(field);
            return shell_array(a, b, std::move(output),
                               [&](auto values, auto& workspace) {
                                   giao::compute_kinetic(a, b, field_value,
                                                         values, workspace);
                               });
        },
        py::arg("a"), py::arg("b"), py::kw_only(),
        py::arg("field") = py::none(), py::arg("out") = py::none());

    module.def(
        "kinetic",
        [](const giao::Basis& basis, const py::object& field,
           py::object output) {
            const auto field_value = field_or_zero(field);
            return basis_array(basis, std::move(output), [&](auto values) {
                giao::compute_kinetic_matrix(basis, field_value, values);
            });
        },
        py::arg("basis"), py::kw_only(), py::arg("field") = py::none(),
        py::arg("out") = py::none());

    module.def(
        "magnetic_kinetic_shell",
        [](const giao::Shell& a, const giao::Shell& b,
           const py::object& field, py::object output) {
            const auto field_value = field_or_zero(field);
            return shell_array(a, b, std::move(output),
                               [&](auto values, auto& workspace) {
                                   giao::compute_magnetic_kinetic(
                                       a, b, field_value, values, workspace);
                               });
        },
        py::arg("a"), py::arg("b"), py::kw_only(),
        py::arg("field") = py::none(), py::arg("out") = py::none());

    module.def(
        "magnetic_kinetic",
        [](const giao::Basis& basis, const py::object& field,
           py::object output) {
            const auto field_value = field_or_zero(field);
            return basis_array(basis, std::move(output), [&](auto values) {
                giao::compute_magnetic_kinetic_matrix(basis, field_value,
                                                       values);
            });
        },
        py::arg("basis"), py::kw_only(), py::arg("field") = py::none(),
        py::arg("out") = py::none());
    module.def(
        "overlap_center_derivatives_shell",
        [](const giao::Shell& a, const giao::Shell& b, const py::object& field,
           py::object output) {
            const auto field_value = field_or_zero(field);
            return derivative_array(std::move(output),
                                    {2, 3, static_cast<py::ssize_t>(a.ao_count()),
                                     static_cast<py::ssize_t>(b.ao_count())},
                                    [&](auto values) {
                                        giao::compute_overlap_center_derivatives(
                                            a, b, field_value, values);
                                    });
        },
        py::arg("a"), py::arg("b"), py::kw_only(), py::arg("field") = py::none(),
        py::arg("out") = py::none());

    module.def(
        "overlap_magnetic_derivatives_shell",
        [](const giao::Shell& a, const giao::Shell& b, const py::object& field,
           py::object output) {
            const auto field_value = field_or_zero(field);
            return derivative_array(std::move(output),
                                    {3, static_cast<py::ssize_t>(a.ao_count()),
                                     static_cast<py::ssize_t>(b.ao_count())},
                                    [&](auto values) {
                                        giao::compute_overlap_magnetic_derivatives(
                                            a, b, field_value, values);
                                    });
        },
        py::arg("a"), py::arg("b"), py::kw_only(), py::arg("field") = py::none(),
        py::arg("out") = py::none());

    module.def(
        "kinetic_center_derivatives_shell",
        [](const giao::Shell& a, const giao::Shell& b, const py::object& field,
           py::object output) {
            const auto field_value = field_or_zero(field);
            return derivative_array(std::move(output),
                                    {2, 3, static_cast<py::ssize_t>(a.ao_count()),
                                     static_cast<py::ssize_t>(b.ao_count())},
                                    [&](auto values) {
                                        giao::compute_kinetic_center_derivatives(
                                            a, b, field_value, values);
                                    });
        },
        py::arg("a"), py::arg("b"), py::kw_only(), py::arg("field") = py::none(),
        py::arg("out") = py::none());

    module.def(
        "kinetic_magnetic_derivatives_shell",
        [](const giao::Shell& a, const giao::Shell& b, const py::object& field,
           py::object output) {
            const auto field_value = field_or_zero(field);
            return derivative_array(std::move(output),
                                    {3, static_cast<py::ssize_t>(a.ao_count()),
                                     static_cast<py::ssize_t>(b.ao_count())},
                                    [&](auto values) {
                                        giao::compute_kinetic_magnetic_derivatives(
                                            a, b, field_value, values);
                                    });
        },
        py::arg("a"), py::arg("b"), py::kw_only(), py::arg("field") = py::none(),
        py::arg("out") = py::none());

    module.def(
        "magnetic_kinetic_center_derivatives_shell",
        [](const giao::Shell& a, const giao::Shell& b, const py::object& field,
           py::object output) {
            const auto field_value = field_or_zero(field);
            return derivative_array(
                std::move(output),
                {2, 3, static_cast<py::ssize_t>(a.ao_count()),
                 static_cast<py::ssize_t>(b.ao_count())},
                [&](auto values) {
                    giao::compute_magnetic_kinetic_center_derivatives(a, b, field_value,
                                                                      values);
                });
        },
        py::arg("a"), py::arg("b"), py::kw_only(), py::arg("field") = py::none(),
        py::arg("out") = py::none());

    module.def(
        "magnetic_kinetic_magnetic_derivatives_shell",
        [](const giao::Shell& a, const giao::Shell& b, const py::object& field,
           py::object output) {
            const auto field_value = field_or_zero(field);
            return derivative_array(
                std::move(output),
                {3, static_cast<py::ssize_t>(a.ao_count()),
                 static_cast<py::ssize_t>(b.ao_count())},
                [&](auto values) {
                    giao::compute_magnetic_kinetic_magnetic_derivatives(
                        a, b, field_value, values);
                });
        },
        py::arg("a"), py::arg("b"), py::kw_only(), py::arg("field") = py::none(),
        py::arg("out") = py::none());

    module.def(
        "nuclear_attraction_center_derivatives_shell",
        [](const giao::Shell& a, const giao::Shell& b,
           const std::vector<giao::Nucleus>& nuclei, const py::object& field,
           py::object output) {
            const auto field_value = field_or_zero(field);
            return derivative_array(
                std::move(output),
                {2, 3, static_cast<py::ssize_t>(a.ao_count()),
                 static_cast<py::ssize_t>(b.ao_count())},
                [&](auto values) {
                    giao::compute_nuclear_attraction_center_derivatives(
                        a, b, nuclei, field_value, values);
                });
        },
        py::arg("a"), py::arg("b"), py::arg("nuclei"), py::kw_only(),
        py::arg("field") = py::none(), py::arg("out") = py::none());

    module.def(
        "nuclear_attraction_nucleus_derivatives_shell",
        [](const giao::Shell& a, const giao::Shell& b,
           const std::vector<giao::Nucleus>& nuclei, const py::object& field,
           py::object output) {
            const auto field_value = field_or_zero(field);
            return derivative_array(
                std::move(output),
                {static_cast<py::ssize_t>(nuclei.size()), 3,
                 static_cast<py::ssize_t>(a.ao_count()),
                 static_cast<py::ssize_t>(b.ao_count())},
                [&](auto values) {
                    giao::compute_nuclear_attraction_nucleus_derivatives(
                        a, b, nuclei, field_value, values);
                });
        },
        py::arg("a"), py::arg("b"), py::arg("nuclei"), py::kw_only(),
        py::arg("field") = py::none(), py::arg("out") = py::none());

    module.def(
        "nuclear_attraction_magnetic_derivatives_shell",
        [](const giao::Shell& a, const giao::Shell& b,
           const std::vector<giao::Nucleus>& nuclei, const py::object& field,
           py::object output) {
            const auto field_value = field_or_zero(field);
            return derivative_array(
                std::move(output),
                {3, static_cast<py::ssize_t>(a.ao_count()),
                 static_cast<py::ssize_t>(b.ao_count())},
                [&](auto values) {
                    giao::compute_nuclear_attraction_magnetic_derivatives(
                        a, b, nuclei, field_value, values);
                });
        },
        py::arg("a"), py::arg("b"), py::arg("nuclei"), py::kw_only(),
        py::arg("field") = py::none(), py::arg("out") = py::none());

    module.def(
        "eri_center_derivatives_shell",
        [](const giao::Shell& a, const giao::Shell& b, const giao::Shell& c,
           const giao::Shell& d, const py::object& field, py::object output) {
            const auto field_value = field_or_zero(field);
            return derivative_array(std::move(output),
                                    {4, 3, static_cast<py::ssize_t>(a.ao_count()),
                                     static_cast<py::ssize_t>(b.ao_count()),
                                     static_cast<py::ssize_t>(c.ao_count()),
                                     static_cast<py::ssize_t>(d.ao_count())},
                                    [&](auto values) {
                                        giao::compute_eri_center_derivatives(
                                            a, b, c, d, field_value, values);
                                    });
        },
        py::arg("a"), py::arg("b"), py::arg("c"), py::arg("d"), py::kw_only(),
        py::arg("field") = py::none(), py::arg("out") = py::none());

    module.def(
        "eri_magnetic_derivatives_shell",
        [](const giao::Shell& a, const giao::Shell& b, const giao::Shell& c,
           const giao::Shell& d, const py::object& field, py::object output) {
            const auto field_value = field_or_zero(field);
            return derivative_array(std::move(output),
                                    {3, static_cast<py::ssize_t>(a.ao_count()),
                                     static_cast<py::ssize_t>(b.ao_count()),
                                     static_cast<py::ssize_t>(c.ao_count()),
                                     static_cast<py::ssize_t>(d.ao_count())},
                                    [&](auto values) {
                                        giao::compute_eri_magnetic_derivatives(
                                            a, b, c, d, field_value, values);
                                    });
        },
        py::arg("a"), py::arg("b"), py::arg("c"), py::arg("d"), py::kw_only(),
        py::arg("field") = py::none(), py::arg("out") = py::none());

    module.def(
        "overlap_nuclear_derivatives",
        [](const giao::Basis& basis, const py::object& field, py::object output) {
            const auto field_value = field_or_zero(field);
            const auto count = static_cast<py::ssize_t>(basis.ao_count());
            return derivative_array(
                std::move(output),
                {static_cast<py::ssize_t>(basis.shells().size()), 3, count, count},
                [&](auto values) {
                    giao::compute_overlap_nuclear_derivative_matrices(
                        basis, field_value, values);
                });
        },
        py::arg("basis"), py::kw_only(), py::arg("field") = py::none(),
        py::arg("out") = py::none());

    module.def(
        "overlap_magnetic_derivatives",
        [](const giao::Basis& basis, const py::object& field, py::object output) {
            const auto field_value = field_or_zero(field);
            const auto count = static_cast<py::ssize_t>(basis.ao_count());
            return derivative_array(
                std::move(output), {3, count, count}, [&](auto values) {
                    giao::compute_overlap_magnetic_derivative_matrices(
                        basis, field_value, values);
                });
        },
        py::arg("basis"), py::kw_only(), py::arg("field") = py::none(),
        py::arg("out") = py::none());

    module.def(
        "kinetic_nuclear_derivatives",
        [](const giao::Basis& basis, const py::object& field, py::object output) {
            const auto field_value = field_or_zero(field);
            const auto count = static_cast<py::ssize_t>(basis.ao_count());
            return derivative_array(
                std::move(output),
                {static_cast<py::ssize_t>(basis.shells().size()), 3, count, count},
                [&](auto values) {
                    giao::compute_kinetic_nuclear_derivative_matrices(
                        basis, field_value, values);
                });
        },
        py::arg("basis"), py::kw_only(), py::arg("field") = py::none(),
        py::arg("out") = py::none());

    module.def(
        "kinetic_magnetic_derivatives",
        [](const giao::Basis& basis, const py::object& field, py::object output) {
            const auto field_value = field_or_zero(field);
            const auto count = static_cast<py::ssize_t>(basis.ao_count());
            return derivative_array(
                std::move(output), {3, count, count}, [&](auto values) {
                    giao::compute_kinetic_magnetic_derivative_matrices(
                        basis, field_value, values);
                });
        },
        py::arg("basis"), py::kw_only(), py::arg("field") = py::none(),
        py::arg("out") = py::none());

    module.def(
        "magnetic_kinetic_nuclear_derivatives",
        [](const giao::Basis& basis, const py::object& field, py::object output) {
            const auto field_value = field_or_zero(field);
            const auto count = static_cast<py::ssize_t>(basis.ao_count());
            return derivative_array(
                std::move(output),
                {static_cast<py::ssize_t>(basis.shells().size()), 3, count, count},
                [&](auto values) {
                    giao::compute_magnetic_kinetic_nuclear_derivative_matrices(
                        basis, field_value, values);
                });
        },
        py::arg("basis"), py::kw_only(), py::arg("field") = py::none(),
        py::arg("out") = py::none());

    module.def(
        "magnetic_kinetic_magnetic_derivatives",
        [](const giao::Basis& basis, const py::object& field, py::object output) {
            const auto field_value = field_or_zero(field);
            const auto count = static_cast<py::ssize_t>(basis.ao_count());
            return derivative_array(
                std::move(output), {3, count, count}, [&](auto values) {
                    giao::compute_magnetic_kinetic_magnetic_derivative_matrices(
                        basis, field_value, values);
                });
        },
        py::arg("basis"), py::kw_only(), py::arg("field") = py::none(),
        py::arg("out") = py::none());

    module.def(
        "nuclear_attraction_nuclear_derivatives",
        [](const giao::Basis& basis, const std::vector<giao::Nucleus>& nuclei,
           const py::object& field, py::object shell_output,
           py::object nucleus_output) {
            const auto field_value = field_or_zero(field);
            const auto count = static_cast<py::ssize_t>(basis.ao_count());
            py::array shell_result = validate_or_create_output(
                std::move(shell_output),
                {static_cast<py::ssize_t>(basis.shells().size()), 3, count, count});
            py::array nucleus_result = validate_or_create_output(
                std::move(nucleus_output),
                {static_cast<py::ssize_t>(nuclei.size()), 3, count, count});
            {
                py::gil_scoped_release release;
                giao::compute_nuclear_attraction_nuclear_derivative_matrices(
                    basis, nuclei, field_value,
                    std::span<giao::Complex>(
                        static_cast<giao::Complex*>(shell_result.mutable_data()),
                        shell_result.size()),
                    std::span<giao::Complex>(
                        static_cast<giao::Complex*>(nucleus_result.mutable_data()),
                        nucleus_result.size()));
            }
            return py::make_tuple(std::move(shell_result), std::move(nucleus_result));
        },
        py::arg("basis"), py::arg("nuclei"), py::kw_only(),
        py::arg("field") = py::none(), py::arg("shell_out") = py::none(),
        py::arg("nucleus_out") = py::none());

    module.def(
        "nuclear_attraction_magnetic_derivatives",
        [](const giao::Basis& basis, const std::vector<giao::Nucleus>& nuclei,
           const py::object& field, py::object output) {
            const auto field_value = field_or_zero(field);
            const auto count = static_cast<py::ssize_t>(basis.ao_count());
            return derivative_array(
                std::move(output), {3, count, count}, [&](auto values) {
                    giao::compute_nuclear_attraction_magnetic_derivative_matrices(
                        basis, nuclei, field_value, values);
                });
        },
        py::arg("basis"), py::arg("nuclei"), py::kw_only(),
        py::arg("field") = py::none(), py::arg("out") = py::none());
    module.def("primitive_moment_center_derivatives",
        [](const giao::PrimitiveGaussian& a, const giao::PrimitiveGaussian& b,
           const py::object& powers, const py::object& origin, const py::object& field) {
            const auto moment = moment_from_python(powers, origin);
            const auto f = field_or_zero(field);
            std::array<giao::Complex, 6> values{};
            { py::gil_scoped_release release;
              values = giao::primitive_moment_center_derivatives(a, b, moment, f); }
            return derivative_values(values, {2, 3});
        }, py::arg("a"), py::arg("b"), py::arg("powers"), py::kw_only(),
        py::arg("origin") = py::make_tuple(0.0, 0.0, 0.0), py::arg("field") = py::none(),
        "Literal centre derivatives, shape (2,3), including London-phase response.");
    module.def("primitive_moment_origin_derivatives",
        [](const giao::PrimitiveGaussian& a, const giao::PrimitiveGaussian& b,
           const py::object& powers, const py::object& origin, const py::object& field) {
            const auto moment = moment_from_python(powers, origin);
            const auto f = field_or_zero(field);
            std::array<giao::Complex, 3> values{};
            { py::gil_scoped_release release;
              values = giao::primitive_moment_origin_derivatives(a, b, moment, f); }
            return derivative_values(values, {3});
        }, py::arg("a"), py::arg("b"), py::arg("powers"), py::kw_only(),
        py::arg("origin") = py::make_tuple(0.0, 0.0, 0.0), py::arg("field") = py::none(),
        "Moment-origin derivatives: -p_axis times the corresponding lower moment.");
    module.def("moment_center_derivatives_shell",
        [](const giao::Shell& a, const giao::Shell& b, const py::object& powers,
           const py::object& origin, const py::object& field, py::object output) {
            const auto moment = moment_from_python(powers, origin);
            const auto f = field_or_zero(field);
            return derivative_array(std::move(output),
                {2, 3, static_cast<py::ssize_t>(a.ao_count()), static_cast<py::ssize_t>(b.ao_count())},
                [&](auto values) { giao::compute_moment_center_derivatives(a, b, moment, f, values); });
        }, py::arg("a"), py::arg("b"), py::arg("powers"), py::kw_only(),
        py::arg("origin") = py::make_tuple(0.0, 0.0, 0.0), py::arg("field") = py::none(),
        py::arg("out") = py::none(), "Literal derivatives, shape (2,3,na,nb), at fixed moment/gauge origins.");
    module.def("moment_origin_derivatives_shell",
        [](const giao::Shell& a, const giao::Shell& b, const py::object& powers,
           const py::object& origin, const py::object& field, py::object output) {
            const auto moment = moment_from_python(powers, origin);
            const auto f = field_or_zero(field);
            return derivative_array(std::move(output),
                {3, static_cast<py::ssize_t>(a.ao_count()), static_cast<py::ssize_t>(b.ao_count())},
                [&](auto values) { giao::compute_moment_origin_derivatives(a, b, moment, f, values); });
        }, py::arg("a"), py::arg("b"), py::arg("powers"), py::kw_only(),
        py::arg("origin") = py::make_tuple(0.0, 0.0, 0.0), py::arg("field") = py::none(),
        py::arg("out") = py::none(), "Moment-origin derivatives, shape (3,na,nb).");
    module.def("moment_nuclear_derivatives",
        [](const giao::Basis& basis, const py::object& powers, const py::object& origin,
           const py::object& field, const py::object& output, const py::object& threads) {
            const auto moment = moment_from_python(powers, origin);
            const auto f = field_or_zero(field);
            const auto thread_count = threads_from_python(threads);
            py::object shell_out = py::none();
            py::object origin_out = py::none();
            if (!output.is_none()) {
                if (!py::isinstance<py::tuple>(output) || py::len(output) != 2) {
                    throw py::value_error("out must be None or a (shell_out, origin_out) tuple");
                }
                shell_out = output[py::int_(0)];
                origin_out = output[py::int_(1)];
            }
            const auto n = static_cast<py::ssize_t>(basis.ao_count());
            auto centers = validate_or_create_output(shell_out,
                {static_cast<py::ssize_t>(basis.shells().size()), 3, n, n});
            auto origins = validate_or_create_output(origin_out, {3, n, n});
            const auto first = reinterpret_cast<std::uintptr_t>(centers.data());
            const auto second = reinterpret_cast<std::uintptr_t>(origins.data());
            if (centers.nbytes() != 0 && origins.nbytes() != 0 &&
                first < second + static_cast<std::uintptr_t>(origins.nbytes()) &&
                second < first + static_cast<std::uintptr_t>(centers.nbytes())) {
                throw py::value_error("shell and origin outputs must not overlap");
            }
            { py::gil_scoped_release release;
              giao::compute_moment_nuclear_derivative_matrices(basis, moment, f,
                  {static_cast<giao::Complex*>(centers.mutable_data()), static_cast<std::size_t>(centers.size())},
                  {static_cast<giao::Complex*>(origins.mutable_data()), static_cast<std::size_t>(origins.size())}, thread_count); }
            return py::make_tuple(centers, origins);
        }, py::arg("basis"), py::arg("powers"), py::kw_only(),
        py::arg("origin") = py::make_tuple(0.0, 0.0, 0.0), py::arg("field") = py::none(),
        py::arg("out") = py::none(), py::arg("threads") = py::none(),
        R"doc(Return (basis-centre, moment-origin) derivatives of <mu|(r-origin)^powers|nu>.

Shapes are (nshell,3,nao,nao) and (3,nao,nao), complex128, in existing AO order.
Derivatives are literally dI/dR and dI/dorigin, with no force/electronic-charge sign.
Includes ordinary Gaussian and London-phase centre response, at fixed B and
fixed gauge origin. General non-negative Cartesian powers are supported within
the value kernel's angular limits. out=(shell_array, origin_array) fills and
returns those exact arrays (either may be None); buffers must not overlap.
threads=None uses the configured library policy. Native work releases the GIL.
)doc");

    module.def("primitive_gradient_center_derivatives",
        [](const giao::PrimitiveGaussian& a, const giao::PrimitiveGaussian& b,
           const std::string& component, const py::object& field) {
            const auto property = axis_from_python(component);
            const auto f = field_or_zero(field);
            std::array<giao::Complex, 6> values{};
            { py::gil_scoped_release release;
              values = giao::primitive_gradient_center_derivatives(a, b, property, f); }
            return derivative_values(values, {2, 3});
        }, py::arg("a"), py::arg("b"), py::arg("component"), py::kw_only(),
        py::arg("field") = py::none(), R"doc(Literal basis-centre derivatives of the gradient integral.

a, b: Cartesian primitive Gaussians.
component: x, y or z selects the ket spatial gradient.
field: phase field B and its gauge origin; None means zero field.

Returns complex128 shape (2, 3); derivative axes are xyz.
B and the magnetic gauge origin are held fixed.
Includes ordinary Gaussian and complete London-phase centre response,
including its action under the ket spatial derivative.
The operator has no explicit B dependence. No force or charge sign is applied.
Native computation releases the GIL.
)doc");

    module.def("gradient_center_derivatives_shell",
        [](const giao::Shell& a, const giao::Shell& b, const std::string& component,
           const py::object& field, py::object output) {
            const auto property = axis_from_python(component);
            const auto f = field_or_zero(field);
            return derivative_array(std::move(output),
                {2, 3, static_cast<py::ssize_t>(a.ao_count()), static_cast<py::ssize_t>(b.ao_count())},
                [&](auto values) { giao::compute_gradient_center_derivatives(a, b, property, f, values); });
        }, py::arg("a"), py::arg("b"), py::arg("component"), py::kw_only(),
        py::arg("field") = py::none(), py::arg("out") = py::none(),
        R"doc(Literal basis-centre derivatives of the gradient integral.

a, b: Cartesian shells, with na and nb AOs.
component: x, y or z selects the ket spatial gradient.
field: phase field B and its gauge origin; None means zero field.

Returns complex128 shape (2, 3, na, nb); derivative axes are xyz.
B and the magnetic gauge origin are held fixed.
Includes ordinary Gaussian and complete London-phase centre response,
including its action under the ket spatial derivative.
The operator has no explicit B dependence. No force or charge sign is applied.
out: optional writable C-contiguous complex128 buffer of the exact shape;
the exact buffer is filled and returned.
Native computation releases the GIL.
)doc");

    module.def("gradient_nuclear_derivatives",
        [](const giao::Basis& basis, const std::string& component, const py::object& field,
           py::object output, const py::object& threads) {
            const auto property = axis_from_python(component);
            const auto f = field_or_zero(field);
            const auto thread_count = threads_from_python(threads);
            const auto n = static_cast<py::ssize_t>(basis.ao_count());
            return derivative_array(std::move(output), {static_cast<py::ssize_t>(basis.shells().size()), 3, n, n},
                [&](auto values) { giao::compute_gradient_nuclear_derivative_matrices(basis, property, f, values, thread_count); });
        }, py::arg("basis"), py::arg("component"), py::kw_only(), py::arg("field") = py::none(),
        py::arg("out") = py::none(), py::arg("threads") = py::none(),
        R"doc(Literal basis-centre derivatives of the gradient integral.

basis: Cartesian AO basis, with nshell independent shell centres.
component: x, y or z selects the ket spatial gradient.
field: phase field B and its gauge origin; None means zero field.

Returns complex128 shape (nshell, 3, nao, nao); derivative axes are xyz.
B and the magnetic gauge origin are held fixed.
Includes ordinary Gaussian and complete London-phase centre response,
including its action under the ket spatial derivative.
The operator has no explicit B dependence. No force or charge sign is applied.
out: optional writable C-contiguous complex128 buffer of the exact shape;
the exact buffer is filled and returned.
threads: None uses the configured library policy; 1 is serial.
Native computation releases the GIL.
)doc");

    module.def("primitive_gradient_magnetic_derivatives",
        [](const giao::PrimitiveGaussian& a, const giao::PrimitiveGaussian& b,
           const std::string& component, const py::object& field) {
            const auto property = axis_from_python(component);
            const auto f = field_or_zero(field);
            std::array<giao::Complex, 3> values{};
            { py::gil_scoped_release release;
              values = giao::primitive_gradient_magnetic_derivatives(a, b, property, f); }
            return derivative_values(values, {3});
        }, py::arg("a"), py::arg("b"), py::arg("component"), py::kw_only(),
        py::arg("field") = py::none(), R"doc(Literal magnetic-field derivatives of the gradient integral.

a, b: Cartesian primitive Gaussians.
component: x, y or z selects the ket spatial gradient.
field: phase field B and its gauge origin; None means zero field.

Returns complex128 shape (3,); derivative axes are xyz.
Basis centres and the magnetic gauge origin are held fixed.
Includes the complete London-phase field response,
including the derivative of the ket wave vector under the spatial derivative.
The operator has no explicit B dependence. No force or charge sign is applied.
Native computation releases the GIL.
)doc");

    module.def("gradient_magnetic_derivatives_shell",
        [](const giao::Shell& a, const giao::Shell& b, const std::string& component,
           const py::object& field, py::object output) {
            const auto property = axis_from_python(component);
            const auto f = field_or_zero(field);
            return derivative_array(std::move(output),
                {3, static_cast<py::ssize_t>(a.ao_count()), static_cast<py::ssize_t>(b.ao_count())},
                [&](auto values) { giao::compute_gradient_magnetic_derivatives(a, b, property, f, values); });
        }, py::arg("a"), py::arg("b"), py::arg("component"), py::kw_only(),
        py::arg("field") = py::none(), py::arg("out") = py::none(),
        R"doc(Literal magnetic-field derivatives of the gradient integral.

a, b: Cartesian shells, with na and nb AOs.
component: x, y or z selects the ket spatial gradient.
field: phase field B and its gauge origin; None means zero field.

Returns complex128 shape (3, na, nb); derivative axes are xyz.
Basis centres and the magnetic gauge origin are held fixed.
Includes the complete London-phase field response,
including the derivative of the ket wave vector under the spatial derivative.
The operator has no explicit B dependence. No force or charge sign is applied.
out: optional writable C-contiguous complex128 buffer of the exact shape;
the exact buffer is filled and returned.
Native computation releases the GIL.
)doc");

    module.def("gradient_magnetic_derivatives",
        [](const giao::Basis& basis, const std::string& component, const py::object& field,
           py::object output, const py::object& threads) {
            const auto property = axis_from_python(component);
            const auto f = field_or_zero(field);
            const auto thread_count = threads_from_python(threads);
            const auto n = static_cast<py::ssize_t>(basis.ao_count());
            return derivative_array(std::move(output), {3, n, n},
                [&](auto values) { giao::compute_gradient_magnetic_derivative_matrices(basis, property, f, values, thread_count); });
        }, py::arg("basis"), py::arg("component"), py::kw_only(), py::arg("field") = py::none(),
        py::arg("out") = py::none(), py::arg("threads") = py::none(),
        R"doc(Literal magnetic-field derivatives of the gradient integral.

basis: Cartesian AO basis, with nshell independent shell centres.
component: x, y or z selects the ket spatial gradient.
field: phase field B and its gauge origin; None means zero field.

Returns complex128 shape (3, nao, nao); derivative axes are xyz.
Basis centres and the magnetic gauge origin are held fixed.
Includes the complete London-phase field response,
including the derivative of the ket wave vector under the spatial derivative.
The operator has no explicit B dependence. No force or charge sign is applied.
out: optional writable C-contiguous complex128 buffer of the exact shape;
the exact buffer is filled and returned.
threads: None uses the configured library policy; 1 is serial.
Native computation releases the GIL.
)doc");

    module.def("primitive_momentum_center_derivatives",
        [](const giao::PrimitiveGaussian& a, const giao::PrimitiveGaussian& b,
           const std::string& component, const py::object& field) {
            const auto property = axis_from_python(component);
            const auto f = field_or_zero(field);
            std::array<giao::Complex, 6> values{};
            { py::gil_scoped_release release;
              values = giao::primitive_momentum_center_derivatives(a, b, property, f); }
            return derivative_values(values, {2, 3});
        }, py::arg("a"), py::arg("b"), py::arg("component"), py::kw_only(),
        py::arg("field") = py::none(), R"doc(Literal basis-centre derivatives of the momentum integral.

a, b: Cartesian primitive Gaussians.
component: x, y or z selects the ket spatial canonical momentum -i*gradient (without vector potential).
field: phase field B and its gauge origin; None means zero field.

Returns complex128 shape (2, 3); derivative axes are xyz.
B and the magnetic gauge origin are held fixed.
Includes ordinary Gaussian and complete London-phase centre response,
including its action under the ket spatial derivative.
The operator has no explicit B dependence. No force or charge sign is applied.
Native computation releases the GIL.
)doc");

    module.def("momentum_center_derivatives_shell",
        [](const giao::Shell& a, const giao::Shell& b, const std::string& component,
           const py::object& field, py::object output) {
            const auto property = axis_from_python(component);
            const auto f = field_or_zero(field);
            return derivative_array(std::move(output),
                {2, 3, static_cast<py::ssize_t>(a.ao_count()), static_cast<py::ssize_t>(b.ao_count())},
                [&](auto values) { giao::compute_momentum_center_derivatives(a, b, property, f, values); });
        }, py::arg("a"), py::arg("b"), py::arg("component"), py::kw_only(),
        py::arg("field") = py::none(), py::arg("out") = py::none(),
        R"doc(Literal basis-centre derivatives of the momentum integral.

a, b: Cartesian shells, with na and nb AOs.
component: x, y or z selects the ket spatial canonical momentum -i*gradient (without vector potential).
field: phase field B and its gauge origin; None means zero field.

Returns complex128 shape (2, 3, na, nb); derivative axes are xyz.
B and the magnetic gauge origin are held fixed.
Includes ordinary Gaussian and complete London-phase centre response,
including its action under the ket spatial derivative.
The operator has no explicit B dependence. No force or charge sign is applied.
out: optional writable C-contiguous complex128 buffer of the exact shape;
the exact buffer is filled and returned.
Native computation releases the GIL.
)doc");

    module.def("momentum_nuclear_derivatives",
        [](const giao::Basis& basis, const std::string& component, const py::object& field,
           py::object output, const py::object& threads) {
            const auto property = axis_from_python(component);
            const auto f = field_or_zero(field);
            const auto thread_count = threads_from_python(threads);
            const auto n = static_cast<py::ssize_t>(basis.ao_count());
            return derivative_array(std::move(output), {static_cast<py::ssize_t>(basis.shells().size()), 3, n, n},
                [&](auto values) { giao::compute_momentum_nuclear_derivative_matrices(basis, property, f, values, thread_count); });
        }, py::arg("basis"), py::arg("component"), py::kw_only(), py::arg("field") = py::none(),
        py::arg("out") = py::none(), py::arg("threads") = py::none(),
        R"doc(Literal basis-centre derivatives of the momentum integral.

basis: Cartesian AO basis, with nshell independent shell centres.
component: x, y or z selects the ket spatial canonical momentum -i*gradient (without vector potential).
field: phase field B and its gauge origin; None means zero field.

Returns complex128 shape (nshell, 3, nao, nao); derivative axes are xyz.
B and the magnetic gauge origin are held fixed.
Includes ordinary Gaussian and complete London-phase centre response,
including its action under the ket spatial derivative.
The operator has no explicit B dependence. No force or charge sign is applied.
out: optional writable C-contiguous complex128 buffer of the exact shape;
the exact buffer is filled and returned.
threads: None uses the configured library policy; 1 is serial.
Native computation releases the GIL.
)doc");

    module.def("primitive_momentum_magnetic_derivatives",
        [](const giao::PrimitiveGaussian& a, const giao::PrimitiveGaussian& b,
           const std::string& component, const py::object& field) {
            const auto property = axis_from_python(component);
            const auto f = field_or_zero(field);
            std::array<giao::Complex, 3> values{};
            { py::gil_scoped_release release;
              values = giao::primitive_momentum_magnetic_derivatives(a, b, property, f); }
            return derivative_values(values, {3});
        }, py::arg("a"), py::arg("b"), py::arg("component"), py::kw_only(),
        py::arg("field") = py::none(), R"doc(Literal magnetic-field derivatives of the momentum integral.

a, b: Cartesian primitive Gaussians.
component: x, y or z selects the ket spatial canonical momentum -i*gradient (without vector potential).
field: phase field B and its gauge origin; None means zero field.

Returns complex128 shape (3,); derivative axes are xyz.
Basis centres and the magnetic gauge origin are held fixed.
Includes the complete London-phase field response,
including the derivative of the ket wave vector under the spatial derivative.
The operator has no explicit B dependence. No force or charge sign is applied.
Native computation releases the GIL.
)doc");

    module.def("momentum_magnetic_derivatives_shell",
        [](const giao::Shell& a, const giao::Shell& b, const std::string& component,
           const py::object& field, py::object output) {
            const auto property = axis_from_python(component);
            const auto f = field_or_zero(field);
            return derivative_array(std::move(output),
                {3, static_cast<py::ssize_t>(a.ao_count()), static_cast<py::ssize_t>(b.ao_count())},
                [&](auto values) { giao::compute_momentum_magnetic_derivatives(a, b, property, f, values); });
        }, py::arg("a"), py::arg("b"), py::arg("component"), py::kw_only(),
        py::arg("field") = py::none(), py::arg("out") = py::none(),
        R"doc(Literal magnetic-field derivatives of the momentum integral.

a, b: Cartesian shells, with na and nb AOs.
component: x, y or z selects the ket spatial canonical momentum -i*gradient (without vector potential).
field: phase field B and its gauge origin; None means zero field.

Returns complex128 shape (3, na, nb); derivative axes are xyz.
Basis centres and the magnetic gauge origin are held fixed.
Includes the complete London-phase field response,
including the derivative of the ket wave vector under the spatial derivative.
The operator has no explicit B dependence. No force or charge sign is applied.
out: optional writable C-contiguous complex128 buffer of the exact shape;
the exact buffer is filled and returned.
Native computation releases the GIL.
)doc");

    module.def("momentum_magnetic_derivatives",
        [](const giao::Basis& basis, const std::string& component, const py::object& field,
           py::object output, const py::object& threads) {
            const auto property = axis_from_python(component);
            const auto f = field_or_zero(field);
            const auto thread_count = threads_from_python(threads);
            const auto n = static_cast<py::ssize_t>(basis.ao_count());
            return derivative_array(std::move(output), {3, n, n},
                [&](auto values) { giao::compute_momentum_magnetic_derivative_matrices(basis, property, f, values, thread_count); });
        }, py::arg("basis"), py::arg("component"), py::kw_only(), py::arg("field") = py::none(),
        py::arg("out") = py::none(), py::arg("threads") = py::none(),
        R"doc(Literal magnetic-field derivatives of the momentum integral.

basis: Cartesian AO basis, with nshell independent shell centres.
component: x, y or z selects the ket spatial canonical momentum -i*gradient (without vector potential).
field: phase field B and its gauge origin; None means zero field.

Returns complex128 shape (3, nao, nao); derivative axes are xyz.
Basis centres and the magnetic gauge origin are held fixed.
Includes the complete London-phase field response,
including the derivative of the ket wave vector under the spatial derivative.
The operator has no explicit B dependence. No force or charge sign is applied.
out: optional writable C-contiguous complex128 buffer of the exact shape;
the exact buffer is filled and returned.
threads: None uses the configured library policy; 1 is serial.
Native computation releases the GIL.
)doc");

    module.def("primitive_moment_magnetic_derivatives",
        [](const giao::PrimitiveGaussian& a, const giao::PrimitiveGaussian& b,
           const py::object& powers, const py::object& origin, const py::object& field) {
            const auto property = moment_from_python(powers, origin);
            const auto f = field_or_zero(field);
            std::array<giao::Complex, 3> values{};
            { py::gil_scoped_release release;
              values = giao::primitive_moment_magnetic_derivatives(a, b, property, f); }
            return derivative_values(values, {3});
        }, py::arg("a"), py::arg("b"), py::arg("powers"), py::kw_only(), py::arg("origin") = py::make_tuple(0.0, 0.0, 0.0),
        py::arg("field") = py::none(), R"doc(Literal magnetic-field derivatives of the moment integral.

a, b: Cartesian primitive Gaussians.
powers: three non-negative Cartesian powers of (r-origin).
field: phase field B and its gauge origin; None means zero field.
origin: moment origin in bohr, default (0,0,0).

Returns complex128 shape (3,); derivative axes are xyz.
Basis centres and the magnetic gauge origin are held fixed. The moment origin is also held fixed.
Includes the complete London-phase field response.
The operator has no explicit B dependence. No force or charge sign is applied.
Native computation releases the GIL.
)doc");

    module.def("moment_magnetic_derivatives_shell",
        [](const giao::Shell& a, const giao::Shell& b, const py::object& powers, const py::object& origin,
           const py::object& field, py::object output) {
            const auto property = moment_from_python(powers, origin);
            const auto f = field_or_zero(field);
            return derivative_array(std::move(output),
                {3, static_cast<py::ssize_t>(a.ao_count()), static_cast<py::ssize_t>(b.ao_count())},
                [&](auto values) { giao::compute_moment_magnetic_derivatives(a, b, property, f, values); });
        }, py::arg("a"), py::arg("b"), py::arg("powers"), py::kw_only(), py::arg("origin") = py::make_tuple(0.0, 0.0, 0.0),
        py::arg("field") = py::none(), py::arg("out") = py::none(),
        R"doc(Literal magnetic-field derivatives of the moment integral.

a, b: Cartesian shells, with na and nb AOs.
powers: three non-negative Cartesian powers of (r-origin).
field: phase field B and its gauge origin; None means zero field.
origin: moment origin in bohr, default (0,0,0).

Returns complex128 shape (3, na, nb); derivative axes are xyz.
Basis centres and the magnetic gauge origin are held fixed. The moment origin is also held fixed.
Includes the complete London-phase field response.
The operator has no explicit B dependence. No force or charge sign is applied.
out: optional writable C-contiguous complex128 buffer of the exact shape;
the exact buffer is filled and returned.
Native computation releases the GIL.
)doc");

    module.def("moment_magnetic_derivatives",
        [](const giao::Basis& basis, const py::object& powers, const py::object& origin, const py::object& field,
           py::object output, const py::object& threads) {
            const auto property = moment_from_python(powers, origin);
            const auto f = field_or_zero(field);
            const auto thread_count = threads_from_python(threads);
            const auto n = static_cast<py::ssize_t>(basis.ao_count());
            return derivative_array(std::move(output), {3, n, n},
                [&](auto values) { giao::compute_moment_magnetic_derivative_matrices(basis, property, f, values, thread_count); });
        }, py::arg("basis"), py::arg("powers"), py::kw_only(), py::arg("origin") = py::make_tuple(0.0, 0.0, 0.0), py::arg("field") = py::none(),
        py::arg("out") = py::none(), py::arg("threads") = py::none(),
        R"doc(Literal magnetic-field derivatives of the moment integral.

basis: Cartesian AO basis, with nshell independent shell centres.
powers: three non-negative Cartesian powers of (r-origin).
field: phase field B and its gauge origin; None means zero field.
origin: moment origin in bohr, default (0,0,0).

Returns complex128 shape (3, nao, nao); derivative axes are xyz.
Basis centres and the magnetic gauge origin are held fixed. The moment origin is also held fixed.
Includes the complete London-phase field response.
The operator has no explicit B dependence. No force or charge sign is applied.
out: optional writable C-contiguous complex128 buffer of the exact shape;
the exact buffer is filled and returned.
threads: None uses the configured library policy; 1 is serial.
Native computation releases the GIL.
)doc");

    module.def("primitive_overlap_mixed_derivatives", [](const giao::PrimitiveGaussian& a, const giao::PrimitiveGaussian& b,const py::object& field) {
        const auto f=field_or_zero(field);
        std::array<giao::Complex,18> values{};
        { py::gil_scoped_release release;
          values=giao::primitive_overlap_mixed_derivatives(a,b,f); }
        return derivative_values(values,{2,3,3});
    }, py::arg("a"), py::arg("b"), py::kw_only(),py::arg("field")=py::none(),
    R"doc(Literal mixed d2I/dR dB for overlap.

Returns complex128 shape (2, 3, 3); axes are centre slot, nuclear xyz,
field xyz, then AO axes where present. The magnetic gauge origin is fixed.
Includes Gaussian and full London centre/field/cross response.
The operator has no explicit B dependence.
No force sign. Native computation releases the GIL.
)doc");

    module.def("overlap_mixed_derivatives_shell", [](const giao::Shell& a, const giao::Shell& b,const py::object& field,py::object output) {
        const auto f=field_or_zero(field);
        return derivative_array(std::move(output),{2,3,3,static_cast<py::ssize_t>(a.ao_count()), static_cast<py::ssize_t>(b.ao_count())},
            [&](auto values) { giao::compute_overlap_mixed_derivatives(a,b,f,values); });
    }, py::arg("a"), py::arg("b"), py::kw_only(),py::arg("field")=py::none(),py::arg("out")=py::none(),
    R"doc(Literal mixed d2I/dR dB for overlap.

Returns complex128 shape (2, 3, 3, na, nb); axes are centre slot, nuclear xyz,
field xyz, then AO axes where present. The magnetic gauge origin is fixed.
Includes Gaussian and full London centre/field/cross response.
The operator has no explicit B dependence.
No force sign. Native computation releases the GIL.
out: optional strict writable C-contiguous complex128 buffer, returned exactly.
)doc");

    module.def("overlap_mixed_derivatives", [](const giao::Basis& basis,const py::object& field,py::object output,const py::object& threads) {
        const auto f=field_or_zero(field);
        const auto thread_count=threads_from_python(threads);
        const auto n=static_cast<py::ssize_t>(basis.ao_count());
        return derivative_array(std::move(output),{static_cast<py::ssize_t>(basis.shells().size()),3,3,n,n},
            [&](auto values) { giao::compute_overlap_mixed_derivative_matrices(basis,f,values,thread_count); });
    },py::arg("basis"),py::kw_only(),py::arg("field")=py::none(),py::arg("out")=py::none(),py::arg("threads")=py::none(),
    R"doc(Literal mixed d2I/dR dB for overlap.

Returns complex128 shape (nshell, 3, 3, nao, nao); axes are centre slot, nuclear xyz,
field xyz, then AO axes where present. The magnetic gauge origin is fixed.
Includes Gaussian and full London centre/field/cross response.
The operator has no explicit B dependence.
No force sign. Native computation releases the GIL.
out: optional strict writable C-contiguous complex128 buffer, returned exactly.
basis: Cartesian AO basis. field: phase field (None means zero).
threads=None uses configured shell-pair threading.
)doc");

    module.def("primitive_kinetic_mixed_derivatives", [](const giao::PrimitiveGaussian& a, const giao::PrimitiveGaussian& b,const py::object& field) {
        const auto f=field_or_zero(field);
        std::array<giao::Complex,18> values{};
        { py::gil_scoped_release release;
          values=giao::primitive_kinetic_mixed_derivatives(a,b,f); }
        return derivative_values(values,{2,3,3});
    }, py::arg("a"), py::arg("b"), py::kw_only(),py::arg("field")=py::none(),
    R"doc(Literal mixed d2I/dR dB for kinetic.

Returns complex128 shape (2, 3, 3); axes are centre slot, nuclear xyz,
field xyz, then AO axes where present. The magnetic gauge origin is fixed.
Includes Gaussian and full London centre/field/cross response.
The operator has no explicit B dependence.
No force sign. Native computation releases the GIL.
)doc");

    module.def("kinetic_mixed_derivatives_shell", [](const giao::Shell& a, const giao::Shell& b,const py::object& field,py::object output) {
        const auto f=field_or_zero(field);
        return derivative_array(std::move(output),{2,3,3,static_cast<py::ssize_t>(a.ao_count()), static_cast<py::ssize_t>(b.ao_count())},
            [&](auto values) { giao::compute_kinetic_mixed_derivatives(a,b,f,values); });
    }, py::arg("a"), py::arg("b"), py::kw_only(),py::arg("field")=py::none(),py::arg("out")=py::none(),
    R"doc(Literal mixed d2I/dR dB for kinetic.

Returns complex128 shape (2, 3, 3, na, nb); axes are centre slot, nuclear xyz,
field xyz, then AO axes where present. The magnetic gauge origin is fixed.
Includes Gaussian and full London centre/field/cross response.
The operator has no explicit B dependence.
No force sign. Native computation releases the GIL.
out: optional strict writable C-contiguous complex128 buffer, returned exactly.
)doc");

    module.def("kinetic_mixed_derivatives", [](const giao::Basis& basis,const py::object& field,py::object output,const py::object& threads) {
        const auto f=field_or_zero(field);
        const auto thread_count=threads_from_python(threads);
        const auto n=static_cast<py::ssize_t>(basis.ao_count());
        return derivative_array(std::move(output),{static_cast<py::ssize_t>(basis.shells().size()),3,3,n,n},
            [&](auto values) { giao::compute_kinetic_mixed_derivative_matrices(basis,f,values,thread_count); });
    },py::arg("basis"),py::kw_only(),py::arg("field")=py::none(),py::arg("out")=py::none(),py::arg("threads")=py::none(),
    R"doc(Literal mixed d2I/dR dB for kinetic.

Returns complex128 shape (nshell, 3, 3, nao, nao); axes are centre slot, nuclear xyz,
field xyz, then AO axes where present. The magnetic gauge origin is fixed.
Includes Gaussian and full London centre/field/cross response.
The operator has no explicit B dependence.
No force sign. Native computation releases the GIL.
out: optional strict writable C-contiguous complex128 buffer, returned exactly.
basis: Cartesian AO basis. field: phase field (None means zero).
threads=None uses configured shell-pair threading.
)doc");

    module.def("primitive_magnetic_kinetic_mixed_derivatives", [](const giao::PrimitiveGaussian& a, const giao::PrimitiveGaussian& b,const py::object& field) {
        const auto f=field_or_zero(field);
        std::array<giao::Complex,18> values{};
        { py::gil_scoped_release release;
          values=giao::primitive_magnetic_kinetic_mixed_derivatives(a,b,f); }
        return derivative_values(values,{2,3,3});
    }, py::arg("a"), py::arg("b"), py::kw_only(),py::arg("field")=py::none(),
    R"doc(Literal mixed d2I/dR dB for magnetic_kinetic.

Returns complex128 shape (2, 3, 3); axes are centre slot, nuclear xyz,
field xyz, then AO axes where present. The magnetic gauge origin is fixed.
Includes Gaussian and full London centre/field/cross response. Includes the
explicit B dependence of the physical operator 1/2*(p+A)^2.
No force sign. Native computation releases the GIL.
)doc");

    module.def("magnetic_kinetic_mixed_derivatives_shell", [](const giao::Shell& a, const giao::Shell& b,const py::object& field,py::object output) {
        const auto f=field_or_zero(field);
        return derivative_array(std::move(output),{2,3,3,static_cast<py::ssize_t>(a.ao_count()), static_cast<py::ssize_t>(b.ao_count())},
            [&](auto values) { giao::compute_magnetic_kinetic_mixed_derivatives(a,b,f,values); });
    }, py::arg("a"), py::arg("b"), py::kw_only(),py::arg("field")=py::none(),py::arg("out")=py::none(),
    R"doc(Literal mixed d2I/dR dB for magnetic_kinetic.

Returns complex128 shape (2, 3, 3, na, nb); axes are centre slot, nuclear xyz,
field xyz, then AO axes where present. The magnetic gauge origin is fixed.
Includes Gaussian and full London centre/field/cross response. Includes the
explicit B dependence of the physical operator 1/2*(p+A)^2.
No force sign. Native computation releases the GIL.
out: optional strict writable C-contiguous complex128 buffer, returned exactly.
)doc");

    module.def("magnetic_kinetic_mixed_derivatives", [](const giao::Basis& basis,const py::object& field,py::object output,const py::object& threads) {
        const auto f=field_or_zero(field);
        const auto thread_count=threads_from_python(threads);
        const auto n=static_cast<py::ssize_t>(basis.ao_count());
        return derivative_array(std::move(output),{static_cast<py::ssize_t>(basis.shells().size()),3,3,n,n},
            [&](auto values) { giao::compute_magnetic_kinetic_mixed_derivative_matrices(basis,f,values,thread_count); });
    },py::arg("basis"),py::kw_only(),py::arg("field")=py::none(),py::arg("out")=py::none(),py::arg("threads")=py::none(),
    R"doc(Literal mixed d2I/dR dB for magnetic_kinetic.

Returns complex128 shape (nshell, 3, 3, nao, nao); axes are centre slot, nuclear xyz,
field xyz, then AO axes where present. The magnetic gauge origin is fixed.
Includes Gaussian and full London centre/field/cross response. Includes the
explicit B dependence of the physical operator 1/2*(p+A)^2.
No force sign. Native computation releases the GIL.
out: optional strict writable C-contiguous complex128 buffer, returned exactly.
basis: Cartesian AO basis. field: phase field (None means zero).
threads=None uses configured shell-pair threading.
)doc");

    module.def("primitive_nuclear_attraction_mixed_derivatives", [](const giao::PrimitiveGaussian& a, const giao::PrimitiveGaussian& b, const std::vector<giao::Nucleus>& nuclei,const py::object& field) {
        const auto f=field_or_zero(field);
        std::array<giao::Complex,18> values{};
        { py::gil_scoped_release release;
          values=giao::primitive_nuclear_attraction_mixed_derivatives(a,b,nuclei,f); }
        return derivative_values(values,{2,3,3});
    }, py::arg("a"), py::arg("b"), py::arg("nuclei"), py::kw_only(),py::arg("field")=py::none(),
    R"doc(Literal mixed d2I/dR dB for nuclear_attraction.

Returns complex128 shape (2, 3, 3); axes are centre slot, nuclear xyz,
field xyz, then AO axes where present. The magnetic gauge origin is fixed.
Includes Gaussian and full London centre/field/cross response.
The operator has no explicit B dependence.
No force sign. Native computation releases the GIL.
)doc");

    module.def("nuclear_attraction_mixed_derivatives_shell", [](const giao::Shell& a, const giao::Shell& b, const std::vector<giao::Nucleus>& nuclei,const py::object& field,py::object output) {
        const auto f=field_or_zero(field);
        return derivative_array(std::move(output),{2,3,3,static_cast<py::ssize_t>(a.ao_count()), static_cast<py::ssize_t>(b.ao_count())},
            [&](auto values) { giao::compute_nuclear_attraction_mixed_derivatives(a,b,nuclei,f,values); });
    }, py::arg("a"), py::arg("b"), py::arg("nuclei"), py::kw_only(),py::arg("field")=py::none(),py::arg("out")=py::none(),
    R"doc(Literal mixed d2I/dR dB for nuclear_attraction.

Returns complex128 shape (2, 3, 3, na, nb); axes are centre slot, nuclear xyz,
field xyz, then AO axes where present. The magnetic gauge origin is fixed.
Includes Gaussian and full London centre/field/cross response.
The operator has no explicit B dependence.
No force sign. Native computation releases the GIL.
out: optional strict writable C-contiguous complex128 buffer, returned exactly.
)doc");

    module.def("primitive_eri_mixed_derivatives", [](const giao::PrimitiveGaussian& a, const giao::PrimitiveGaussian& b, const giao::PrimitiveGaussian& c, const giao::PrimitiveGaussian& d,const py::object& field) {
        const auto f=field_or_zero(field);
        std::array<giao::Complex,36> values{};
        { py::gil_scoped_release release;
          values=giao::primitive_eri_mixed_derivatives(a,b,c,d,f); }
        return derivative_values(values,{4,3,3});
    }, py::arg("a"), py::arg("b"), py::arg("c"), py::arg("d"), py::kw_only(),py::arg("field")=py::none(),
    R"doc(Literal mixed d2I/dR dB for eri.

Returns complex128 shape (4, 3, 3); axes are centre slot, nuclear xyz,
field xyz, then AO axes where present. The magnetic gauge origin is fixed.
Includes Gaussian and full London centre/field/cross response.
The operator has no explicit B dependence.
No force sign. Native computation releases the GIL.
)doc");

    module.def("eri_mixed_derivatives_shell", [](const giao::Shell& a, const giao::Shell& b, const giao::Shell& c, const giao::Shell& d,const py::object& field,py::object output) {
        const auto f=field_or_zero(field);
        return derivative_array(std::move(output),{4,3,3,static_cast<py::ssize_t>(a.ao_count()), static_cast<py::ssize_t>(b.ao_count()), static_cast<py::ssize_t>(c.ao_count()), static_cast<py::ssize_t>(d.ao_count())},
            [&](auto values) { giao::compute_eri_mixed_derivatives(a,b,c,d,f,values); });
    }, py::arg("a"), py::arg("b"), py::arg("c"), py::arg("d"), py::kw_only(),py::arg("field")=py::none(),py::arg("out")=py::none(),
    R"doc(Literal mixed d2I/dR dB for eri.

Returns complex128 shape (4, 3, 3, na, nb, nc, nd); axes are centre slot, nuclear xyz,
field xyz, then AO axes where present. The magnetic gauge origin is fixed.
Includes Gaussian and full London centre/field/cross response.
The operator has no explicit B dependence.
No force sign. Native computation releases the GIL.
out: optional strict writable C-contiguous complex128 buffer, returned exactly.
)doc");

    module.def("primitive_nuclear_attraction_nucleus_mixed_derivatives",[](const giao::PrimitiveGaussian& a,const giao::PrimitiveGaussian& b,
        const std::vector<giao::Nucleus>& nuclei,const py::object& field,py::object output) {
        const auto f=field_or_zero(field);
        return derivative_array(std::move(output),{static_cast<py::ssize_t>(nuclei.size()),3,3},
            [&](auto values) { giao::primitive_nuclear_attraction_nucleus_mixed_derivatives(a,b,nuclei,f,values); });
    },py::arg("a"),py::arg("b"),py::arg("nuclei"),py::kw_only(),py::arg("field")=py::none(),py::arg("out")=py::none(),
    R"doc(Potential-centre mixed attraction derivatives, complex128 shape (nnucleus, 3, 3).

Axes: nucleus, nuclear xyz, field xyz. nuclei lists independent potential
centres. Includes full London response at fixed basis centres and gauge origin.
The Coulomb operator has no explicit B dependence. No force sign.
out: optional strict writable C-contiguous complex128 buffer, returned exactly.
Native computation releases the GIL.
)doc");

    module.def("nuclear_attraction_nucleus_mixed_derivatives_shell",[](const giao::Shell& a,const giao::Shell& b,
        const std::vector<giao::Nucleus>& nuclei,const py::object& field,py::object output) {
        const auto f=field_or_zero(field);
        return derivative_array(std::move(output),{static_cast<py::ssize_t>(nuclei.size()),3,3,static_cast<py::ssize_t>(a.ao_count()),static_cast<py::ssize_t>(b.ao_count())},
            [&](auto values) { giao::compute_nuclear_attraction_nucleus_mixed_derivatives(a,b,nuclei,f,values); });
    },py::arg("a"),py::arg("b"),py::arg("nuclei"),py::kw_only(),py::arg("field")=py::none(),py::arg("out")=py::none(),
    R"doc(Potential-centre mixed attraction derivatives, complex128 shape (nnucleus, 3, 3, na, nb).

Axes: nucleus, nuclear xyz, field xyz, bra AO, ket AO. nuclei lists independent potential
centres. Includes full London response at fixed basis centres and gauge origin.
The Coulomb operator has no explicit B dependence. No force sign.
out: optional strict writable C-contiguous complex128 buffer, returned exactly.
Native computation releases the GIL.
)doc");

    module.def("nuclear_attraction_mixed_derivatives",
        [](const giao::Basis& basis, const std::vector<giao::Nucleus>& nuclei,
           const py::object& field, const py::object& output, const py::object& threads) {
            const auto f = field_or_zero(field);
            const auto thread_count = threads_from_python(threads);
            py::object shell_out = py::none();
            py::object nucleus_out = py::none();
            if (!output.is_none()) {
                if (!py::isinstance<py::tuple>(output) || py::len(output) != 2) {
                    throw py::value_error("out must be None or a (shell_out, nucleus_out) tuple");
                }
                shell_out = output[py::int_(0)];
                nucleus_out = output[py::int_(1)];
            }
            const auto n = static_cast<py::ssize_t>(basis.ao_count());
            auto centers = validate_or_create_output(shell_out,
                {static_cast<py::ssize_t>(basis.shells().size()), 3, 3, n, n});
            auto potentials = validate_or_create_output(nucleus_out, {static_cast<py::ssize_t>(nuclei.size()), 3, 3, n, n});
            const auto first = reinterpret_cast<std::uintptr_t>(centers.data());
            const auto second = reinterpret_cast<std::uintptr_t>(potentials.data());
            if (centers.nbytes() != 0 && potentials.nbytes() != 0 &&
                first < second + static_cast<std::uintptr_t>(potentials.nbytes()) &&
                second < first + static_cast<std::uintptr_t>(centers.nbytes())) {
                throw py::value_error("shell and potential outputs must not overlap");
            }
            { py::gil_scoped_release release;
              giao::compute_nuclear_attraction_mixed_derivative_matrices(basis, nuclei, f,
                  {static_cast<giao::Complex*>(centers.mutable_data()), static_cast<std::size_t>(centers.size())},
                  {static_cast<giao::Complex*>(potentials.mutable_data()), static_cast<std::size_t>(potentials.size())}, thread_count); }
            return py::make_tuple(centers, potentials);
        }, py::arg("basis"), py::arg("nuclei"), py::kw_only(), py::arg("field") = py::none(),
        py::arg("out") = py::none(), py::arg("threads") = py::none(),
        R"doc(Return (shell-centre,potential-centre) mixed attraction derivatives.

Shapes (nshell,3,3,nao,nao), (nnucleus,3,3,nao,nao); axes are centre,
nuclear xyz, field xyz, bra AO, ket AO. Literal d2I/dR dB, no force sign.
Includes full Gaussian/London cross response at fixed gauge origin. The
Coulomb operator has no explicit B term. out=(shell_out,nucleus_out) uses
strict nonoverlapping buffers; either may be None. threads=None follows
the configured policy; native work releases the GIL.
)doc");

    py::class_<giao::EriDerivativeSchwarzBounds>(module,"_EriDerivativeSchwarzBounds")
        .def(py::init([](const giao::Basis& basis,const py::object& field,const py::object& threads) {
            const auto f=field_or_zero(field);
            const auto count=threads_from_python(threads);
            std::unique_ptr<giao::EriDerivativeSchwarzBounds> result;
            { py::gil_scoped_release release;
              result=std::make_unique<giao::EriDerivativeSchwarzBounds>(basis,f,count); }
            return result;
        }),py::arg("basis"),py::kw_only(),py::arg("field")=py::none(),py::arg("threads")=py::none(),py::keep_alive<1,2>())
        .def("quartet_bound",[](const giao::EriDerivativeSchwarzBounds& bounds,const std::array<std::uint32_t,4>& q) {
            return bounds.quartet_bound({q[0],q[1],q[2],q[3]});
        });

    module.def("_eri_derivative_batch",[](const giao::Basis& basis,
        const std::vector<std::array<std::uint32_t,4>>& requested,const py::object& field,
        double threshold,const py::object& threads,const giao::EriDerivativeSchwarzBounds* bounds) {
        const auto f=field_or_zero(field);
        const auto count=threads_from_python(threads);
        if (!std::isfinite(threshold) || threshold<0.0) {
            throw py::value_error("screening_threshold must be finite and non-negative");
        }
        if (threshold>0.0 && bounds==nullptr) {throw py::value_error("derivative screening requires derivative bounds");}
        if (bounds!=nullptr && !bounds->matches(basis,f)) {throw py::value_error("derivative bounds do not match basis and field");}
        std::vector<giao::ShellQuartetIndex> quartets;
        std::vector<std::size_t> offsets{0U};
        std::vector<std::array<std::size_t,6>> shapes;
        {
            py::gil_scoped_release release;
            for (const auto& indices:requested) {
                const giao::ShellQuartetIndex q{indices[0],indices[1],indices[2],indices[3]};
                for (const auto index:indices) {
                    if (index>=basis.shells().size()) {throw std::out_of_range("derivative shell quartet index is out of range");}
                }
                if (threshold>0.0 && bounds->quartet_bound(q)<threshold) {continue;}
                const auto& a=basis.shells()[q.a];const auto& b=basis.shells()[q.b];
                const auto& c=basis.shells()[q.c];const auto& d=basis.shells()[q.d];
                const auto size=giao::shell_quartet_size(a,b,c,d);
                const auto maximum=static_cast<std::size_t>(std::numeric_limits<py::ssize_t>::max());
                if (size>(maximum-offsets.back())/12U) {throw std::length_error("derivative batch size overflow");}
                offsets.push_back(offsets.back()+12U*size);
                quartets.push_back(q);
                shapes.push_back({4U,3U,a.ao_count(),b.ao_count(),c.ao_count(),d.ao_count()});
            }
        }
        py::array_t<std::uint32_t> quartet_result({static_cast<py::ssize_t>(quartets.size()),py::ssize_t{4}});
        py::array_t<std::int64_t> shape_result({static_cast<py::ssize_t>(quartets.size()),py::ssize_t{6}});
        py::array_t<std::int64_t> offset_result(offsets.size());
        py::array_t<giao::Complex> values(offsets.back());
        auto qview=quartet_result.mutable_unchecked<2>();
        auto shape=shape_result.mutable_unchecked<2>();
        for (std::size_t i=0;i<quartets.size();++i) {
            for (std::size_t k=0;k<4U;++k) {qview(static_cast<py::ssize_t>(i),static_cast<py::ssize_t>(k))=quartets[i].as_array()[k];}
            for (std::size_t k=0;k<6U;++k) {shape(static_cast<py::ssize_t>(i),static_cast<py::ssize_t>(k))=static_cast<std::int64_t>(shapes[i][k]);}
        }
        for (std::size_t i=0;i<offsets.size();++i) {offset_result.mutable_data()[i]=static_cast<std::int64_t>(offsets[i]);}
        {
            py::gil_scoped_release release;
            giao::compute_eri_derivative_blocks(basis,quartets,offsets,f,
                {values.mutable_data(),static_cast<std::size_t>(values.size())},count);
        }
        return py::make_tuple(quartet_result,shape_result,offset_result,values,requested.size(),requested.size()-quartets.size());
    },py::arg("basis"),py::arg("quartets"),py::kw_only(),py::arg("field")=py::none(),
    py::arg("screening_threshold")=0.0,py::arg("threads")=py::none(),py::arg("bounds")=py::none());

    module.def("_cholesky_diagonal",[](const giao::Basis& basis,
        const std::vector<std::array<std::size_t,2>>& pairs,const py::object& field,const py::object& threads) {
        const auto f=field_or_zero(field);const auto count=threads_from_python(threads);
        const auto n=static_cast<py::ssize_t>(basis.ao_count());
        py::array_t<double> result({n,n});
        {py::gil_scoped_release release;giao::detail::cholesky_diagonal(basis,pairs,f,
            {result.mutable_data(),static_cast<std::size_t>(result.size())},count);}
        return result;
    },py::arg("basis"),py::arg("pairs"),py::kw_only(),py::arg("field")=py::none(),py::arg("threads")=py::none());
    module.def("_cholesky_columns",[](const giao::Basis& basis,
        const std::vector<std::array<std::size_t,2>>& pivots,const py::object& field,const py::object& threads) -> py::array {
        const auto f=field_or_zero(field);const auto count=threads_from_python(threads);
        const auto n=static_cast<py::ssize_t>(basis.ao_count());
        const auto rows=static_cast<py::ssize_t>(pivots.size());
        if (f.B.x==0.0 && f.B.y==0.0 && f.B.z==0.0) {
            py::array_t<double> result({rows,n*n});
            {py::gil_scoped_release release;giao::detail::cholesky_columns(basis,pivots,f,
                {result.mutable_data(),static_cast<std::size_t>(result.size())},count);}
            return result;
        }
        py::array_t<giao::Complex> result({rows,n*n});
        {py::gil_scoped_release release;giao::detail::cholesky_columns(basis,pivots,f,
            {result.mutable_data(),static_cast<std::size_t>(result.size())},count);}
        return result;
    },py::arg("basis"),py::arg("pivots"),py::kw_only(),py::arg("field")=py::none(),py::arg("threads")=py::none());
    module.def("moment_mixed_derivatives",
        [](const giao::Basis& basis, const py::object& powers, const py::object& origin,
           const py::object& field, const py::object& output, const py::object& threads) {
            const auto moment = moment_from_python(powers, origin);
            const auto f = field_or_zero(field);
            const auto thread_count = threads_from_python(threads);
            py::object shell_out = py::none();
            py::object origin_out = py::none();
            if (!output.is_none()) {
                if (!py::isinstance<py::tuple>(output) || py::len(output) != 2) {
                    throw py::value_error("out must be None or a (shell_out, origin_out) tuple");
                }
                shell_out = output[py::int_(0)];
                origin_out = output[py::int_(1)];
            }
            const auto n = static_cast<py::ssize_t>(basis.ao_count());
            auto centers = validate_or_create_output(shell_out,
                {static_cast<py::ssize_t>(basis.shells().size()), 3, 3, n, n});
            auto origins = validate_or_create_output(origin_out, {3, 3, n, n});
            const auto first = reinterpret_cast<std::uintptr_t>(centers.data());
            const auto second = reinterpret_cast<std::uintptr_t>(origins.data());
            if (centers.nbytes() != 0 && origins.nbytes() != 0 &&
                first < second + static_cast<std::uintptr_t>(origins.nbytes()) &&
                second < first + static_cast<std::uintptr_t>(centers.nbytes())) {
                throw py::value_error("shell and origin outputs must not overlap");
            }
            { py::gil_scoped_release release;
              giao::compute_moment_mixed_derivative_matrices(basis, moment, f,
                  {static_cast<giao::Complex*>(centers.mutable_data()), static_cast<std::size_t>(centers.size())},
                  {static_cast<giao::Complex*>(origins.mutable_data()), static_cast<std::size_t>(origins.size())}, thread_count); }
            return py::make_tuple(centers, origins);
        }, py::arg("basis"), py::arg("powers"), py::kw_only(),
        py::arg("origin") = py::make_tuple(0.0, 0.0, 0.0), py::arg("field") = py::none(),
        py::arg("out") = py::none(), py::arg("threads") = py::none(),
        R"doc(Return (shell-centre, moment-origin) mixed d2I/dR dB responses.

basis is a Cartesian basis; powers selects (r-origin)^powers. origin defaults
to zero. field supplies the phase B and fixed magnetic gauge origin.
Shapes: (nshell,3,3,nao,nao) and (3,3,nao,nao), nuclear/origin axis then B axis.
Includes full Gaussian/London cross response; the moment operator has no explicit
B dependence. The moment origin is fixed for the basis response, and its own
mixed response is returned separately. No force or charge signs.
out=(shell_out,origin_out), each None or strict nonoverlapping complex128 buffer.
threads=None follows configured native shell-pair threading; work releases GIL.
)doc");

    module.def("_atom_one_electron_derivatives",[](const giao::Basis& basis,
        const std::vector<std::size_t>& atoms,std::size_t natom,const std::vector<giao::Nucleus>& nuclei,
        const std::string& name,bool mixed,const py::object& powers,const py::object& origin,
        const py::object& field,const py::object& output,const py::object& threads) -> py::object {
        giao::detail::AtomOperator op;
        if(name=="overlap"){op=giao::detail::AtomOperator::overlap;}
        else if(name=="kinetic"){op=giao::detail::AtomOperator::kinetic;}
        else if(name=="magnetic_kinetic"){op=giao::detail::AtomOperator::magnetic_kinetic;}
        else if(name=="nuclear_attraction"){op=giao::detail::AtomOperator::attraction;}
        else if(name=="moment"){op=giao::detail::AtomOperator::moment;}
        else{throw py::value_error("unknown atom derivative operator");}
        const auto moment=moment_from_python(powers,origin);
        const auto f=field_or_zero(field);const auto count=threads_from_python(threads);
        const auto n=static_cast<py::ssize_t>(basis.ao_count());
        std::vector<py::ssize_t> shape{static_cast<py::ssize_t>(natom),3};
        std::vector<py::ssize_t> origin_shape{3};
        if(mixed){shape.push_back(3);origin_shape.push_back(3);}
        shape.insert(shape.end(),{n,n});origin_shape.insert(origin_shape.end(),{n,n});
        py::object atom_out=output,origin_out=py::none();
        if(name=="moment" && !output.is_none()){
            if(!py::isinstance<py::tuple>(output) || py::len(output)!=2){throw py::value_error("moment out must be an (atom_out,origin_out) tuple");}
            atom_out=output[py::int_(0)];origin_out=output[py::int_(1)];
        }
        auto result=validate_or_create_output(atom_out,shape);
        py::array origin_result;
        std::span<giao::Complex> origin_values;
        if(name=="moment"){
            origin_result=validate_or_create_output(origin_out,origin_shape);
            const auto first=reinterpret_cast<std::uintptr_t>(result.data());
            const auto second=reinterpret_cast<std::uintptr_t>(origin_result.data());
            if(result.nbytes()!=0 && origin_result.nbytes()!=0 && first<second+static_cast<std::uintptr_t>(origin_result.nbytes()) && second<first+static_cast<std::uintptr_t>(result.nbytes())){throw py::value_error("atom and origin outputs must not overlap");}
            origin_values={static_cast<giao::Complex*>(origin_result.mutable_data()),static_cast<std::size_t>(origin_result.size())};
        }
        {py::gil_scoped_release release;giao::detail::atom_one_electron_derivatives(basis,atoms,natom,nuclei,f,op,moment,mixed,
            {static_cast<giao::Complex*>(result.mutable_data()),static_cast<std::size_t>(result.size())},origin_values,count);}
        if(name=="moment"){return py::make_tuple(result,origin_result);}
        return result;
    },py::arg("basis"),py::arg("shell_atoms"),py::arg("natom"),py::arg("nuclei"),py::arg("name"),
    py::arg("mixed"),py::arg("powers"),py::arg("origin"),py::kw_only(),py::arg("field")=py::none(),
    py::arg("out")=py::none(),py::arg("threads")=py::none());

    module.def("_contract_eri_derivative_batch",[](const giao::Basis& basis,
        const std::vector<std::size_t>& atoms,std::size_t natom,
        const std::vector<std::array<std::uint32_t,4>>& requested,
        const std::vector<std::size_t>& offsets,py::array values,
        const std::vector<std::array<py::array,2>>& coulomb,
        const std::vector<std::array<py::array,2>>& exchange,const py::object& gamma,
        py::object output,const py::object& threads){
        const auto count=threads_from_python(threads);
        std::vector<giao::ShellQuartetIndex> quartets;
        for(const auto& q:requested){quartets.push_back({q[0],q[1],q[2],q[3]});}
        auto result=validate_or_create_output(output,{static_cast<py::ssize_t>(natom),3});
        const auto n=static_cast<py::ssize_t>(basis.ao_count());
        auto matrix_span=[&](const py::array& value){
            if(!value.dtype().is(py::dtype::of<giao::Complex>()) || !(value.flags()&py::array::c_style) || value.ndim()!=2 || value.shape(0)!=n || value.shape(1)!=n){throw py::value_error("density matrices must be C-contiguous complex128 (nao,nao)");}
            return std::span<const giao::Complex>(static_cast<const giao::Complex*>(value.data()),static_cast<std::size_t>(value.size()));
        };
        std::vector<giao::detail::DensityPair> jp,kp;
        for(const auto& pair:coulomb){jp.push_back({matrix_span(pair[0]),matrix_span(pair[1])});}
        for(const auto& pair:exchange){kp.push_back({matrix_span(pair[0]),matrix_span(pair[1])});}
        if(!values.dtype().is(py::dtype::of<giao::Complex>()) || !(values.flags()&py::array::c_style) || values.ndim()!=1){throw py::value_error("derivative values must be contiguous complex128");}
        std::span<const giao::Complex> complex_gamma;
        std::span<const double> real_gamma;
        if(!gamma.is_none()){
            if(!py::isinstance<py::array>(gamma)){throw py::value_error("gamma must be an ndarray");}
            const auto array=gamma.cast<py::array>();
            if(!(array.flags()&py::array::c_style) || array.ndim()!=4){throw py::value_error("gamma must be C-contiguous with four AO axes");}
            for(py::ssize_t axis=0;axis<4;++axis){if(array.shape(axis)!=n){throw py::value_error("gamma AO shape mismatch");}}
            if(array.dtype().is(py::dtype::of<giao::Complex>())){complex_gamma={static_cast<const giao::Complex*>(array.data()),static_cast<std::size_t>(array.size())};}
            else if(array.dtype().is(py::dtype::of<double>())){real_gamma={static_cast<const double*>(array.data()),static_cast<std::size_t>(array.size())};}
            else{throw py::value_error("gamma dtype must be float64 or complex128");}
        }
        {py::gil_scoped_release release;giao::detail::contract_eri_derivatives(basis,atoms,natom,quartets,offsets,
            {static_cast<const giao::Complex*>(values.data()),static_cast<std::size_t>(values.size())},jp,kp,complex_gamma,real_gamma,
            {static_cast<giao::Complex*>(result.mutable_data()),static_cast<std::size_t>(result.size())},count);}
        return result;
    },py::arg("basis"),py::arg("shell_atoms"),py::arg("natom"),py::arg("quartets"),py::arg("offsets"),
    py::arg("values"),py::arg("coulomb_pairs"),py::arg("exchange_pairs"),py::arg("gamma"),py::arg("out"),
    py::kw_only(),py::arg("threads")=py::none());

    module.def("primitive_kinetic_angular_momentum",[](const giao::PrimitiveGaussian& a,const giao::PrimitiveGaussian& b,const std::string& component,const py::object& origin,const py::object& field){
        const auto f=field_or_zero(field);
const auto axis=axis_from_python(component);
        const giao::CartesianMoment validated_origin(vec3_from_python(origin,"origin"),{});
        const auto g=validated_origin.origin;
        py::gil_scoped_release release;giao::IntegralWorkspace workspace;
        return giao::primitive_kinetic_angular_momentum(a,b,axis,g,f,workspace);
    },py::arg("a"),py::arg("b"),py::arg("component"),py::kw_only(),py::arg("origin")=py::make_tuple(0.0,0.0,0.0),py::arg("field")=py::none(),R"doc(Return complex scalar of [(r-G) cross pi0]_component.

component is x/y/z; origin is the independent operator origin G in bohr,
default zero. field supplies the fixed London phase field B0 and gauge G0.
pi0=p+A0, p=-i*gradient, A0=B0 cross (r-G0)/2. Both canonical and vector-
potential terms are included. G does not change B0, G0 or the London phases.
Cartesian AO order/normalization and the p+A convention
match magnetic_kinetic. Native computation releases the GIL.
)doc");

    module.def("kinetic_angular_momentum_shell",[](const giao::Shell& a,const giao::Shell& b,const std::string& component,const py::object& origin,const py::object& field,py::object output){
        const auto f=field_or_zero(field);
const auto axis=axis_from_python(component);
        const giao::CartesianMoment validated_origin(vec3_from_python(origin,"origin"),{});
        const auto g=validated_origin.origin;
        return shell_array(a,b,std::move(output),[&](auto values,auto& workspace){
            giao::compute_kinetic_angular_momentum(a,b,axis,g,f,values,workspace);
        });
    },py::arg("a"),py::arg("b"),py::arg("component"),py::kw_only(),py::arg("origin")=py::make_tuple(0.0,0.0,0.0),py::arg("field")=py::none(),py::arg("out")=py::none(),R"doc(Return complex128 (na,nb) shell block of [(r-G) cross pi0]_component.

component is x/y/z; origin is the independent operator origin G in bohr,
default zero. field supplies the fixed London phase field B0 and gauge G0.
pi0=p+A0, p=-i*gradient, A0=B0 cross (r-G0)/2. Both canonical and vector-
potential terms are included. G does not change B0, G0 or the London phases.
Cartesian AO order/normalization and the p+A convention
match magnetic_kinetic. Native computation releases the GIL.
out is a strict writable C-contiguous complex128 buffer, returned exactly.
)doc");

    module.def("kinetic_angular_momentum",[](const giao::Basis& basis,const std::string& component,const py::object& origin,const py::object& field,py::object output,const py::object& threads){
        const auto f=field_or_zero(field);
const auto axis=axis_from_python(component);
        const giao::CartesianMoment validated_origin(vec3_from_python(origin,"origin"),{});
        const auto g=validated_origin.origin;
        const auto count=threads_from_python(threads);
        return basis_array(basis,std::move(output),[&](auto values){
            giao::compute_kinetic_angular_momentum_matrix(basis,axis,g,f,values,count);
        });
    },py::arg("basis"),py::arg("component"),py::kw_only(),py::arg("origin")=py::make_tuple(0.0,0.0,0.0),py::arg("field")=py::none(),py::arg("out")=py::none(),py::arg("threads")=py::none(),R"doc(Return complex128 (nao,nao) AO matrix of [(r-G) cross pi0]_component.

component is x/y/z; origin is the independent operator origin G in bohr,
default zero. field supplies the fixed London phase field B0 and gauge G0.
pi0=p+A0, p=-i*gradient, A0=B0 cross (r-G0)/2. Both canonical and vector-
potential terms are included. G does not change B0, G0 or the London phases.
Cartesian AO order/normalization and the p+A convention
match magnetic_kinetic. Native computation releases the GIL.
out is a strict writable C-contiguous complex128 buffer, returned exactly.
threads=None follows the configured native shell-pair policy.
)doc");

    module.def("primitive_magnetic_kinetic_operator",[](const giao::PrimitiveGaussian& a,const giao::PrimitiveGaussian& b,const py::object& operator_field,const py::object& origin,const py::object& field){
        const auto f=field_or_zero(field);
const giao::MagneticField operator_value(operator_field.is_none()?f.B:vec3_from_python(operator_field,"operator_field"),origin.is_none()?f.gauge_origin:vec3_from_python(origin,"origin"));
        py::gil_scoped_release release;giao::IntegralWorkspace workspace;
        return giao::primitive_magnetic_kinetic_operator(a,b,f,operator_value,workspace);
    },py::arg("a"),py::arg("b"),py::kw_only(),py::arg("operator_field")=py::none(),py::arg("origin")=py::none(),py::arg("field")=py::none(),R"doc(Return complex scalar of 1/2*[p+Aop]^2 with fixed London phases.

field supplies the phase field B0/gauge G0. operator_field is the independent
three-vector C; origin is its independent origin G, with Aop=C cross (r-G)/2.
Defaults C=B0 and G=G0 reproduce magnetic_kinetic bitwise; C=0 gives canonical
kinetic. Includes canonical, linear C angular and quadratic C second-moment
terms. Changing C or G never changes the London phases. No force/charge sign.
Native computation releases the GIL.
)doc");

    module.def("magnetic_kinetic_operator_shell",[](const giao::Shell& a,const giao::Shell& b,const py::object& operator_field,const py::object& origin,const py::object& field,py::object output){
        const auto f=field_or_zero(field);
const giao::MagneticField operator_value(operator_field.is_none()?f.B:vec3_from_python(operator_field,"operator_field"),origin.is_none()?f.gauge_origin:vec3_from_python(origin,"origin"));
        return shell_array(a,b,std::move(output),[&](auto values,auto& workspace){
            giao::compute_magnetic_kinetic_operator(a,b,f,operator_value,values,workspace);
        });
    },py::arg("a"),py::arg("b"),py::kw_only(),py::arg("operator_field")=py::none(),py::arg("origin")=py::none(),py::arg("field")=py::none(),py::arg("out")=py::none(),R"doc(Return complex128 (na,nb) shell block of 1/2*[p+Aop]^2 with fixed London phases.

field supplies the phase field B0/gauge G0. operator_field is the independent
three-vector C; origin is its independent origin G, with Aop=C cross (r-G)/2.
Defaults C=B0 and G=G0 reproduce magnetic_kinetic bitwise; C=0 gives canonical
kinetic. Includes canonical, linear C angular and quadratic C second-moment
terms. Changing C or G never changes the London phases. No force/charge sign.
Native computation releases the GIL.
out is a strict writable C-contiguous complex128 buffer, returned exactly.
)doc");

    module.def("magnetic_kinetic_operator",[](const giao::Basis& basis,const py::object& operator_field,const py::object& origin,const py::object& field,py::object output,const py::object& threads){
        const auto f=field_or_zero(field);
const giao::MagneticField operator_value(operator_field.is_none()?f.B:vec3_from_python(operator_field,"operator_field"),origin.is_none()?f.gauge_origin:vec3_from_python(origin,"origin"));
        const auto count=threads_from_python(threads);
        return basis_array(basis,std::move(output),[&](auto values){
            giao::compute_magnetic_kinetic_operator_matrix(basis,f,operator_value,values,count);
        });
    },py::arg("basis"),py::kw_only(),py::arg("operator_field")=py::none(),py::arg("origin")=py::none(),py::arg("field")=py::none(),py::arg("out")=py::none(),py::arg("threads")=py::none(),R"doc(Return complex128 (nao,nao) AO matrix of 1/2*[p+Aop]^2 with fixed London phases.

field supplies the phase field B0/gauge G0. operator_field is the independent
three-vector C; origin is its independent origin G, with Aop=C cross (r-G)/2.
Defaults C=B0 and G=G0 reproduce magnetic_kinetic bitwise; C=0 gives canonical
kinetic. Includes canonical, linear C angular and quadratic C second-moment
terms. Changing C or G never changes the London phases. No force/charge sign.
Native computation releases the GIL.
out is a strict writable C-contiguous complex128 buffer, returned exactly.
threads=None follows the configured native shell-pair policy.
)doc");

}
