#include "giao_integrals/boys.hpp"
#include "giao_integrals/nuclear.hpp"
#include "giao_integrals/overlap.hpp"
#include "giao_integrals/property.hpp"
#include "giao_integrals/types.hpp"

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

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

}  // namespace

PYBIND11_MODULE(_giao_integrals, module) {
    module.doc() = "C++20 one-electron integrals for Cartesian GIAO/London Gaussians";
    module.attr("__version__") = "0.4.0";

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
}
