#include "giao_integrals/boys.hpp"
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
           double screening_threshold, std::size_t threads) {
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
        py::arg("screening_threshold") = 0.0, py::arg("threads") = 1U);

    module.def(
        "_eri_batch",
        [](const giao::Basis& basis,
           const std::vector<std::array<std::uint32_t, 4>>& quartet_arrays,
           const py::object& field, double screening_threshold,
           std::size_t threads, const giao::EriSchwarzBounds* bounds) {
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
        py::arg("screening_threshold") = 0.0, py::arg("threads") = 1U,
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
}
