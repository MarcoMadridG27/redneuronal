#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <vector>
#include <cmath>
#include <algorithm>

namespace Activation {

    /**
     * Función de activación ReLU.
     * @tparam T Tipo de dato (por ejemplo, float, double).
     * @param x Valor de entrada.
     * @return Máximo entre 0 y x.
     */
    template <typename T>
    T relu(T x) {
        return std::max(static_cast<T>(0), x);
    }

    /**
     * Derivada de la función ReLU.
     * @tparam T Tipo de dato.
     * @param x Valor de entrada.
     * @return 1 si x > 0, de lo contrario 0.
     */
    template <typename T>
    T relu_derivative(T x) {
        return x > 0 ? static_cast<T>(1) : static_cast<T>(0);
    }

    /**
     * Función de activación Softmax.
     * @tparam T Tipo de dato.
     * @param x Vector de entrada.
     * @return Vector transformado con softmax.
     */
    template <typename T>
    std::vector<T> softmax(const std::vector<T>& x) {
        std::vector<T> exp_values(x.size());
        T max_elem = *std::max_element(x.begin(), x.end()); // Evitar overflow
        T sum_exp = 0;

        // Calcular exponenciales
        for (size_t i = 0; i < x.size(); ++i) {
            exp_values[i] = std::exp(x[i] - max_elem); // Exponencial estabilizada
            sum_exp += exp_values[i];
        }

        // Normalizar
        for (T& value : exp_values) {
            value /= sum_exp;
        }

        return exp_values;
    }
}

#endif // ACTIVATION_H
