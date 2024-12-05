#ifndef COMMON_H
#define COMMON_H

#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <iostream>
#include <algorithm> // Para std::max_element
#include <cstdint>   // Para uint32_t
#include <type_traits> // Para verificar tipos en plantillas

// Constantes globales
constexpr double EPSILON = 1e-6; // Pequeño valor para evitar divisiones por cero
constexpr int INPUT_SIZE = 784;  // Número de píxeles en las imágenes MNIST
constexpr int OUTPUT_SIZE = 10;  // Número de categorías (dígitos 0-9)

// Tipos de datos genéricos para manejar matrices y vectores
template <typename T>
using Matrix = std::vector<std::vector<T>>;
template <typename T>
using Vector = std::vector<T>;

// Estructura genérica para leer el encabezado del archivo MNIST
struct file_header_t {
    uint32_t magic{};
    uint32_t images{};
    uint32_t rows{};
    uint32_t columns{};
};

// Función para convertir valores de big endian a little endian
template <typename T>
T to_big_endian(T value) {
    static_assert(std::is_integral<T>::value, "Solo se admiten tipos enteros");
    if constexpr (sizeof(T) == 4) {
        return ((value >> 24) & 0xff) |
               ((value << 8) & 0xff0000) |
               ((value >> 8) & 0xff00) |
               ((value << 24) & 0xff000000);
    } else if constexpr (sizeof(T) == 2) {
        return ((value >> 8) & 0xff) | ((value << 8) & 0xff00);
    } else {
        return value;
    }
}

// Convierte el encabezado completo
inline file_header_t convert_big_endian(const file_header_t& src) {
    return {
            to_big_endian(src.magic),
            to_big_endian(src.images),
            to_big_endian(src.rows),
            to_big_endian(src.columns)
    };
}

// Funciones útiles

/**
 * Genera un número aleatorio en un rango específico.
 * @tparam T Tipo de dato.
 * @param min Valor mínimo.
 * @param max Valor máximo.
 * @return Número aleatorio en el rango [min, max].
 */
template <typename T>
T random_value(T min, T max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dis(min, max);
    return dis(gen);
}

/**
 * Inicializa una matriz con valores aleatorios.
 * @tparam T Tipo de dato.
 * @param rows Número de filas.
 * @param cols Número de columnas.
 * @return Matriz inicializada con valores aleatorios.
 */
template <typename T>
Matrix<T> initialize_matrix(int rows, int cols) {
    Matrix<T> mat(rows, Vector<T>(cols));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            mat[i][j] = random_value<T>(-0.5, 0.5); // Inicializa con valores pequeños
        }
    }
    return mat;
}

/**
 * Calcula el producto punto entre dos vectores.
 * @tparam T Tipo de dato.
 * @param a Primer vector.
 * @param b Segundo vector.
 * @return Producto punto de los vectores.
 */
template <typename T>
T dot_product(const Vector<T>& a, const Vector<T>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Los vectores deben tener el mismo tamaño.");
    }
    T result = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }
    return result;
}

/**
 * Calcula la transposición de una matriz.
 * @tparam T Tipo de dato.
 * @param mat Matriz original.
 * @return Matriz transpuesta.
 */
template <typename T>
Matrix<T> transpose(const Matrix<T>& mat) {
    if (mat.empty()) return {};
    Matrix<T> result(mat[0].size(), Vector<T>(mat.size()));
    for (size_t i = 0; i < mat.size(); ++i) {
        for (size_t j = 0; j < mat[0].size(); ++j) {
            result[j][i] = mat[i][j];
        }
    }
    return result;
}

/**
 * Aplica una función a todos los elementos de un vector.
 * @tparam T Tipo de dato.
 * @param vec Vector original.
 * @param func Función a aplicar.
 * @return Nuevo vector con la función aplicada.
 */
template <typename T, typename Function>
Vector<T> apply_function(const Vector<T>& vec, Function func) {
    Vector<T> result(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
        result[i] = func(vec[i]);
    }
    return result;
}

/**
 * Aplica una función a todos los elementos de una matriz.
 * @tparam T Tipo de dato.
 * @param mat Matriz original.
 * @param func Función a aplicar.
 * @return Nueva matriz con la función aplicada.
 */
template <typename T, typename Function>
Matrix<T> apply_function(const Matrix<T>& mat, Function func) {
    Matrix<T> result(mat.size(), Vector<T>(mat[0].size()));
    for (size_t i = 0; i < mat.size(); ++i) {
        for (size_t j = 0; j < mat[0].size(); ++j) {
            result[i][j] = func(mat[i][j]);
        }
    }
    return result;
}

/**
 * Calcula la función softmax sobre un vector.
 * @tparam T Tipo de dato.
 * @param vec Vector original.
 * @return Vector transformado con softmax.
 */
template <typename T>
Vector<T> softmax(const Vector<T>& vec) {
    T max_elem = *std::max_element(vec.begin(), vec.end());
    Vector<T> exp_values(vec.size());
    T sum_exp = 0;
    for (size_t i = 0; i < vec.size(); ++i) {
        exp_values[i] = std::exp(vec[i] - max_elem);
        sum_exp += exp_values[i];
    }
    for (T& value : exp_values) {
        value /= sum_exp;
    }
    return exp_values;
}

/**
 * Codifica una etiqueta en formato one-hot.
 * @tparam T Tipo de dato.
 * @param label Etiqueta a codificar.
 * @param num_classes Número total de clases.
 * @return Vector codificado en formato one-hot.
 */
template <typename T>
Vector<T> one_hot_encode(int label, size_t num_classes) {
    Vector<T> encoded(num_classes, static_cast<T>(0));
    encoded[label] = static_cast<T>(1);
    return encoded;
}

#endif // COMMON_H
