#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <iostream>
#include <iomanip> // Para formatear la salida

/**
 * Muestra una matriz en la consola (usada para depuración o visualización).
 * @tparam T Tipo de dato de la matriz.
 * @param matrix Matriz a mostrar.
 */
template <typename T>
void display_matrix(const std::vector<std::vector<T>>& matrix) {
    for (const auto& row : matrix) {
        for (const auto& value : row) {
            std::cout << std::setw(5) << value << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

/**
 * Visualiza una imagen en la consola (por ejemplo, MNIST).
 * Los valores mayores a un umbral se muestran como '1', los demás como espacio.
 * @tparam T Tipo de dato del vector de la imagen.
 * @param image Imagen a visualizar (en formato vector).
 * @param rows Número de filas de la imagen.
 * @param columns Número de columnas de la imagen.
 */
template <typename T>
void display_image(const std::vector<T>& image, int rows, int columns) {
    if (image.size() != static_cast<size_t>(rows * columns)) {
        throw std::invalid_argument("El tamaño de la imagen no coincide con las dimensiones proporcionadas.");
    }

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < columns; ++j) {
            // Mostrar '1' si el valor supera un umbral, de lo contrario espacio
            std::cout << (image[i * columns + j] > static_cast<T>(0.5) ? "1" : " ") << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

/**
 * Muestra un vector en la consola (usado para depuración).
 * @tparam T Tipo de dato del vector.
 * @param vec Vector a mostrar.
 */
template <typename T>
void display_vector(const std::vector<T>& vec) {
    for (const auto& value : vec) {
        std::cout << value << " ";
    }
    std::cout << std::endl;
}

#endif // UTILS_H
