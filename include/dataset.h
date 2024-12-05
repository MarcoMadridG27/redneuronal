#ifndef DATASET_H
#define DATASET_H

#include <string>
#include <vector>
#include <fstream>
#include <stdexcept>
#include "common.h" // Incluye funciones para endian conversion y file_header_t

template <typename T>
class Dataset {
private:
    Matrix<T> training_images;
    std::vector<int> training_labels;
    Matrix<T> test_images;
    std::vector<int> test_labels;

    // Función privada para leer imágenes desde un archivo
    Matrix<T> read_images(const std::string& file_path) {
        std::ifstream file(file_path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Error: no se pudo abrir el archivo de imágenes " + file_path);
        }

        // Leer encabezado
        file_header_t header;
        file.read(reinterpret_cast<char*>(&header), sizeof(header));
        header = convert_big_endian(header);

        // Validar el encabezado
        if (header.rows == 0 || header.columns == 0 || header.images == 0) {
            throw std::runtime_error("Error: el archivo de imágenes tiene dimensiones inválidas.");
        }

        // Leer imágenes
        Matrix<T> images(header.images, Vector<T>(header.rows * header.columns));
        for (auto& image : images) {
            std::vector<uint8_t> buffer(header.rows * header.columns);
            file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
            if (file.gcount() != static_cast<std::streamsize>(buffer.size())) {
                throw std::runtime_error("Error: no se pudieron leer todas las imágenes del archivo.");
            }
            for (size_t i = 0; i < buffer.size(); ++i) {
                image[i] = static_cast<T>(buffer[i]) / static_cast<T>(255.0); // Normalización
            }
        }
        return images;
    }

    // Función privada para leer etiquetas desde un archivo
    std::vector<int> read_labels(const std::string& file_path) {
        std::ifstream file(file_path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Error: no se pudo abrir el archivo de etiquetas " + file_path);
        }

        // Leer encabezado
        uint32_t magic_number, num_items;
        file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
        file.read(reinterpret_cast<char*>(&num_items), sizeof(num_items));
        magic_number = to_big_endian(magic_number);
        num_items = to_big_endian(num_items);

        // Validar encabezado
        if (magic_number != 2049) {
            throw std::runtime_error("Error: el archivo de etiquetas no tiene un encabezado válido.");
        }

        // Leer etiquetas
        std::vector<int> labels(num_items);
        for (size_t i = 0; i < num_items; ++i) {
            uint8_t label;
            file.read(reinterpret_cast<char*>(&label), sizeof(label));
            if (file.gcount() != sizeof(label)) {
                throw std::runtime_error("Error: no se pudieron leer todas las etiquetas del archivo.");
            }
            labels[i] = static_cast<int>(label);
        }
        return labels;
    }

public:
    // Constructor que inicializa los datos de entrenamiento y prueba
    Dataset(const std::string& train_image_path,
            const std::string& train_label_path,
            const std::string& test_image_path,
            const std::string& test_label_path) {
        training_images = read_images(train_image_path);
        training_labels = read_labels(train_label_path);
        test_images = read_images(test_image_path);
        test_labels = read_labels(test_label_path);
    }

    // Métodos para acceder a los datos
    const Matrix<T>& get_training_images() const { return training_images; }
    const std::vector<int>& get_training_labels() const { return training_labels; }
    const Matrix<T>& get_test_images() const { return test_images; }
    const std::vector<int>& get_test_labels() const { return test_labels; }
};

#endif // DATASET_H
