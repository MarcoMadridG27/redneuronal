#include <iostream>
#include "../include/common.h"
#include "../include/dataset.h"
#include "../include/network.h"
#include "../include/utils.h"

int main() {
    try {
        // Crear el dataset
        Dataset<double> mnist(
                "../data/train-images.idx3-ubyte",
                "../data/train-labels.idx1-ubyte",
                "../data/t10k-images.idx3-ubyte",
                "../data/t10k-labels.idx1-ubyte"
        );

        // Obtener las imágenes y etiquetas
        const auto& train_images = mnist.get_training_images();
        const auto& train_labels = mnist.get_training_labels();
        const auto& test_images = mnist.get_test_images();
        const auto& test_labels = mnist.get_test_labels();

        // Convertir etiquetas a formato one-hot
        std::vector<Vector<double>> train_labels_one_hot;
        for (int label : train_labels) {
            train_labels_one_hot.push_back(one_hot_encode<double>(label, OUTPUT_SIZE));
        }

        // Crear la red neuronal
        NeuralNetwork<double> nn({INPUT_SIZE, 128, OUTPUT_SIZE}, 0.001);

        // Entrenar la red neuronal
        std::cout << "Entrenando la red neuronal..." << std::endl;
        nn.train(train_images, train_labels_one_hot, 3);

        // Evaluar la red en el conjunto de prueba
        double accuracy = nn.evaluate(test_images, test_labels);
        std::cout << "Precisión en el conjunto de prueba: " << accuracy << "%" << std::endl;

        // Realizar predicción para una imagen del conjunto de prueba
        int index = 0; // Cambiar para probar diferentes imágenes
        int predicted_label = nn.predict(test_images[index]);

        // Mostrar resultados
        std::cout << "Etiqueta real: " << test_labels[index] << std::endl;
        std::cout << "Etiqueta predicha: " << predicted_label << std::endl;

        // Visualizar la imagen
        display_image(test_images[index], 28, 28);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}
