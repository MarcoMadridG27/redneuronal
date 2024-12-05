#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include <functional> // Para funciones de activación
#include <cmath>
#include <stdexcept>
#include <random>
#include <iostream>
#include "common.h"   // Constantes y funciones comunes

template <typename T>
class NeuralNetwork {
private:
    std::vector<Matrix<T>> weights;     // Pesos entre las capas
    std::vector<Vector<T>> biases;      // Sesgos para cada capa
    std::vector<Vector<T>> activations; // Salidas de activación por capa
    std::vector<Vector<T>> z_values;    // Valores intermedios (z = wx + b)
    T learning_rate;                    // Tasa de aprendizaje

    // Métodos auxiliares

    /**
     * Realiza la propagación hacia adelante.
     * @param input Entrada de la red.
     * @return Salida de la red después de la última capa.
     */
    Vector<T> forward_propagation(const Vector<T>& input) {
        Vector<T> output = input;
        activations.clear();
        z_values.clear();

        for (size_t i = 0; i < weights.size(); ++i) {
            // Calcular z = w * x + b
            Vector<T> z(weights[i].size(), 0.0);
            for (size_t j = 0; j < weights[i].size(); ++j) {
                z[j] = dot_product(weights[i][j], output) + biases[i][j];
            }

            z_values.push_back(z);

            // Aplicar función de activación (ReLU excepto en la última capa, que usa softmax)
            if (i == weights.size() - 1) {
                output = softmax(z); // Última capa (softmax)
            } else {
                output = apply_function(z, [](T x) { return std::max(static_cast<T>(0), x); }); // ReLU
            }

            activations.push_back(output);
        }

        return output;
    }

    /**
     * Realiza la retropropagación para ajustar los pesos y sesgos.
     * @param input Entrada original.
     * @param target Salida esperada (etiqueta codificada como un vector one-hot).
     */
    void backward_propagation(const Vector<T>& input, const Vector<T>& target) {
        // Gradiente de la última capa (diferencia entre salida y objetivo)
        Vector<T> delta = activations.back();
        for (size_t i = 0; i < delta.size(); ++i) {
            delta[i] -= target[i];
        }

        // Propagar hacia atrás
        for (int layer = weights.size() - 1; layer >= 0; --layer) {
            // Actualizar pesos y sesgos
            for (size_t i = 0; i < weights[layer].size(); ++i) {
                for (size_t j = 0; j < weights[layer][i].size(); ++j) {
                    weights[layer][i][j] -= learning_rate * delta[i] * (layer == 0 ? input[j] : activations[layer - 1][j]);
                }
                biases[layer][i] -= learning_rate * delta[i];
            }

            // Calcular delta para la capa anterior
            if (layer > 0) {
                Vector<T> new_delta(weights[layer][0].size(), 0.0);
                for (size_t j = 0; j < weights[layer][0].size(); ++j) {
                    for (size_t i = 0; i < weights[layer].size(); ++i) {
                        new_delta[j] += delta[i] * weights[layer][i][j];
                    }
                    new_delta[j] *= (z_values[layer - 1][j] > 0 ? 1 : 0); // Derivada de ReLU
                }
                delta = new_delta;
            }
        }
    }

public:
    /**
     * Constructor de la red neuronal.
     * @param architecture Vector que define el número de neuronas en cada capa.
     * @param learning_rate Tasa de aprendizaje.
     */
    NeuralNetwork(const std::vector<int>& architecture, T learning_rate) : learning_rate(learning_rate) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dis(-0.5, 0.5);

        for (size_t i = 1; i < architecture.size(); ++i) {
            weights.emplace_back(Matrix<T>(architecture[i], Vector<T>(architecture[i - 1])));
            biases.emplace_back(Vector<T>(architecture[i], 0.0));
            for (auto& row : weights.back()) {
                for (auto& weight : row) {
                    weight = dis(gen); // Inicializar pesos aleatorios
                }
            }
        }
    }

    /**
     * Entrena la red neuronal con el dataset proporcionado.
     * @param inputs Entradas de entrenamiento.
     * @param labels Etiquetas (en formato one-hot).
     * @param epochs Número de épocas de entrenamiento.
     */
    void train(const std::vector<Vector<T>>& inputs, const std::vector<Vector<T>>& labels, int epochs) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            T total_loss = 0.0;
            for (size_t i = 0; i < inputs.size(); ++i) {
                Vector<T> output = forward_propagation(inputs[i]);
                backward_propagation(inputs[i], labels[i]);

                // Calcular pérdida (Cross-Entropy Loss)
                for (size_t j = 0; j < labels[i].size(); ++j) {
                    total_loss -= labels[i][j] * std::log(output[j] + EPSILON);
                }
            }
            std::cout << "Época " << epoch + 1 << ": Pérdida = " << total_loss / inputs.size() << std::endl;
        }
    }

    /**
     * Evalúa la red neuronal en un conjunto de prueba.
     * @param inputs Entradas de prueba.
     * @param labels Etiquetas correspondientes.
     * @return Precisión de la red en el conjunto de prueba.
     */
    double evaluate(const std::vector<Vector<T>>& inputs, const std::vector<int>& labels) {
        int correct = 0;
        for (size_t i = 0; i < inputs.size(); ++i) {
            int predicted = predict(inputs[i]);
            if (predicted == labels[i]) {
                ++correct;
            }
        }
        return static_cast<double>(correct) / inputs.size() * 100.0;
    }

    /**
     * Predice la etiqueta de una entrada.
     * @param input Entrada de la red.
     * @return Etiqueta predicha.
     */
    int predict(const Vector<T>& input) {
        Vector<T> output = forward_propagation(input);
        return std::distance(output.begin(), std::max_element(output.begin(), output.end()));
    }
};

#endif // NETWORK_H
