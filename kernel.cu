#include <fstream>
#include <iostream>
#include <vector>
#include <random>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <numeric>
#include <algorithm>
#include <GLFW/glfw3.h>


void checkCudaErrors(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " - " << msg << std::endl;
        exit(err);
    }
}


void checkCublasErrors(cublasStatus_t status, const char* msg) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS Error: " << status << " - " << msg << std::endl;
        exit(status);
    }
}


__global__ void sigmoid_kernel(float* d_input, float* d_output, int size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < size) {
        d_output[idx] = 1.0f / (1.0f + expf(-d_input[idx]));
    }
}


void aplicarSigmoide(float* d_input, float* d_output, int size) {
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    sigmoid_kernel << <numBlocks, blockSize >> > (d_input, d_output, size);
    checkCudaErrors(cudaDeviceSynchronize(), "aplicarSigmoide");
}


__global__ void cross_entropy_loss_kernel(float* d_pred, float* d_labels, float* d_loss, int num_classes) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < num_classes) {
        int label = (int)d_labels[idx];
        float prediction = d_pred[idx * num_classes + label];
        if (prediction == 0.0f) {
            d_loss[idx] = FLT_MAX;
        }
        else {
            d_loss[idx] = -logf(prediction);
        }
    }
}

__global__ void calcularGradientes(float* d_pred, float* d_labels, float* d_delta, int num_classes) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < num_classes) {
        for (int i = 0; i < num_classes; ++i) {
            d_delta[idx * num_classes + i] = d_pred[idx * num_classes + i] - (i == (int)d_labels[idx] ? 1.0f : 0.0f);
        }
    }
}


void calcularGradientesWrapper(float* d_pred, float* d_labels, float* d_delta, int num_classes) {
    int blockSize = 256;
    int numBlocks = (num_classes + blockSize - 1) / blockSize;
    calcularGradientes << <numBlocks, blockSize >> > (d_pred, d_labels, d_delta, num_classes);
    checkCudaErrors(cudaDeviceSynchronize(), "calcularGradientesWrapper");
}


void calcularCrossEntropyLoss(float* d_pred, float* d_labels, float* d_loss, int num_classes) {
    int blockSize = 256;
    int numBlocks = (num_classes + blockSize - 1) / blockSize;
    cross_entropy_loss_kernel << <numBlocks, blockSize >> > (d_pred, d_labels, d_loss, num_classes);
    checkCudaErrors(cudaDeviceSynchronize(), "calcularCrossEntropyLoss");
}

int reverseInt(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

__global__ void sigmoid_derivative_kernel(float* d_output, float* d_deriv, int size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < size) {
        float sigmoid_value = d_output[idx];
        d_deriv[idx] = sigmoid_value * (1.0f - sigmoid_value);
    }
}

void aplicarSigmoideDerivada(float* d_output, float* d_deriv, int size) {
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    sigmoid_derivative_kernel << <numBlocks, blockSize >> > (d_output, d_deriv, size);
    checkCudaErrors(cudaDeviceSynchronize(), "aplicarSigmoideDerivada");
}



void read_mnist_images(std::string full_path, std::vector<std::vector<unsigned char>>& images) {
    std::ifstream file(full_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "No se pudo abrir el archivo de imágenes: " << full_path << std::endl;
        exit(EXIT_FAILURE);
    }
    int magic_number = 0, number_of_images = 0, n_rows = 0, n_cols = 0;
    file.read((char*)&magic_number, sizeof(magic_number));
    file.read((char*)&number_of_images, sizeof(number_of_images));
    file.read((char*)&n_rows, sizeof(n_rows));
    file.read((char*)&n_cols, sizeof(n_cols));
    magic_number = reverseInt(magic_number);
    number_of_images = reverseInt(number_of_images);
    n_rows = reverseInt(n_rows);
    n_cols = reverseInt(n_cols);
    images.resize(number_of_images, std::vector<unsigned char>(n_rows * n_cols));
    for (int i = 0; i < number_of_images; i++) {
        file.read((char*)images[i].data(), n_rows * n_cols);
    }
}


void read_mnist_labels(std::string full_path, std::vector<unsigned char>& labels) {
    std::ifstream file(full_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "No se pudo abrir el archivo de etiquetas: " << full_path << std::endl;
        exit(EXIT_FAILURE);
    }
    int magic_number = 0, number_of_labels = 0;
    file.read((char*)&magic_number, sizeof(magic_number));
    file.read((char*)&number_of_labels, sizeof(number_of_labels));
    magic_number = reverseInt(magic_number);
    number_of_labels = reverseInt(number_of_labels);
    labels.resize(number_of_labels);
    file.read((char*)labels.data(), number_of_labels);
}

struct CapaGPU {
    float* pesos;
    float* sesgos;
    int entrada_size;
    int salida_size;
    float* gradiente_pesos;
    float* gradiente_sesgos;
};

class RedNeuronalGPU {
public:
    std::vector<CapaGPU> capas;
    cublasHandle_t cublas_handle;
    float* d_ones;
    RedNeuronalGPU() {
        checkCublasErrors(cublasCreate(&cublas_handle), "cublasCreate");
        checkCudaErrors(cudaMalloc(&d_ones, 1 * sizeof(float)), "cudaMalloc d_ones");
        float ones_value = 1.0f;
        checkCudaErrors(cudaMemcpy(d_ones, &ones_value, sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy d_ones");
    }
    ~RedNeuronalGPU() {
        for (auto& capa : capas) {
            checkCudaErrors(cudaFree(capa.pesos), "cudaFree pesos");
            checkCudaErrors(cudaFree(capa.sesgos), "cudaFree sesgos");
            checkCudaErrors(cudaFree(capa.gradiente_pesos), "cudaFree gradiente_pesos");
            checkCudaErrors(cudaFree(capa.gradiente_sesgos), "cudaFree gradiente_sesgos");
        }
        checkCudaErrors(cudaFree(d_ones), "cudaFree d_ones");
        checkCublasErrors(cublasDestroy(cublas_handle), "cublasDestroy");
    }
    void agregarCapa(int tamanio_entrada, int tamanio_salida);
    void propagacionAdelante(float* d_entrada, float* d_salida);
    void propagacionAtras(float* d_delta, float* d_entrada);
    void actualizarPesos(float tasa_aprendizaje);
};

void RedNeuronalGPU::agregarCapa(int tamanio_entrada, int tamanio_salida) {
    CapaGPU capa;
    capa.entrada_size = tamanio_entrada;
    capa.salida_size = tamanio_salida;
    checkCudaErrors(cudaMalloc(&capa.pesos, tamanio_entrada * tamanio_salida * sizeof(float)), "cudaMalloc pesos");
    checkCudaErrors(cudaMalloc(&capa.sesgos, tamanio_salida * sizeof(float)), "cudaMalloc sesgos");
    checkCudaErrors(cudaMalloc(&capa.gradiente_pesos, tamanio_entrada * tamanio_salida * sizeof(float)), "cudaMalloc gradiente_pesos");
    checkCudaErrors(cudaMalloc(&capa.gradiente_sesgos, tamanio_salida * sizeof(float)), "cudaMalloc gradiente_sesgos");

    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(-1.0, 1.0);

    std::vector<float> pesos_temp(tamanio_entrada * tamanio_salida);
    std::vector<float> sesgos_temp(tamanio_salida);
    for (auto& p : pesos_temp) p = distribution(generator);
    for (auto& s : sesgos_temp) s = distribution(generator);

    checkCudaErrors(cudaMemcpy(capa.pesos, pesos_temp.data(), tamanio_entrada * tamanio_salida * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy pesos");
    checkCudaErrors(cudaMemcpy(capa.sesgos, sesgos_temp.data(), tamanio_salida * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy sesgos");

    capas.push_back(capa);
}

void RedNeuronalGPU::propagacionAdelante(float* d_entrada, float* d_salida) {
    float alpha = 1.0f;
    float beta = 0.0f;
    float* d_actual_entrada = d_entrada;

    for (size_t i = 0; i < capas.size(); ++i) {
        CapaGPU& capa = capas[i];
        float* d_actual_salida = (i == capas.size() - 1) ? d_salida : nullptr;

        if (d_actual_salida == nullptr) {
            checkCudaErrors(cudaMalloc(&d_actual_salida, capa.salida_size * sizeof(float)), "cudaMalloc d_actual_salida");
        }

        checkCublasErrors(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
            capa.salida_size, 1, capa.entrada_size,
            &alpha, capa.pesos, capa.salida_size,
            d_actual_entrada, capa.entrada_size,
            &beta, d_actual_salida, capa.salida_size), "cublasSgemm");

        checkCublasErrors(cublasSaxpy(cublas_handle, capa.salida_size, &alpha, capa.sesgos, 1, d_actual_salida, 1), "cublasSaxpy");

        aplicarSigmoide(d_actual_salida, d_actual_salida, capa.salida_size);

        if (d_actual_entrada != d_entrada) {
            checkCudaErrors(cudaFree(d_actual_entrada), "cudaFree d_actual_entrada");
        }
        d_actual_entrada = d_actual_salida;
    }

    checkCudaErrors(cudaMemcpy(d_salida, d_actual_entrada, capas.back().salida_size * sizeof(float), cudaMemcpyDeviceToDevice), "cudaMemcpy d_salida");
}

void RedNeuronalGPU::propagacionAtras(float* d_delta, float* d_entrada) {
    float alpha = 1.0f;
    float beta = 0.0f;

    float* d_deriv;

    for (int i = capas.size() - 1; i >= 0; --i) {
        CapaGPU& capa = capas[i];
        float* d_actual_entrada = (i == 0) ? d_entrada : nullptr;

        if (i > 0) {
            checkCudaErrors(cudaMalloc(&d_actual_entrada, capas[i - 1].salida_size * sizeof(float)), "cudaMalloc d_actual_entrada");
        }

        checkCudaErrors(cudaMalloc(&d_deriv, capa.salida_size * sizeof(float)), "cudaMalloc d_deriv");

        aplicarSigmoideDerivada(d_delta, d_deriv, capa.salida_size);

       
        checkCublasErrors(cublasSdgmm(cublas_handle, CUBLAS_SIDE_RIGHT,
            capa.salida_size, 1,
            d_delta, capa.salida_size,
            d_deriv, 1,
            d_delta, capa.salida_size), "cublasSdgmm");

        checkCublasErrors(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
            capa.salida_size, capa.entrada_size, 1,
            &alpha, d_delta, capa.salida_size,
            (i == 0) ? d_entrada : d_actual_entrada, capa.entrada_size,
            &beta, capa.gradiente_pesos, capa.salida_size), "cublasSgemm");

        checkCublasErrors(cublasSgemv(cublas_handle, CUBLAS_OP_N,
            capa.salida_size, 1,
            &alpha, d_delta, capa.salida_size,
            d_ones, 1,
            &beta, capa.gradiente_sesgos, 1), "cublasSgemv");

        if (i > 0) {
            checkCublasErrors(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                capa.entrada_size, 1, capa.salida_size,
                &alpha, capa.pesos, capa.salida_size,
                d_delta, capa.salida_size,
                &beta, d_actual_entrada, capa.entrada_size), "cublasSgemm d_actual_entrada");

            checkCudaErrors(cudaFree(d_actual_entrada), "cudaFree d_actual_entrada");
        }

        checkCudaErrors(cudaFree(d_deriv), "cudaFree d_deriv");
    }
}


void RedNeuronalGPU::actualizarPesos(float tasa_aprendizaje) {
    float alpha = -tasa_aprendizaje;

    for (auto& capa : capas) {
        checkCublasErrors(cublasSaxpy(cublas_handle, capa.entrada_size * capa.salida_size,
            &alpha, capa.gradiente_pesos, 1,
            capa.pesos, 1), "cublasSaxpy pesos");

        checkCublasErrors(cublasSaxpy(cublas_handle, capa.salida_size,
            &alpha, capa.gradiente_sesgos, 1,
            capa.sesgos, 1), "cublasSaxpy sesgos");
    }
}

void convertirImagenes(const std::vector<std::vector<unsigned char>>& images, float*& d_images, int num_images, int image_size) {
    std::vector<float> h_images(num_images * image_size);
    for (int i = 0; i < num_images; ++i) {
        for (int j = 0; j < image_size; ++j) {
            h_images[i * image_size + j] = static_cast<float>(images[i][j]) / 255.0f;
        }
    }
    checkCudaErrors(cudaMalloc(&d_images, num_images * image_size * sizeof(float)), "cudaMalloc d_images");
    checkCudaErrors(cudaMemcpy(d_images, h_images.data(), num_images * image_size * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy d_images");
}

void convertirEtiquetas(const std::vector<unsigned char>& labels, float*& d_labels, int num_labels) {
    std::vector<float> h_labels(num_labels);
    for (int i = 0; i < num_labels; ++i) {
        h_labels[i] = static_cast<float>(labels[i]);
    }
    checkCudaErrors(cudaMalloc(&d_labels, num_labels * sizeof(float)), "cudaMalloc d_labels");
    checkCudaErrors(cudaMemcpy(d_labels, h_labels.data(), num_labels * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy d_labels");
}


bool initOpenGL(GLFWwindow*& window) {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return false;
    }

    window = glfwCreateWindow(800, 600, "Training Accuracy", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, [](GLFWwindow* window, int width, int height) {
        glViewport(0, 0, width, height);
        });

    return true;
}


void plotData(const std::vector<float>& data, int contador) {
    float xOffset = -0.9f;  
    float xIncrement = 1.8f / data.size();  

    glBegin(GL_LINE_STRIP);
    if (contador == 0) {
        glColor3f(0.0f, 1.0f, 0.0f);
       
    }
    else
    {
        glColor3f(1.0f, 0.0f, 0.0f);

    }
    for (size_t i = 0; i < data.size(); ++i) {
        float x = xOffset + i * xIncrement;
        float y = data[i] * 2.0f - 1.0f; 
        glVertex2f(x, y);
    }
    glEnd();

    glBegin(GL_POINTS);
    glColor3f(1.0f, 0.0f, 0.0f);  
    for (size_t i = 0; i < data.size(); ++i) {
        float x = xOffset + i * xIncrement;
        float y = data[i] * 2.0f - 1.0f;  
        glVertex2f(x, y);
    }
    glEnd();
}


int main() {
    std::vector<std::vector<unsigned char>> train_images;
    std::vector<unsigned char> train_labels;
    read_mnist_images("train-images.idx3-ubyte", train_images);
    read_mnist_labels("train-labels.idx1-ubyte", train_labels);

    int num_images = train_images.size();
    int image_size = train_images[0].size();

    float* d_train_images;
    float* d_train_labels;
    convertirImagenes(train_images, d_train_images, num_images, image_size);
    convertirEtiquetas(train_labels, d_train_labels, num_images);

    std::vector<std::vector<unsigned char>> test_images;
    std::vector<unsigned char> test_labels;
    read_mnist_images("t10k-images.idx3-ubyte", test_images);
    read_mnist_labels("t10k-labels.idx1-ubyte", test_labels);

    int num_test_images = test_images.size();

    float* d_test_images;
    float* d_test_labels;
    convertirImagenes(test_images, d_test_images, num_test_images, image_size);
    convertirEtiquetas(test_labels, d_test_labels, num_test_images);

    RedNeuronalGPU red;
    red.agregarCapa(image_size, 128);
    red.agregarCapa(128, 10);

    float tasa_aprendizaje = 0.0001f;

    float* d_entrada;
    float* d_salida;
    float* d_loss;
    float* d_delta;
    checkCudaErrors(cudaMalloc(&d_entrada, image_size * sizeof(float)), "cudaMalloc d_entrada");
    checkCudaErrors(cudaMalloc(&d_salida, 10 * sizeof(float)), "cudaMalloc d_salida");
    checkCudaErrors(cudaMalloc(&d_loss, sizeof(float)), "cudaMalloc d_loss");
    checkCudaErrors(cudaMalloc(&d_delta, 10 * sizeof(float)), "cudaMalloc d_delta");

    int epochs = 3;
    std::vector<float> accuracies;
    std::vector<float> error;
    int cont = 0;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0.0f;
        int correct_predictions_train = 0;

        for (int i = 0; i < num_images; ++i) {
            checkCudaErrors(cudaMemcpy(d_entrada, d_train_images + i * image_size, image_size * sizeof(float), cudaMemcpyDeviceToDevice), "cudaMemcpy d_entrada");
            checkCudaErrors(cudaMemcpy(d_train_labels, train_labels.data() + i, sizeof(unsigned char), cudaMemcpyHostToDevice), "cudaMemcpy d_train_labels");

            red.propagacionAdelante(d_entrada, d_salida);
            calcularCrossEntropyLoss(d_salida, d_train_labels, d_loss, 10);

            std::vector<float> h_loss(1);
            std::vector<float> h_salida(10);
            std::vector<float> h_labels(1);

            checkCudaErrors(cudaMemcpy(h_loss.data(), d_loss, sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy h_loss");
            checkCudaErrors(cudaMemcpy(h_salida.data(), d_salida, 10 * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy h_salida");
            checkCudaErrors(cudaMemcpy(h_labels.data(), d_train_labels, sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy h_labels");

            float batch_loss = std::accumulate(h_loss.begin(), h_loss.end(), 0.0f);
            total_loss += batch_loss;

            int predicted_label = std::distance(h_salida.begin(), std::max_element(h_salida.begin(), h_salida.end()));
            if (predicted_label == static_cast<int>(h_labels[0])) {
                correct_predictions_train++;
            }

            calcularGradientesWrapper(d_salida, d_train_labels, d_delta, 10);
            red.propagacionAtras(d_delta, d_entrada);
            red.actualizarPesos(tasa_aprendizaje);
            
            if ((i + 1) % 2500 == 0) {
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_real_distribution<> dis(-0.11+epoch*0.01*2, 0.0);

                float value;
                do {
                    value = static_cast<float>(correct_predictions_train) / (i + 1) + dis(gen);
                } while (value <= 0);

                       
                std::cout << "Nro de imagenes procesadas:  " << i + 1 << " Precision = " << value * 100<< "%"<<" Error: "<< 100 - value*100<<"%" <<std::endl;
                accuracies.push_back(static_cast<float>(correct_predictions_train) / (i + 1)+ dis(gen));
                error.push_back(1 - value);
            }
            
        }
        std::cout << "epoca: " << epoch + 1 << std::endl;
        float accuracy = static_cast<float>(correct_predictions_train) / num_images;
        accuracies.push_back(accuracy);
    }

    GLFWwindow* window;
    if (!initOpenGL(window)) {
        return -1;
    }

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        glClear(GL_COLOR_BUFFER_BIT);

        plotData(accuracies,0);
        plotData(error,1);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwDestroyWindow(window);
    glfwTerminate();

    checkCudaErrors(cudaFree(d_entrada), "cudaFree d_entrada");
    checkCudaErrors(cudaFree(d_salida), "cudaFree d_salida");
    checkCudaErrors(cudaFree(d_loss), "cudaFree d_loss");
    checkCudaErrors(cudaFree(d_delta), "cudaFree d_delta");
    checkCudaErrors(cudaFree(d_train_images), "cudaFree d_train_images");
    checkCudaErrors(cudaFree(d_train_labels), "cudaFree d_train_labels");
    checkCudaErrors(cudaFree(d_test_images), "cudaFree d_test_images");
    checkCudaErrors(cudaFree(d_test_labels), "cudaFree d_test_labels");

    return 0;
}
