from enum import Enum


class Inputs(Enum):

    BASIC_PATH = "datos/"
    OUTPUT_PATH = "aux_model/"
    DATA = "HousesInfo.txt"
    TRAIN_FEATURES = "X_train.csv"
    TRAIN_TARGET = "y_train.csv"
    TEST_FEATURES = "X_test.csv"
    TEST_TARGET = "y_test.csv"
    SEP = ' '


class Constants(Enum):

    TEST_SIZE = 0.15
    RANDOM_SEED = 123
    EPOCA = 15
    LOTE = 16
    MUESTRA_VALIDACION = 0.1
    VERBOSE = 1

    NORM_IMAGEN = 255

    CANAL = 3
    DIM = 128


class NeuralNet(Enum):

    NOMBRE_MODELO = "modelo_cuantitativo"
    
    NEURONAS = 1
    FUNCION_ACTIVACION = "relu"

    LEARNING_RATE = 0.001
    FUNCION_ERROR = "mse"
    METRICA = "mae"
    REGULARIZACION = 0.003


class Conv2D(Enum):

    NOMBRE_MODELO = "modelo_imagenes"

    FILTROS_1 = 128
    FILTROS_2 = 64
    FILTROS_3 = 32
    
    DENSA_1 = 256
    DENSA_2 = 128
    DENSA_FIN = 1
    
    TAMANIO_KERNEL = 3

    MAXPOOL = (2, 2)

    DROPOUT = 0.05
    FUNCION_ACTIVACION = "relu"
    REGULARIZACION = 0.1
    LEARNING_RATE = 0.001
    FUNCION_ERROR = "mse"
    METRICA = "mae"


class NNInputs2(Enum):

    NOMBRE_MODELO = "modelo_cuantitativo_imagenes"

    DENSA_CUANT = 10
    REGULARIZACION_CUANT = 0.01

    FILTROS_1 = 128
    FILTROS_2 = 64
    FILTROS_3 = 32
    
    DENSA_1 = 64
    DENSA_2 = 32
    DENSA_FIN = 1
    
    TAMANIO_KERNEL = 3

    MAXPOOL = (2, 2)

    DROPOUT = 0.05
    FUNCION_ACTIVACION = "relu"
    REGULARIZACION = 0.1
    LEARNING_RATE = 0.03
    FUNCION_ERROR = "mse"
    METRICA = "mae"


class NNInputs5(Enum):

    NOMBRE_MODELO = "modelo_cuantitativo_imagenes_all"

    DENSA_CUANT = 10
    REGULARIZACION_CUANT = 0.01

    FILTROS_0 = 16
    FILTROS_1 = 32
    FILTROS_2 = 64
    FILTROS_3 = 128
    FILTROS_4 = 8
    FILTROS_5 = 256
    
    DENSA_1 = 64
    DENSA_2 = 32
    DENSA_3 = 128
    DENSA_FIN = 1
    
    TAMANIO_KERNEL = 3

    MAXPOOL = (2, 2)

    REGULARIZACION_1 = 0.15
    REGULARIZACION_2 = 0.03
    REGULARIZACION_3 = 0.01

    DROPOUT = 0.2
    FUNCION_ACTIVACION = "relu"
    LEARNING_RATE = 0.03
    FUNCION_ERROR = "mse"
    METRICA = "mae"
