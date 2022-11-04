
/*
Alumna:Briones Magallon Irene Ameyalli
Grupo: 5BM1
Materia: Vision Artificial
*/

////////////////////////////////Cabeceras/////////////////////////////////////
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
/////////////////////////////////////////////////////////////////////////////
using namespace cv;
using namespace std;
/////////////////////////////////////////////////////////////////////////////

//Función de la operación matematica del Filtro de Gauss

float** windFil(int m, int n, float s)
{
    float** ventana = new float* [m];
    for (int i = 0; i < m; i++)
        ventana[i] = new float[n];
    int a = (m - 1) / 2; 
    int b = (n - 1) / 2;
    float x = -a;
    float y = b;
    float pi = 3.1416;

    for (int i = 0; i < m; i++) {
        x = -a;
        for (int j = 0; j < n; j++) {
            ventana[i][j] = (float)exp(-(x * x + y * y) / (2 * s * s)) / (2 * pi * s * s);
            x++;
        }
        y--;
    }
    return ventana;
}

//Función para copiar la imagen en Escala de Grises

void copiaImagenEscalaGrises(Mat imagen, int filaInicio, int columnaInicio, Mat imagenACopiar)
{
    for (int i = 0; i < imagenACopiar.rows; i++) // Para cada fila de la imagen a copiar
    {
        for (int j = 0; j < imagenACopiar.cols; j++) // Para cada columna de la imagen a copiar
        {
            // Copia la imagen
            imagen.at<uchar>(Point(columnaInicio + j, filaInicio + i)) = imagenACopiar.at<uchar>(Point(j, i));
        }
    }
}

//Funcion que rellena secciones en la Imagen en Escala de Grises

void rellenaSeccionImagenEscalaGrises(Mat imagen, int filaInicio, int filaFin, int columnaInicio, int columnaFin, uint valor)
{
    for (int i = filaInicio; i < filaFin; i++) // Para cada fila de la seccion
    {
        for (int j = columnaInicio; j < columnaFin; j++) // Para cada columna de la seccion
        {
            // Relllena la imagen
            imagen.at<uchar>(Point(j, i)) = valor;
        }
    }
}

//Función para agregar Bordes a la imagen

Mat agregaBordesVentanaFiltroEscalaGrises(Mat imagen, int filasBorde, int columnasBorde)
{
    // Variables de la imagen
    int filasImagen = imagen.rows;
    int columnasImagen = imagen.cols;
    // Variables de la imagen con bordes
    int filasImagenBordes = filasBorde + filasImagen + filasBorde;
    int columnasImagenBordes = columnasBorde + columnasImagen + columnasBorde;
    Mat imagenBordes(filasImagenBordes, columnasImagenBordes, CV_8UC1);

    // Copia la imagen
    copiaImagenEscalaGrises(imagenBordes, filasBorde, columnasBorde, imagen);
    // Rellena el borde superior
    rellenaSeccionImagenEscalaGrises(imagenBordes, 0, filasBorde, 0, columnasImagenBordes, 0);
    // Rellena el borde inferior
    rellenaSeccionImagenEscalaGrises(imagenBordes, filasBorde + filasImagen, filasImagenBordes, 0, columnasImagenBordes, 0);
    // Rellena el borde izquierdo
    rellenaSeccionImagenEscalaGrises(imagenBordes, 0, filasImagenBordes, 0, columnasBorde, 0);
    // Rellena el borde derecho
    rellenaSeccionImagenEscalaGrises(imagenBordes, 0, filasImagenBordes, columnasBorde + columnasImagen, columnasImagenBordes, 0);
    return imagenBordes;
}

//Función de la redimención de la Imagen

Mat redi(int m, int n, Mat img)
{
    int a = (m - 1) / 2;
    int b = (n - 1) / 2;
    int f = img.rows; // Lectura de cuantas filas tiene la imagen
    int c = img.cols;
    Mat img2(f + m - 1, c + n - 1, CV_8UC1);

    for (int i = 0; i < f; i++)
    {
        for (int j = 0; j < c; j++)
        {
            img2.at<uchar>(Point(j + b, i + a)) = img.at<uchar>(Point(j, i));
        }
    }

    // Parte Superior de la redimención de la Imagen
    for (int i = 0; i < a; i++) {
        for (int j = 0; j < c + n - 1; j++) {
            img2.at<uchar>(Point(j, i)) = 0;
        }
    }
    // Parte Inferior de la redimención de la Imagen
    for (int i = f + a; i < f + m - 1; i++) {
        for (int j = 0; j < c + n - 1; j++) {
            img2.at<uchar>(Point(j, i)) = 0;
        }
    }
    // Parte Izquierda de la redimención de la Imagen
    for (int i = 0; i < f + m - 1; i++) {
        for (int j = 0; j < b; j++) {
            img2.at<uchar>(Point(j, i)) = 0;
        }
    }
    // Parte Derecha de la redimención de la Imagen
    for (int i = 0; i < f + m - 1; i++) {
        for (int j = c + b; j < c + n - 1; j++) {
            img2.at<uchar>(Point(j, i)) = 0;
        }
    }
    return img2;


}


//Funcion de la aplicación del filtro Gaussiano a la Imagen

Mat aplicaFiltro(float** filtro, int m, int n, Mat imagen)
{
    int filas = imagen.rows - m + 1;
    int columnas = imagen.cols - n + 1;
    Mat imagenFiltro(filas, columnas, CV_8UC1);

    for (int i = 0; i < filas; i++)
    {
        for (int j = 0; j < columnas; j++)
        {
            float convolucion = 0;
            for (int k = 0; k < m; k++)
            {
                for (int l = 0; l < n; l++)
                {
                    convolucion += filtro[k][l] * imagen.at<uchar>(Point(j + l, i + k));
                }
            }
            imagenFiltro.at<uchar>(Point(j, i)) = convolucion;
        }
    }
    return imagenFiltro;
}

/////////////////////////Inicio de la funcion principal///////////////////
int main()
{
    /********Declaracion de variables generales*********/
    char NombreImagen[] = "2.png";
    Mat imagen; // Matriz que contiene nuestra imagen sin importar el formato
    /************************/
    Mat imagenGrisFunc;
    /*********Lectura de la imagen*********/
    imagen = imread(NombreImagen, IMREAD_UNCHANGED);

    if (!imagen.data)
    {
        cout << "Error al cargar la imagen: " << NombreImagen << endl;
        exit(1);
    }
    /************************/

    float** ventana;
    int m, n;
    float s;

    //Ingreso de datos necesarios para la aplicación de filtros 

    cout << "\n\nIngresa el valor de m (Filas)\n\n";
    cin >> m;

    cout << "\n\nIngresa el valor de n (Columnas)\n\n";
    cin >> n;

    cout << "\n\nIngresa el valor de Sigma\n\n";
    cin >> s;

    ventana = windFil(m, n, s); // Se llama a la función de la parte matematica del Filtro de Gauss

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            cout << (ventana[i])[j] << ' ';
        }
        cout << '\n';
    }

    Mat im = agregaBordesVentanaFiltroEscalaGrises(imagen, (m - 1) / 2, (n - 1) / 2); //Imagen ya en forma de Matriz para que se agreguen los bordes

    //Se mostrará la imagen Original

    namedWindow("Imagen Original", WINDOW_AUTOSIZE);
    imshow("Imagen Original", imagen);

    //Valores que se muestran en consola

    cout << "\n\nImagen original - Dimensiones\n";
    cout << imagen.rows << '\n';
    cout << imagen.cols << '\n';

    //Se mostrará la imagen con bordes
    namedWindow("Imagen con bordes", WINDOW_AUTOSIZE);
    imshow("Imagen con bordes", im);

    //Valores que se muestran en consola

    cout << "\nImagen con bordes - Dimensiones\n";
    cout << im.rows << '\n';
    cout << im.cols << '\n';
    Mat gauss = aplicaFiltro(ventana, m, n, im);

    //Se mostrará la imagen con la Aplicación del Filtro de Gauss

    namedWindow("Filtro Gauss", WINDOW_AUTOSIZE);
    imshow("Filtro Gauss", gauss);

    //Valores que se muestran en consola

    cout << "\nImagen con Filtro Gaussiano - Dimensiones\n";
    cout << gauss.rows << '\n';
    cout << gauss.cols << '\n';

    //imwrite("2.png", imagenGrisesNTSC);

    waitKey(0);
    return 1;
}