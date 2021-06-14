#include <complex>
#include <vector>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
using namespace std;

// Fourier transform (DFT)
vector<complex<double>> dft(vector<complex<double>> data)
{
    int numberOfData = data.size();
    int K = numberOfData;
    complex<double> frequency;
    vector<complex<double>> frequencies;
    frequencies.reserve(K);

    for (int k = 0; k < K; k++)
    {
        frequency = (0, 0);
        for (int n = 0; n < numberOfData; n++)
        {
            double realPart = cos(((2 * M_PI) / numberOfData) * k * n);
            double imagPart = sin(((2 * M_PI) / numberOfData) * k * n);
            complex<double> w(realPart, -imagPart);
            frequency += data[n] * w;
        }
        frequencies.push_back(frequency);
    }
    return frequencies;
}

// Fast fourier transform (FFT)
vector<complex<double>> fft(vector<complex<double>> &data)
{
    int numberOfData = data.size();
    if (numberOfData == 1) {return data;}    

    /* Split the sampels into even and odd subsums */
    int halfOfData = numberOfData/2;
    vector<complex<double>> evenSamples(halfOfData,0);
    vector<complex<double>> oddSamples(halfOfData,0);

    for (int i = 0; i != halfOfData; i++)
    {
        evenSamples[i] = data[2*i];
        oddSamples[i] = data[2*i + 1];
    }
    // Perform the recursive FFT operation on the odd and even sides
    vector<complex<double>> evenFrequencies(halfOfData,0);
    evenFrequencies = fft(evenSamples);
    vector<complex<double>> oddFrequencies(halfOfData,0);
    oddFrequencies = fft(oddSamples);
    /*------ END RECURSION ______ */

    // Combine the values found
    vector<complex<double>> frequencies(numberOfData,0);
    for (int k = 0; k != numberOfData/2; k++)
    {
        complex<double> complexExponential = polar(1.0, -2*M_PI*k/numberOfData) * oddFrequencies[k];
        frequencies[k] = evenFrequencies[k] + complexExponential;
        frequencies[k+numberOfData/2] = evenFrequencies[k] - complexExponential;
    }
    return frequencies;
}

// Connection between python and c++
PYBIND11_MODULE(module_name ,handle){
    handle.def("dft", &dft, "A function returns dft");
    handle.def("fft", &fft, "A function returns fft");
}
