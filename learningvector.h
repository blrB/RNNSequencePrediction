#ifndef LEARNINGVECTOR
#define LEARNINGVECTOR

#include <armadillo>
#include <vector>

using namespace std;
using namespace arma;

class LearningVector
{
private:
    double answer;
    vector<double> numbers;
    mat X;
public:
    void addElement(double newElement);
    vector<double> getNumbers();
    void setNumbers(vector<double> numbers);
    int size();
    double getContextNeuron();
    void setContextNeuron(double neuron);
    mat getX();
    void setX(mat x);
    double getAnswer();
    void setAnswer(double answer);
    void createMatrixX();
};

#endif // LEARNINGVECTOR

