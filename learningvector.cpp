#include "learningvector.h"
#include <armadillo>
#include <vector>

using namespace std;
using namespace arma;

void LearningVector::addElement(double newElement){
    numbers.push_back(newElement);
}

vector<double> LearningVector::getNumbers(){
    return numbers;
}

void LearningVector::setNumbers(vector<double> numbers){
    this->numbers = numbers;
}

int LearningVector::size(){
    return numbers.size();
}

double LearningVector::getContextNeuron(){
    return X(0, numbers.size());
}

void LearningVector::setContextNeuron(double neuron){
    X(0, numbers.size()) = neuron;
}

mat LearningVector::getX(){
    return X;
}

void LearningVector::setX(mat X){
    this->X = X;
}

double LearningVector::getAnswer(){
    return answer;
}

void LearningVector::setAnswer(double answer){
    this->answer = answer;
}

void LearningVector::createMatrixX(){
    vector<double> matrix(numbers);
    matrix.push_back(0);
    X = mat(matrix).t();
}
