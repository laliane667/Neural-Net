#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>

using namespace std;

class TrainingData
{
public:
    TrainingData(const string filename);
    bool isEof(void){ return m_trainingDataFile.eof();}
    void getTopology(vector<double> &topology);

    double getNextInputs(vector<double> &inputVals);
    double getTargetOutputs(vector<double> &targetOutputVals);

private:

    ifstream m_trainingDataFile;

};

void TrainingData::getTopology(vector<double> &topology)
{
    string line, label;

    getline( m_trainingDataFile, line);
    stringstream ss(line);
    ss >> label;

    if(this->isEof() || label.compare("topology:") != 0)
        abort();
    while(!ss.eof())
    {
        double n;
        ss >> n;
        topology.push_back(n);
    }

    return;
}

TrainingData::TrainingData(const string filename)
{
    m_trainingDataFile.open(filename.c_str());
}

double TrainingData::getNextInputs(vector <double> &inputVals)
{
    inputVals.clear();
    string line;

    getline(m_trainingDataFile, line);
    stringstream ss(line);

    string label;
    ss >> label;

    if(label.compare("in:") == 0)
    {
        double oneValue;
        while(ss >> oneValue)
            inputVals.push_back(oneValue);
    }

    return inputVals.size();
}

double TrainingData::getTargetOutputs(vector <double> &targetOutputVals)
{
    targetOutputVals.clear();
    string line;

    getline(m_trainingDataFile, line);
    stringstream ss(line);

    string label;
    ss >> label;

    if(label.compare("out:") == 0)
    {
        double oneValue;
        while(ss >> oneValue)
            targetOutputVals.push_back(oneValue);
    }

    return targetOutputVals.size();
}

struct Connection
{
    double weight;
    double deltaWeight;
};


class Neuron;

typedef vector<Neuron> Layer;

class Neuron
{
public:
    Neuron(double numOutputs, double myIndex);
    void setOutputVal(double val) { m_outputVal = val; }
    double getOutputVal(void) const { return m_outputVal; }
    void feedForward(const Layer &prevLayer);
    void calcOutputGradients(double targetVal);
    void calcHiddenGradients(const Layer &nextLayer);
    void updateInputWeights(Layer &prevLayer);

private:
    static double eta;
    static double alpha;
    static double transferFunction(double x);
    static double transferFunctionDerivative(double x);
    static double randomWeight(void) { return rand() / double(RAND_MAX); }
    double sumDOW(const Layer &nextLayer) const;
    double m_outputVal;
    vector<Connection> m_outputWeights;
    double m_myIndex;
    double m_gradient;
};

double Neuron::eta = 0.15;
double Neuron::alpha = 0.5;


void Neuron::updateInputWeights(Layer &prevLayer)
{

    for (double n = 0; n < prevLayer.size(); ++n) {
        Neuron &neuron = prevLayer[n];
        double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

        double newDeltaWeight =
                // Individual input, magnified by the gradient and train rate:
                eta
                * neuron.getOutputVal()
                * m_gradient
                // Also add momentum = a fraction of the previous delta weight;
                + alpha
                * oldDeltaWeight;

        neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
        neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
    }
}

double Neuron::sumDOW(const Layer &nextLayer) const
{
    double sum = 0.0;


    for (double n = 0; n < nextLayer.size() - 1; ++n) {
        sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
    }

    return sum;
}

void Neuron::calcHiddenGradients(const Layer &nextLayer)
{
    double dow = sumDOW(nextLayer);
    m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
}

void Neuron::calcOutputGradients(double targetVal)
{
    double delta = targetVal - m_outputVal;
    m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
}

double Neuron::transferFunction(double x)
{

    return tanh(x);
}

double Neuron::transferFunctionDerivative(double x)
{
    return 1.0 - x * x;
}

void Neuron::feedForward(const Layer &prevLayer)
{
    double sum = 0.0;

    for (double n = 0; n < prevLayer.size(); ++n) {
        sum += prevLayer[n].getOutputVal() *
                prevLayer[n].m_outputWeights[m_myIndex].weight;
    }

    m_outputVal = Neuron::transferFunction(sum);
}

Neuron::Neuron(double numOutputs, double myIndex)
{
    for (double c = 0; c < numOutputs; ++c) {
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();
    }

    m_myIndex = myIndex;
}


class Net
{
public:
    Net(const vector<double> &topology);
    void feedForward(const vector<double> &inputVals);
    void backProp(const vector<double> &targetVals);
    void getResults(vector<double> &resultVals) const;
    double getRecentAverageError(void) const { return m_recentAverageError; }

private:
    vector<Layer> m_layers; // m_layers[layerNum][neuronNum]
    double m_error;
    double m_recentAverageError;
    static double m_recentAverageSmoothingFactor;
};


double Net::m_recentAverageSmoothingFactor = 100.0;


void Net::getResults(vector<double> &resultVals) const
{
    resultVals.clear();

    for (double n = 0; n < m_layers.back().size() - 1; ++n) {
        resultVals.push_back(m_layers.back()[n].getOutputVal());
    }
}

void Net::backProp(const vector<double> &targetVals)
{

    Layer &outputLayer = m_layers.back();
    m_error = 0.0;

    for (double n = 0; n < outputLayer.size() - 1; ++n) {
        double delta = targetVals[n] - outputLayer[n].getOutputVal();
        m_error += delta * delta;
    }
    m_error /= outputLayer.size() - 1; // get average error squared
    m_error = sqrt(m_error); // RMS

    m_recentAverageError =
            (m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
            / (m_recentAverageSmoothingFactor + 1.0);

    for (double n = 0; n < outputLayer.size() - 1; ++n) {
        outputLayer[n].calcOutputGradients(targetVals[n]);
    }

    for (double layerNum = m_layers.size() - 2; layerNum > 0; --layerNum) {
        Layer &hiddenLayer = m_layers[layerNum];
        Layer &nextLayer = m_layers[layerNum + 1];

        for (double n = 0; n < hiddenLayer.size(); ++n) {
            hiddenLayer[n].calcHiddenGradients(nextLayer);
        }
    }

    for (double layerNum = m_layers.size() - 1; layerNum > 0; --layerNum) {
        Layer &layer = m_layers[layerNum];
        Layer &prevLayer = m_layers[layerNum - 1];

        for (double n = 0; n < layer.size() - 1; ++n) {
            layer[n].updateInputWeights(prevLayer);
        }
    }
}

void Net::feedForward(const vector<double> &inputVals)
{
    assert(inputVals.size() == m_layers[0].size() - 1);

    for (double i = 0; i < inputVals.size(); ++i) {
        m_layers[0][i].setOutputVal(inputVals[i]);
    }

    for (double layerNum = 1; layerNum < m_layers.size(); ++layerNum) {
        Layer &prevLayer = m_layers[layerNum - 1];
        for (double n = 0; n < m_layers[layerNum].size() - 1; ++n) {
            m_layers[layerNum][n].feedForward(prevLayer);
        }
    }
}

Net::Net(const vector<double> &topology)
{
    double numLayers = topology.size();
    for (double layerNum = 0; layerNum < numLayers; ++layerNum) {
        m_layers.push_back(Layer());
        double numOutputs = layerNum == (topology.size() - 1) ? 0 : topology[layerNum + 1];

        for (double neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) {
            m_layers.back().push_back(Neuron(numOutputs, neuronNum));
            cout << "Made a Neuron!" << endl;
        }

        m_layers.back().back().setOutputVal(1.0);
    }
}


void showVectorVals(string label, vector<double> &v)
{
    cout << label << " ";
    for (double i = 0; i < v.size(); ++i) {
        cout << v[i] << " ";
    }

    cout << endl;
}

int main()
{
    TrainingData trainData("training.txt");

    vector<double> topology;
    trainData.getTopology(topology);

    Net myNet(topology);

    vector<double> inputVals, targetVals, resultVals;
    int trainingPass = 0;

    while (!trainData.isEof()) {

        ++trainingPass;
        cout << endl << "Pass" << trainingPass;

        if (trainData.getNextInputs(inputVals)!= topology[0]) {
            break;
        }

        showVectorVals(": Inputs: ", inputVals);
        myNet.feedForward(inputVals);

        myNet.getResults(resultVals);

        showVectorVals("Outputs: ",resultVals);

        trainData.getTargetOutputs(targetVals);
        showVectorVals("Targets: ", targetVals);

        assert(targetVals.size() == topology.back());

        myNet.backProp(targetVals);

        cout << "Net recent average error: " <<  myNet.getRecentAverageError() << endl;
    }
    cout << endl << "Done" << endl;

}
