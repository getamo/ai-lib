#ifndef AINET_H
#define AINET_H

#include <iostream>
#include <functional>
#include <map>
#include <vector>
#include <unordered_set>
#include <string>
#include <cmath>
#include <cstdlib> 
#include <queue>

class Network;

// Library functions
float MSE(std::vector<float> targetvector, std::vector<float> outvector);
float total_cost(std::vector<float> targetvector, std::vector<float> outvector);
float cost(float target, float out);
float cost_prime(float target, float output);
float sigmoid(float value);
float sum_sigmoid(std::vector<float> valuevector);
float sigmoid_prime(float value);
float sum_sigmoid_prime(std::vector<float> valuevector);
float timeweight(float value, float weight);
float timeweight_primeweight(float value, float weight);
float timeweight_primevalue(float value, float weight);

// Library classes
class Node {
    friend class Network;
private:
    std::function<float(std::vector<float>)> nodefunction;
    std::function<float(std::vector<float>)> nodefunction_prime;
    std::vector<float> nodeinput;
    float nodeoutput;
    float bias = 0;

public:
    Node();
    Node(const std::function<float(std::vector<float>)> &inputfunction);
    void setnodefuction_prime(const std::function<float(std::vector<float>)> &nodefunctionprime);
    void runnodefunction();
    void clear_input();
    void setbias(float value);
};

class Edge {
    friend class Network;
private:
    float weight = 0;
    std::array<int, 4> connectionarray;
    std::function<float(float, float)> edgefunction;
    std::function<float(float, float)> edgefunction_primeweight;
    std::function<float(float, float)> edgefunction_primevalue;

public:
    Edge();
    Edge(std::function<float(float, float)> &inedgefunction, int iny, int outy, int inx = -1, int outx = -1);
    void setweight(float value);
    void setedgefunction_primeweight(const std::function<float(float, float)> &inedgefunctionprimeweight);
    void setedgefunction_primevalue(const std::function<float(float, float)> &inedgefunctionprimevalue);
};
class Network {
private:
    std::unordered_map<int, std::unordered_map<int, Node>> nodespace;
    std::unordered_map<int, std::unordered_map<int, Edge>> edgespace;
    std::vector<std::array<int, 2>> inspace;
    std::vector<std::array<int, 2>> outspace;
    std::unordered_map<int, std::unordered_map<int, std::vector<std::pair<int, int>>>> outgroup_edgespace;
    std::unordered_map<int, std::unordered_map<int, std::vector<std::pair<int, int>>>> ingroup_edgespace;

public:
    // Base network editing
    void putnode(int x, int y, Node innode);
    void erasenode(int x, int y);
    void erasenodelayer(int x);
    void putedge(int x, int y, Edge inedge);
    void eraseedge(int x, int y);
    void eraseedgelayer(int x);

    // Base node and edge activation
    void activatenode(int x, int y);
    void clearnodeinput(int x, int y);
    void activateedge(int x, int y);

    // Base value manipulation
    std::vector<float> getnodeinput(int x, int y);
    float getnodeoutput(int x, int y);
    std::array<int, 4> getedgeconnection(int x, int y);
    void setnodeinput(int x, int y, std::vector<float> invector);
    void setnodeouput(int x, int y, float invalue);

    // Additional functions
    void setinput_node(int x, int y);
    void setoutput_node(int x, int y);
    void setinput_nodebylayer(int x);
    void setoutput_nodebylayer(int x);
    std::vector<float> getoutput();
    void setinputnode_output(std::vector<float> invector);
    void organize_edge();
    void create_node(int layerpos, int startpos, unsigned int endpos, std::function<float(std::vector<float>)> nodefunction = sum_sigmoid, std::function<float(std::vector<float>)> nodefunction_prime = sum_sigmoid_prime);
    void xavier_init(Edge &edge, int n_in, int n_out);
    unsigned int connectall(int frontx, int backx, std::function<float(float, float)> edgefunction = timeweight, std::function<float(float, float)> edgefunction_primeweight = timeweight_primeweight, std::function<float(float, float)> edgefunction_primevalue = timeweight_primevalue);
    void clearallinput();
    void feedforward();
    void stochastic_backpropagate(std::vector<float> targetvector, float learning_rate = 0.1, const std::function<float(float, float)> &costprime = cost_prime);
};

#endif //AINET_H