#include "ainet.h"
//libery function
//square cost function
float MSE(std::vector<float> targetvector, std::vector<float> outvector){
    float sum = 0;
    for (int i=0; i < outvector.size(); i++){
        sum += pow(targetvector[i] - outvector[i], 2);
    }
    return sum/outvector.size()/2;
}
float total_cost(std::vector<float> targetvector, std::vector<float> outvector){
    float sum = 0;
    for (int i=0; i < outvector.size(); i++){
        sum += pow(targetvector[i] - outvector[i], 2)/2;
    }
    return sum;
}
float cost(float target, float out){
    return pow(target - out, 2)/2;
}
float cost_prime(float target, float output){
    return output - target;
}
//logistic function
float sigmoid(float value){
    return 1/(1+exp(-value));
}
float sum_sigmoid(std::vector<float> valuevector){
    float sum = 0;
    for (auto & value : valuevector) {
    sum += value;
    }
    //std::cout<<sum;
    return sigmoid(sum);
}
float sigmoid_prime(float value){
    float out = sigmoid(value);
    return out*(1-out);
}
float sum_sigmoid_prime(std::vector<float> valuevector){
    float sum = 0;
    for (auto & value : valuevector) {
    sum += value;
    }
    float out = sigmoid(sum);
    return out*(1-out);
}

//default edge function
float timeweight(float value, float weight){
    return value * weight;
}
float timeweight_primeweight(float value, float weight){
    return value;
}
float timeweight_primevalue(float value, float weight){
    return weight;
}

//libery class

Node::Node(){}
Node::Node(const std::function<float(std::vector<float>)> & inputfunction):nodefunction(inputfunction){}

void Node::setnodefuction_prime(const std::function<float(std::vector<float>)> & nodefunctionprime){
    nodefunction_prime = nodefunctionprime;
}
void Node::runnodefunction(){
    nodeinput.push_back(bias);
    nodeoutput = nodefunction(nodeinput);
    nodeinput.pop_back();
}
void Node::clear_input(){
    nodeinput.clear();
}
void Node::setbias(float value){
    bias = value;
}


Edge::Edge(){}
// -1 mean to take x from prediction n,n+1 in edgespace
Edge::Edge(std::function<float(float, float)>& inedgefunction, int iny, int outy, int inx, int outx): edgefunction(inedgefunction), connectionarray({iny, outy, inx, outx}){}
void Edge::setweight(float value){
    weight = value;
}
void Edge::setedgefunction_primeweight(const std::function<float(float, float)> & inedgefunctionprimeweight){
    edgefunction_primeweight = inedgefunctionprimeweight;
}
void Edge::setedgefunction_primevalue(const std::function<float(float, float)> & inedgefunctionprimevalue){
    edgefunction_primevalue = inedgefunctionprimevalue;
}



//base network editing
void Network::putnode(int x, int y, Node innode){
    nodespace[x][y] = innode;
}
void Network::erasenode(int x, int y){
    nodespace[x].erase(y);
}
void Network::erasenodelayer(int x){
    nodespace.erase(x);
}
void Network::putedge(int x, int y, Edge inedge){
    Edge currentedge = inedge;
    std::array<int, 4> connection = currentedge.connectionarray;
    //check for default condition and set connect to current layer and next layer if True
    if (connection[2] == -1){
        connection[2] = x;
    }
    if (connection[3] == -1){
        connection[3] = x + 1;
    }
    edgespace[x][y] = currentedge;
}
void Network::eraseedge(int x, int y){
    edgespace[x].erase(y);
}
void Network::eraseedgelayer(int x){
    edgespace.erase(x);
}

//base node and edge activation
void Network::activatenode(int x, int y){
    nodespace[x][y].runnodefunction();
}
void Network::clearnodeinput(int x, int y){
    nodespace[x][y].clear_input();
}
void Network::activateedge(int x, int y){
    //get edge node
    Edge currentedge = edgespace[x][y];
    std::array<int, 4> connection = currentedge.connectionarray;
    //run edge's function with inputnode value, take function return value and append it to output node
    nodespace[connection[3]][connection[1]].nodeinput
    .push_back(currentedge.edgefunction
    (nodespace[connection[2]][connection[0]].nodeoutput, currentedge.weight));
}

//base value manipution
//get value
std::vector<float> Network::getnodeinput(int x, int y){
    return nodespace[x][y].nodeinput;
}
float Network::getnodeoutput(int x, int y){
    return nodespace[x][y].nodeoutput;
}
std::array<int, 4> Network::getedgeconnection(int x, int y){
    return edgespace[x][y].connectionarray;
}
//set value
void Network::setnodeinput(int x, int y, std::vector<float> invector){
    nodespace[x][y].nodeinput = invector;
}
void Network::setnodeouput(int x, int y, float invalue){
    nodespace[x][y].nodeoutput = invalue;
}

//additional function
//set in-out node
void Network::setinput_node(int x, int y){
    inspace.push_back({x, y});
}
void Network::setoutput_node(int x, int y){
    outspace.push_back({x, y});
}
void Network::setinput_nodebylayer(int x){
    for(auto & node: nodespace[x]){
        inspace.push_back({x, node.first});
    }
}
void Network::setoutput_nodebylayer(int x){
    for(auto & node: nodespace[x]){
        outspace.push_back({x, node.first});
    }
}
//get value in adding order
std::vector<float> Network::getoutput(){
    std::vector<float> outvector;
    for (int i=0; i < outspace.size(); i++){
        int  x = outspace[i][0], y = outspace[i][1];
        outvector.push_back(nodespace[x][y].nodeoutput);
    }
    return outvector;
}
//get value in adding order
void Network::setinputnode_output(std::vector<float> invector){
    for (int i=0; i < inspace.size(); i++){
        int  x = inspace[i][0], y = inspace[i][1];
        nodespace[x][y].nodeoutput = invector[i];
    }
}
//neural network function
//store edge in array map to it output node
void Network::organize_edge(){
    for (auto & layer : edgespace) {
        for (auto & inedge : layer.second){
            Edge currentedge = inedge.second;
            std::pair<int, int>spacepos = {layer.first, inedge.first};
            outgroup_edgespace[currentedge.connectionarray[3]][currentedge.connectionarray[1]].push_back(spacepos);
            ingroup_edgespace[currentedge.connectionarray[2]][currentedge.connectionarray[0]].push_back(spacepos);
        }
    }
}
//connect all node in 2 layer with edge, insert edge in edgespace at the same key as input layer
void Network::create_node(int layerpos, int startpos, unsigned int endpos
, std::function<float(std::vector<float>)> nodefunction, std::function<float(std::vector<float>)> nodefunction_prime){
    for (auto i = startpos; i < endpos; i++ ){
        Node innode(nodefunction);
        innode.setbias(0);
        innode.setnodefuction_prime(nodefunction_prime);
        nodespace[layerpos].insert({i, innode});
    }
}
void Network::xavier_init(Edge &edge, int n_in, int n_out) {
// Calculate the Xavier initialization scale factor
float scale_factor = sqrt(6.0 / (n_in + n_out));

// Initialize the weight with a random value within the range [-scale_factor, scale_factor]
float weight = (static_cast<float>(rand()) / RAND_MAX) * 2.0f * scale_factor - scale_factor;
edge.setweight(weight);
}
unsigned int Network::connectall(int frontx, int backx, std::function<float(float, float)> edgefunction, std::function<float(float, float)> edgefunction_primeweight, std::function<float(float, float)> edgefunction_primevalue){
    unsigned int c = 0;
    for (auto & frontnodey : nodespace[frontx]){
        for (auto & backnodey : nodespace[backx]){
            Edge inedge(edgefunction, frontnodey.first, backnodey.first, frontx, backx);
            xavier_init(inedge, nodespace[frontx].size(), nodespace[backx].size());
            inedge.setedgefunction_primeweight(edgefunction_primeweight);
            inedge.setedgefunction_primevalue(edgefunction_primevalue);
            //c++ hehe
            edgespace[frontx].insert({c++, inedge});
        }
    }
    return c;
}
void Network::clearallinput(){
    for (auto & layer : nodespace) {
        for (auto & node : layer.second)
            node.second.clear_input();
    }
}
//feedforwark ***does not work with loop connection / rnn***
void Network::feedforward() {
    // To track visited neurons
    std::unordered_map<int, std::unordered_map<int, bool>> visited;
    std::queue<std::pair<int, int>> nodes_to_process;
    //input nodes to the queue
    for (const auto& input_node : inspace) {
        int currentx = input_node[0];
        int currenty = input_node[1];
        visited[currentx][currenty] = true;
        for (const auto& currentedgepos : ingroup_edgespace[currentx][currenty]) {//
            Edge currentedge = edgespace[currentedgepos.first][currentedgepos.second];
            std::array<int, 4> connectionarray = currentedge.connectionarray;
            nodespace[connectionarray[3]][connectionarray[1]].nodeinput
            .push_back(currentedge.edgefunction
            (nodespace[currentx][currenty].nodeoutput, currentedge.weight));
            std::pair<int, int> next_node = {connectionarray[3], connectionarray[1]};
            nodes_to_process.push(next_node);
        }
    }
    while (!nodes_to_process.empty()) {
        std::pair<int, int> node = nodes_to_process.front();
        nodes_to_process.pop();

        int currentx = node.first;
        int currenty = node.second;

        // Check if the neuron is visited
        if (visited[currentx][currenty]) {
            continue; // Skip this neuron
        }

        // Run the node's function
        //std::cout << (nodespace[currentx][currenty].nodeinput)[0];
        nodespace[currentx][currenty].runnodefunction();
        if (outgroup_edgespace[currentx][currenty].size() != nodespace[currentx][currenty].nodeinput.size()){
                nodes_to_process.push({currentx, currenty});
                continue;
        }
        visited[currentx][currenty] = true;
        // Add nodes connected to this one to the queue
        for (const auto& currentedgepos : ingroup_edgespace[currentx][currenty]) {
            Edge currentedge = edgespace[currentedgepos.first][currentedgepos.second];
            std::array<int, 4> connectionarray = currentedge.connectionarray;
            nodespace[connectionarray[3]][connectionarray[1]].nodeinput
            .push_back(currentedge.edgefunction
            (nodespace[currentx][currenty].nodeoutput, currentedge.weight));
            std::pair<int, int> next_node = {connectionarray[3], connectionarray[1]};
            nodes_to_process.push(next_node);
        }
    }
}

//stochastic backpropagation ***does not work with loop connection / rnn or not clearing input***
void Network::stochastic_backpropagate(std::vector<float> targetvector, float learning_rate, const std::function<float(float, float)> & costprime){
    //output gradient for hidden layer
    std::unordered_map<int, std::unordered_map<int , std::vector<float>>> output_gradientmap;
    //neuron to calculate
    std::unordered_map<int, std::unordered_map<int, bool>> visited;
    std::queue<std::pair<int, int>> nodes_to_process;
    //outputlayer
    //find input and output gradient of output neuron
    for (int i=0; i < outspace.size(); i++){
        int currentx = outspace[i][0];
        int currenty = outspace[i][1];
        if (visited[currentx][currenty]){
            continue;
        }
        //check if all previous input gradient was calculated else halt this neuron until calculatetion finish
        if (output_gradientmap[currentx][currenty].size() != ingroup_edgespace[currentx][currenty].size()){
                nodes_to_process.push({currentx, currenty});
                continue;
        }
        visited[currentx][currenty] = true;
        float target = targetvector[i];
        Node currentnode = nodespace[currentx][currenty];
        float input_gradient = currentnode.nodefunction_prime(currentnode.nodeinput) * cost_prime(target, currentnode.nodeoutput);
        nodespace[currentx][currenty].setbias(currentnode.bias - (learning_rate*input_gradient));
        //find weight gradient of each edge
        for (const auto & currentedgepos: outgroup_edgespace[currentx][currenty]){
            Edge currentedge = edgespace[currentedgepos.first][currentedgepos.second];
            std::array<int, 4> connectionarray = currentedge.connectionarray;
            float next_nodeoutput = nodespace[connectionarray[2]][connectionarray[0]].nodeoutput;
            float weight_gradient = input_gradient * currentedge.edgefunction_primeweight(next_nodeoutput, currentedge.weight);
            output_gradientmap[connectionarray[2]][connectionarray[0]].push_back(input_gradient * currentedge.edgefunction_primevalue(next_nodeoutput, currentedge.weight));
            nodes_to_process.push({connectionarray[2], connectionarray[0]});
            //update weight
            edgespace[currentedgepos.first][currentedgepos.second].setweight(currentedge.weight - (learning_rate*weight_gradient));
        }
    }
    //hidden layer
    while (!nodes_to_process.empty()){
        std::pair<int, int> node = nodes_to_process.front();
        nodes_to_process.pop();
        int currentx = node.first;
        int currenty = node.second;
        if (visited[currentx][currenty]){
            continue;
        }
        //check if all previous input gradient was calculated else halt this neuron until calculation finish
        if (output_gradientmap[currentx][currenty].size() != ingroup_edgespace[currentx][currenty].size()){
            nodes_to_process.push({currentx, currenty});
            continue;
        }
        visited[currentx][currenty] = true;
        Node currentnode = nodespace[currentx][currenty];
        float output_gradient = 0;
        for (auto & previous_gradient: output_gradientmap[currentx][currenty]){
            output_gradient += previous_gradient;
        }
        float input_gradient = currentnode.nodefunction_prime(currentnode.nodeinput) * output_gradient;
        nodespace[currentx][currenty].setbias(currentnode.bias - (learning_rate*input_gradient));
        for (auto & currentedgepos: outgroup_edgespace[currentx][currenty]){
            Edge currentedge = edgespace[currentedgepos.first][currentedgepos.second];
            std::array<int, 4> connectionarray = currentedge.connectionarray;
            float next_nodeoutput = nodespace[connectionarray[2]][connectionarray[0]].nodeoutput;
            float weight_gradient = input_gradient * currentedge.edgefunction_primeweight(next_nodeoutput, currentedge.weight);
            output_gradientmap[connectionarray[2]][connectionarray[0]].push_back(input_gradient * currentedge.edgefunction_primevalue(next_nodeoutput, currentedge.weight));
            nodes_to_process.push({connectionarray[2], connectionarray[0]});
            //update weight
            edgespace[currentedgepos.first][currentedgepos.second].setweight(currentedge.weight - (learning_rate*weight_gradient));
        }
        
    }
}