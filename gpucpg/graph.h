#include <iostream>
#include <vector>

struct Graph
{
    int N;
    int M;
    std::vector<int> Vs;
    std::vector<int> Es;
    std::vector<float> Ws;
    bool weighted;

    Graph(int N, int M, std::vector<int>& Vs,
        std::vector<int>& Es, std::vector<float>& Ws, 
        bool weighted) {
            this->N = N;
            this->M = M;
            std::copy(Vs.begin(), Vs.end(), std::back_inserter(this->Vs));
            std::copy(Es.begin(), Es.end(), std::back_inserter(this->Es));
            std::copy(Ws.begin(), Ws.end(), std::back_inserter(this->Ws));
            this->weighted = weighted;
        }
    
    void write_to_bin(std::string& filename) {
        std::ofstream ofs(filename, std::ios::binary);
        if (!ofs) {
            std::cerr << "Error opening file for writing: " << filename << std::endl;
            return;
        }
        
        ofs.write(reinterpret_cast<char*>(&N), sizeof(unsigned int));
        ofs.write(reinterpret_cast<char*>(&M), sizeof(unsigned int));
        ofs.write(reinterpret_cast<char*>(Vs.data()), (N+1)*sizeof(unsigned int));
        ofs.write(reinterpret_cast<char*>(Es.data()), M*sizeof(unsigned int));
        ofs.write(reinterpret_cast<char*>(Ws.data()), M*sizeof(float));
        
        ofs.close();
    }

    void read_from_bin(std::string& filename) {
        std::ifstream ifs(filename, std::ios::binary);
        if (!ifs) {
            std::cerr << "Error opening file for reading: " << filename << std::endl;
            return;
        }
        ifs.read(reinterpret_cast<char*>(&N), sizeof(N));
        ifs.read(reinterpret_cast<char*>(&M), sizeof(M));
        Vs.resize(N+1);
        Es.resize(M);
        Ws.resize(M);
        ifs.read(reinterpret_cast<char*>(Vs.data()), (N+1)*sizeof(int));
        ifs.read(reinterpret_cast<char*>(Es.data()), M*sizeof(int));
        ifs.read(reinterpret_cast<char*>(Ws.data()), M*sizeof(float));
        ifs.close();
    }


};

