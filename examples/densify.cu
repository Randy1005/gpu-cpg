#include "gpucpg.cuh"

int main(int argc, char* argv[]) {
  if (argc != 4) {
		std::cerr << "usage: ./a.out [desired_avg_degree] [input] [output]\n";
		std::exit(EXIT_FAILURE);
	}

	auto desired_avg_degree = std::stoi(argv[1]);
	std::string input_filename = argv[2];
	std::string output_filename = argv[3];
	gpucpg::CpGen cpgen;

	cpgen.read_input(input_filename);
	cpgen.densify_graph(desired_avg_degree);
	
	cpgen.export_to_benchmark(output_filename);

	return 0;
}