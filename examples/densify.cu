#include "gpucpg.cuh"

int main(int argc, char* argv[]) {
  if (argc != 4 && argc != 5) {
		std::cerr << "usage: densify [desired_avg_degree] [input] [output] [seed=1]\n";
		std::exit(EXIT_FAILURE);
	}

	auto desired_avg_degree = std::stoi(argv[1]);
	std::string input_filename = argv[2];
	std::string output_filename = argv[3];
	const auto seed = argc == 5 ? static_cast<std::uint32_t>(std::stoul(argv[4])) : 1u;
	gpucpg::CpGen cpgen;

	cpgen.read_input(input_filename);
	cpgen.densify_graph(desired_avg_degree, seed);
	
	cpgen.export_to_benchmark(output_filename);

	return 0;
}
