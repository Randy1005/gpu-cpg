#include "gpucpg.cuh"
#include <cassert>
#include <iomanip>

int main(int argc, char* argv[]) {
  if (argc != 6) {
    std::cerr << "usage: ./a.out [benchmark] [k] [delta_beg] [delta_inc] [N]\n";
    std::exit(1);
  }

  std::string benchmark = argv[1];
  auto num_paths = std::stoi(argv[2]);
  auto delta_beg = std::stof(argv[3]);
  auto delta_inc = std::stof(argv[4]);
  auto N = std::stoi(argv[5]);
  // delta_beg, delta_beg+delta_inc, ..., delta_beg+(N-1)*delta_inc

  auto cr_method = gpucpg::CsrReorderMethod::E_ORIENTED;
  bool enable_warp_spur{true};

  int max_dev_lvls{10};
  bool enable_compress{true};

  gpucpg::CpGen cpgen;
  cpgen.read_input(benchmark);


  // remove the full path from the benchmark name, just keep the file name
  std::string benchmark_name = benchmark.substr(benchmark.find_last_of("/\\")+1);

  // also remove the ".txt" extension
  benchmark_name = benchmark_name.substr(0, benchmark_name.find_last_of('.'));


  std::ofstream rt_vs_delta_csv(benchmark_name+"-rt-vs-delta.csv");
 
  // csvs to record the paths per step
  std::vector<std::ofstream> gen_paths_per_step_csvs(N);


  // write the header (also record the steps taken)
  rt_vs_delta_csv << "delta,steps,rt\n";
  
  for (int d = 0; d < N; d++) {

    float delta = delta_beg+d*delta_inc;
    cpgen.report_paths(num_paths, max_dev_lvls, enable_compress,
      gpucpg::PropDistMethod::LEVELIZE_THEN_RELAX, gpucpg::PfxtExpMethod::SHORT_LONG,
      false, 0.005f, 5.0f, 8, false, false, false, true,
      cr_method, enable_warp_spur, std::make_optional<float>(delta));

    // write the paths generated per step (accumulated)
    gen_paths_per_step_csvs[d].open(benchmark_name+"-acc-paths-per-step-d="+std::to_string(delta)+".csv");
    int total_steps = cpgen.paths_gen_per_step.size();
    gen_paths_per_step_csvs[d] << "step,paths\n";
    int accum_paths{0};
    for (int i = 0; i < total_steps; i++) {
      accum_paths += cpgen.paths_gen_per_step[i];
      gen_paths_per_step_csvs[d] << i << ',' << accum_paths << '\n';
    }
    gen_paths_per_step_csvs[d].close();

    auto total_time = (cpgen.prop_time+cpgen.expand_time)/1ms;
    rt_vs_delta_csv << std::fixed << std::setprecision(2)
      << delta << ','
      << cpgen.short_long_expansion_steps << ','
      << total_time << '\n';

    cpgen.reset();
  }

  // close the csv files
  rt_vs_delta_csv.close();

  std::cout << benchmark_name << " runtime vs delta (+work per step) written.\n";
  return 0;
}