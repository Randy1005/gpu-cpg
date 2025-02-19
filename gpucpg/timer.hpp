#include <chrono>
#include <iostream>

using namespace std::chrono_literals;
struct Timer {
  using clock_t = std::chrono::high_resolution_clock;
  using time_point_t = clock_t::time_point;
  using elapsed_time_t = std::chrono::duration<double, std::micro>;


  time_point_t tbeg, tend;
  elapsed_time_t elapsed_time;

	bool is_paused{false};

  void start() {
		elapsed_time = elapsed_time_t::zero();
		tbeg = clock_t::now();
  }

  void stop() {
		tend = clock_t::now();
		elapsed_time = std::chrono::duration_cast<elapsed_time_t>(tend-tbeg);	
	}

	void pause() {
		tend = clock_t::now();
		elapsed_time += std::chrono::duration_cast<elapsed_time_t>(tend-tbeg);
		is_paused = true;
	}

	void resume() {
		tbeg = clock_t::now();
		is_paused = false;
	}

	void reset() {
		elapsed_time = elapsed_time_t::zero();
	}

	auto get_elapsed_time() {
		return elapsed_time;
	}
};
