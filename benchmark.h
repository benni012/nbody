#ifndef NBODY_BENCHMARK_H
#define NBODY_BENCHMARK_H
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#define BENCHMARK_START(name) Benchmark::getInstance().start(name)
#define BENCHMARK_STOP(name) Benchmark::getInstance().stop(name)

class Benchmark {
public:
  static Benchmark &getInstance() {
    static Benchmark instance;
    return instance;
  }
  void enableBenchmarking(bool enable) { benchmarkingEnabled = enable; }

  void start(const std::string &name) {
    if (!benchmarkingEnabled)
      return;
    auto now = std::chrono::high_resolution_clock::now();
    cpuTimers[name].first = now;
  }

  void stop(const std::string &name) {
    if (!benchmarkingEnabled)
      return;
    auto now = std::chrono::high_resolution_clock::now();
    auto &timer = cpuTimers[name];
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(now - timer.first)
            .count();
    cpuTimings[name].push_back(duration);
  }

  void saveResults(const std::string &filename) {
    if (!benchmarkingEnabled)
      return;
    std::ofstream file(filename);
    if (!file.is_open()) {
      std::cerr << "Failed to open file: " << filename << std::endl;
      return;
    }
    std::cout << "Timings:" << std::endl;
    std::cout << std::left << std::setw(20) << "Function" << std::setw(10)
              << "Calls" << std::setw(20) << "Mean Time (us)" << "Std Dev (us)"
              << std::endl;
    file << "Function,Calls,Mean Time (us),Std Dev (us)" << std::endl;
    for (const auto &[name, timings] : cpuTimings) {
      auto [mean, stdDev] = computeStats(timings);
      std::cout << std::left << std::setw(20) << name << std::setw(10)
                << timings.size() << std::setw(20) << mean << std::setw(20)
                << stdDev << std::endl;
      file << name << "," << timings.size() << "," << mean << "," << stdDev
           << std::endl;
    }
    file.close();
  }

private:
  Benchmark() : benchmarkingEnabled(false) {}
  ~Benchmark() = default;

  bool benchmarkingEnabled;
  // map func name : chrono ms
  std::unordered_map<
      std::string,
      std::pair<std::chrono::high_resolution_clock::time_point, long long>>
      cpuTimers;
  std::unordered_map<std::string, std::vector<long long>> cpuTimings;

  template <typename T>
  std::pair<T, T> computeStats(const std::vector<T> &data) const {
    T mean = std::accumulate(data.begin(), data.end(), T(0)) / data.size();
    T variance = std::accumulate(data.begin(), data.end(), T(0),
                                 [mean](T acc, T val) {
                                   return acc + (val - mean) * (val - mean);
                                 }) /
                 data.size();
    return {mean, std::sqrt(variance)};
  }
};
#endif // NBODY_BENCHMARK_H
