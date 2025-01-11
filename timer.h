#ifndef TIMER_H
#define TIMER_H

#include <chrono>

class Timer {
public:
    explicit Timer(bool startImmediately = false) : running(false) {
        if (startImmediately) {
            start();
        }
    }

    void start() {
        startTime = std::chrono::high_resolution_clock::now();
        running = true;
    }

    double stop() {
        if (running) {
            auto endTime = std::chrono::high_resolution_clock::now();
            running = false;
            std::chrono::duration<double> elapsed = endTime - startTime;
            return elapsed.count();
        }
        return 0.0;
    }

    double elapsed() const {
        if (running) {
            auto currentTime = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = currentTime - startTime;
            return elapsed.count();
        }
        return 0.0;
    }

private:
    std::chrono::high_resolution_clock::time_point startTime;
    bool running;
};

#endif // TIMER_H
