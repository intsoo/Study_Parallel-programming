/*
 * =====================================================================================
 *
 *       Filename:  mkClockMeasure.hpp
 *
 *    Description: to count clock
 *
 *        Version:  1.0
 *        Created:  09/17/20 23:36:29
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Yoon, Myung Kuk, myungkuk.yoon@gmail.com
 *   Organization:  Ewha Womans University
 *
 * =====================================================================================
 */

#pragma once
#include <time.h>
#include <string>

const int64_t nanoToSecTime=1000000000;

class mkClockMeasure {
 private:
  int64_t count;
  int64_t time;
  struct timespec begin, end;
  std::string name;

 public:
  mkClockMeasure(const std::string input) : count(0), time(0), name(input) {}
  ~mkClockMeasure() {}

  void clockReset() {
    time = 0;
    count = 0;
  }
  void clockResume() { clock_gettime(CLOCK_MONOTONIC, &begin); }
  void clockPause() {
    count++;
    clock_gettime(CLOCK_MONOTONIC, &end);
    time += (int64_t)(end.tv_sec - begin.tv_sec) * nanoToSecTime +
	    (int64_t)(end.tv_nsec - begin.tv_nsec);
  }
  void clockPrint() {
    printf("mkClockMeasure[");
    if (count == 0) {
      printf("*None*]\n");
      return;
    }
    if (time < 1000) {
      printf("Total Time: \t%lf (ns), \tAvg Time: \t%lf (ns), \tCount: \t%ld",
	     (double)time, (double)time / count, count);
    } else if (time < 1000000) {
      printf("Total Time: \t%lf (us), \tAvg Time: \t%lf (us), \tCount: \t%ld",
	     (double)time / 1000, (double)time / (1000 * count), count);
    } else if (time < nanoToSecTime) {
      printf("Total Time: \t%lf (ms), \tAvg Time: \t%lf (ms), \tCount: \t%ld",
	     (double)time / 1000000, (double)time / (1000000 * count), count);
    } else {
      printf("Total Time: \t%lf (s), \tAvg Time: \t%lf (s), \tCount: \t%ld",
	     (double)time / nanoToSecTime, (double)time / (nanoToSecTime * count),
	     count);
    }
    printf("]\n");
  }
};
