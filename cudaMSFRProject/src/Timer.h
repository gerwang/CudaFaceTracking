#pragma once
#include <ctime>


class Timer 
{
public:
  Timer():now_(time(0)), ltm_(localtime(&now_))
  {}

  void update_timer()
  {
    now_ = time(0);
    ltm_ = localtime(&now_);
  }

  const int year() const { return ltm_->tm_year + 1900; }
  const int month() const { return ltm_->tm_mon + 1; }
  const int day() const { return ltm_->tm_mday; }
  const int hour() const { return ltm_->tm_hour; }
  const int minute() const { return ltm_->tm_min; }
  const int second() const { return ltm_->tm_sec; }
  const int date() const { return year() * 10000 + month() * 100 + day(); }
  const int hms() const { return hour() * 10000 + minute() * 100 + second(); }

private:
  time_t now_;
  tm *ltm_;
  //int year_, month_, day_, hour_, minute_, second_;
};
