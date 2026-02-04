// main.cpp - simple RingBuffer usage example

#include <iostream>

#include "ring_buffer.hpp"

int main() {
  RingBuffer<int> rb(3);

  std::cout << "push 1,2,3 into capacity=3\n";
  rb.push(1);
  rb.push(2);
  rb.push(3);

  std::cout << "buffer full? " << std::boolalpha << rb.full() << "\n";
  std::cout << "push(4) success? " << rb.push(4) << " (expected false)\n";

  std::cout << "pop all:\n";
  while (auto v = rb.pop()) {
    std::cout << "  " << *v << "\n";
  }

  std::cout << "\noverwrite example (capacity=3):\n";
  rb.push_overwrite(10);
  rb.push_overwrite(11);
  rb.push_overwrite(12);
  rb.push_overwrite(13);  // overwrites 10
  rb.push_overwrite(14);  // overwrites 11

  std::cout << "pop all (expected 12,13,14):\n";
  while (auto v = rb.pop()) {
    std::cout << "  " << *v << "\n";
  }

  return 0;
}

