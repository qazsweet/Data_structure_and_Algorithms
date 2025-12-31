// ring_buffer.hpp
// Simple fixed-capacity ring (circular) buffer.
// C++17, single-producer/single-consumer friendly if externally synchronized.
//
// Key operations:
// - push(...): returns false if full (does not overwrite)
// - push_overwrite(...): always writes; overwrites oldest when full
// - pop(): returns std::optional<T>

#pragma once

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

template <class T>
class RingBuffer {
 public:
  explicit RingBuffer(std::size_t capacity)
      : data_(capacity), capacity_(capacity) {
    if (capacity_ == 0) {
      throw std::invalid_argument("RingBuffer capacity must be > 0");
    }
  }

  [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }
  [[nodiscard]] std::size_t size() const noexcept { return size_; }
  [[nodiscard]] bool empty() const noexcept { return size_ == 0; }
  [[nodiscard]] bool full() const noexcept { return size_ == capacity_; }

  void clear() noexcept {
    head_ = 0;
    tail_ = 0;
    size_ = 0;
  }

  bool push(const T& value) { return emplace_impl(value); }
  bool push(T&& value) { return emplace_impl(std::move(value)); }

  // Writes even if full; if full, drops the oldest element.
  void push_overwrite(const T& value) { emplace_overwrite_impl(value); }
  void push_overwrite(T&& value) { emplace_overwrite_impl(std::move(value)); }

  std::optional<T> pop() {
    if (empty()) return std::nullopt;
    T out = std::move(data_[head_]);
    head_ = next_index(head_);
    --size_;
    return out;
  }

  // Peek at the oldest element without removing it.
  // Precondition: !empty()
  const T& front() const {
    if (empty()) throw std::out_of_range("RingBuffer::front on empty buffer");
    return data_[head_];
  }

 private:
  std::vector<T> data_;
  std::size_t capacity_{0};
  std::size_t head_{0};  // next read
  std::size_t tail_{0};  // next write
  std::size_t size_{0};

  std::size_t next_index(std::size_t i) const noexcept {
    return (i + 1) % capacity_;
  }

  template <class U>
  bool emplace_impl(U&& value) {
    if (full()) return false;
    data_[tail_] = std::forward<U>(value);
    tail_ = next_index(tail_);
    ++size_;
    return true;
  }

  template <class U>
  void emplace_overwrite_impl(U&& value) {
    if (full()) {
      // Drop oldest to make room.
      head_ = next_index(head_);
      --size_;
    }
    data_[tail_] = std::forward<U>(value);
    tail_ = next_index(tail_);
    ++size_;
  }
};

