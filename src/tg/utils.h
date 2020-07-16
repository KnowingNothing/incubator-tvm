#ifndef TVM_TG_UTILS_H_
#define TVM_TG_UTILS_H_

#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>
#include <utility>
#include <tuple>
#include <chrono>

#include <tvm/te/operation.h>
#include <tvm/te/schedule.h>
#include <tvm/te/tensor.h>
#include <tvm/driver/driver_api.h>
#include <tvm/target/target.h>
#include <tvm/runtime/module.h>


namespace tvm {

namespace tg {


#define TG_DEFINE_OBJECT_SELF_METHOD(ObjectName)         \
  ObjectName* Self() {                                   \
    CHECK(data_ != nullptr);                             \
    return static_cast<ObjectName*>(data_.get());        \
  }


// template<typename ELE>
// class UnpackVec {
//  public:
// 	UnpackVec(const std::vector<ELE>& vec) : vec(vec), index(0) {}

//   template<typename T>
// 	ELE unpack()	{
// 		return  vec[index++];
// 	}

//  private:
// 	const std::vector<ELE>& vec;
// 	int index;
// };

// template<typename R, typename... Args, typename ELE>
// auto call_function(std::function<R(Args...)> f, std::vector<ELE> &v) {
//     UnpackVec<ELE> unpackvec(v);
//     return f(unpackvec.unpack<Args>()...);
// }

template<typename Function, typename Tuple, size_t ... I>
void call(Function f, Tuple &t, std::index_sequence<I ...>) {
     f(std::get<I>(t) ...);
}

template<typename Function, typename Tuple>
void call(Function f, Tuple &t) {
    static constexpr auto size = std::tuple_size<Tuple>::value;
    return call(f, t, std::make_index_sequence<size>());
}

template<typename Function, typename T, typename Tuple>
void call_function(Function f, std::vector<T> &v, Tuple &t) {
  if (v.empty()) call(f, t);
  else {
    auto new_t = std::tuple_cat(std::make_tuple(v.back()), t);
    v.pop_back();
    call_function(f, v, t);
  }
}

template<typename Function, typename T>
void call_function(Function f, std::vector<T> v) {
  auto t = std::make_tuple();
  call_function(f, v, t);
}

class ThreadPool {
public:
    ThreadPool(size_t, unsigned int);

    template<typename FType, typename... Args>
    auto push_front(FType&& f, Args&&... args) -> std::future<typename std::result_of<FType(Args...)>::type> {
      using return_type = decltype(f(args...));

      auto task = std::make_shared< std::packaged_task<return_type()> >(
              std::bind(f, std::forward<Args>(args)...)
          );
          
      std::future<return_type> res = task->get_future();
      {
          std::unique_lock<std::mutex> lock(deque_mutex);

          if(stop)
              throw std::runtime_error("push_front on stopped ThreadPool");

          tasks.emplace_back([task, this](){
            std::thread th([task](){ (*task)(); });
            std::this_thread::sleep_for(std::chrono::milliseconds(this->timeout));
            pthread_cancel(th.native_handle());
          });
      }
      condition.notify_one();
      return res;
    }

    template<typename FType, typename... Args>
    auto push_back(FType&& f, Args&&... args) -> std::future<typename std::result_of<FType(Args...)>::type> {
      using return_type = decltype(f(args...));

      auto task = std::make_shared< std::packaged_task<return_type()> >(
              std::bind(f, std::forward<Args>(args)...)
          );
          
      std::future<return_type> res = task->get_future();
      {
          std::unique_lock<std::mutex> lock(deque_mutex);

          if(stop)
              throw std::runtime_error("push_back on stopped ThreadPool");

          tasks.emplace_back([task, this](){
            std::thread th([task](){ (*task)(); });
            std::this_thread::sleep_for(std::chrono::milliseconds(this->timeout));
            pthread_cancel(th.native_handle());
          });
      }
      condition.notify_one();
      return res;
    }

    void clear_threads();

    static ThreadPool& Global();

    ~ThreadPool();
private:

    std::vector< std::thread > workers;
    std::deque< std::function<void()> > tasks;
    
    std::mutex deque_mutex;
    std::condition_variable condition;
    bool stop;

    unsigned int timeout;

    static const int REFRESH_EPOCH = 128;
};


template<typename T>
class Queue {
 private:
  std::queue<T> q;
  std::mutex mutex;

 public:
  void push(T value);
  T pop();
  bool empty();
};

 
}  // namespace tg

}  // namespace tvm

#endif  //  TVM_TG_UTILS_H_