#ifndef INFCCL_COMMON_H_
#define INFCCL_COMMON_H_

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <atomic>
#include <thread>
#include <unordered_map>
#include <vector>
#include <string>
#include <memory>
#include <functional>

#if defined(__x86_64__)
#include <immintrin.h>
#define INFCCL_PAUSE() _mm_pause()
#else
#define INFCCL_PAUSE()
#endif

#define INFCCL_LIKELY(x)   __builtin_expect(!!(x), 1)
#define INFCCL_UNLIKELY(x) __builtin_expect(!!(x), 0)

#ifndef INFCCL_MAX_DEVS
#define INFCCL_MAX_DEVS 8
#endif
namespace infccl {

enum ErrorCode {
    OK                = 0,
    ERR_CUDA          = -1,
    ERR_MEMORY        = -2,
    ERR_INVALID_ARG   = -3,
    ERR_TRANSPORT     = -4,
    ERR_BATCH_BUSY    = -5,
    ERR_NOT_FOUND     = -6,
    ERR_OVERLAP       = -7,
    ERR_TIMEOUT       = -8,
    ERR_UNSUPPORTED   = -9,
};

static inline int64_t getCurrentTimeNano() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return int64_t(ts.tv_sec) * 1000000000LL + int64_t(ts.tv_nsec);
}

static inline bool memOverlap(const void* a, size_t alen, const void* b, size_t blen) {
    return (a >= b && a < (const char*)b + blen) ||
           (b >= a && b < (const char*)a + alen);
}

class RWSpinlock {
    union Ticket {
        constexpr Ticket() : whole(0) {}
        uint64_t whole;
        uint32_t readWrite;
        struct { uint16_t write; uint16_t read; uint16_t users; };
    } ticket_;

    template<class T> static T loadAcquire(T* addr) {
        T t = *addr; asm volatile("" ::: "memory"); return t;
    }
    template<class T> static void storeRelease(T* addr, T v) {
        asm volatile("" ::: "memory"); *addr = v;
    }

public:
    RWSpinlock() {}
    RWSpinlock(const RWSpinlock&) = delete;
    RWSpinlock& operator=(const RWSpinlock&) = delete;

    bool tryLock() {
        Ticket t;
        uint64_t old = t.whole = loadAcquire(&ticket_.whole);
        if (t.users != t.write) return false;
        ++t.users;
        return __sync_bool_compare_and_swap(&ticket_.whole, old, t.whole);
    }

    void lock() {
        uint32_t cnt = 0;
        while (!tryLock()) { INFCCL_PAUSE(); if (++cnt > 1000) std::this_thread::yield(); }
    }

    void unlock() {
        Ticket t; t.whole = loadAcquire(&ticket_.whole);
        ++t.read; ++t.write;
        storeRelease(&ticket_.readWrite, t.readWrite);
    }

    bool tryLockShared() {
        Ticket t, old;
        old.whole = t.whole = loadAcquire(&ticket_.whole);
        old.users = old.read; ++t.read; ++t.users;
        return __sync_bool_compare_and_swap(&ticket_.whole, old.whole, t.whole);
    }

    void lockShared() {
        uint32_t cnt = 0;
        while (!tryLockShared()) { INFCCL_PAUSE(); if (++cnt > 1000) std::this_thread::yield(); }
    }

    void unlockShared() { __sync_fetch_and_add(&ticket_.write, 1); }

    struct WriteGuard {
        WriteGuard(RWSpinlock& l) : l_(l) { l_.lock(); }
        ~WriteGuard() { l_.unlock(); }
        WriteGuard(const WriteGuard&) = delete;
        RWSpinlock& l_;
    };

    struct ReadGuard {
        ReadGuard(RWSpinlock& l) : l_(l) { l_.lockShared(); }
        ~ReadGuard() { l_.unlockShared(); }
        ReadGuard(const ReadGuard&) = delete;
        RWSpinlock& l_;
    };
};

class TicketLock {
    std::atomic<int> next_{0};
    std::atomic<int> serving_{0};
    uint64_t pad_[14];
public:
    void lock() {
        int my = next_.fetch_add(1, std::memory_order_relaxed);
        while (serving_.load(std::memory_order_acquire) != my) std::this_thread::yield();
    }
    void unlock() { serving_.fetch_add(1, std::memory_order_release); }
};

}
#endif
