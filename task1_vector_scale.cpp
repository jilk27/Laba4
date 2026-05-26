#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

// ──────────────────────────────────────────────
//  CPU-версия: умножение каждого элемента на k
// ──────────────────────────────────────────────
void vectorScaleCPU(const std::vector<float>& A,
                    std::vector<float>&       B,
                    float                     k)
{
    for (size_t i = 0; i < A.size(); ++i)
        B[i] = A[i] * k;
}

int main()
{
    try {
        // ── Выбор устройства ──────────────────────────────────────────────
        // Пробуем GPU; если его нет, падаем на CPU.
        sycl::queue q(sycl::default_selector_v,
                      [](sycl::exception_list el) {
                          for (auto& e : el)
                              std::rethrow_exception(e);
                      });

        std::cout << "Устройство: "
                  << q.get_device().get_info<sycl::info::device::name>()
                  << "\n\n";

        // ── Параметры ─────────────────────────────────────────────────────
        const size_t N = 1'000'000;
        const float  k = 2.5f;

        // ── Инициализация данных ──────────────────────────────────────────
        std::vector<float> A(N), B_cpu(N, 0.0f), B_gpu(N, 0.0f);
        for (size_t i = 0; i < N; ++i)
            A[i] = static_cast<float>(i) * 0.001f;   // произвольные числа

        // ─────────────────────────────────────────────────────────────────
        //  CPU
        // ─────────────────────────────────────────────────────────────────
        auto t0 = std::chrono::high_resolution_clock::now();
        vectorScaleCPU(A, B_cpu, k);
        auto t1 = std::chrono::high_resolution_clock::now();

        double cpu_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "CPU time : " << cpu_ms << " ms\n";

        // ─────────────────────────────────────────────────────────────────
        //  GPU / SYCL
        // ─────────────────────────────────────────────────────────────────
        {
            // Буферы: bufA — только чтение, bufB — только запись
            sycl::buffer<float, 1> bufA(A.data(),     sycl::range<1>(N));
            sycl::buffer<float, 1> bufB(B_gpu.data(), sycl::range<1>(N));

            auto t2 = std::chrono::high_resolution_clock::now();

            q.submit([&](sycl::handler& h) {
                // accessor'ы получаем внутри command group
                auto accA = bufA.get_access<sycl::access::mode::read>(h);
                auto accB = bufB.get_access<sycl::access::mode::write>(h);

                // Один Work-Item на один элемент
                h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
                    accB[idx] = accA[idx] * k;
                });
            });

            q.wait();   // ждём завершения ядра перед замером времени

            auto t3 = std::chrono::high_resolution_clock::now();
            double gpu_ms =
                std::chrono::duration<double, std::milli>(t3 - t2).count();
            std::cout << "GPU time : " << gpu_ms << " ms\n";
            std::cout << "Ускорение: " << cpu_ms / gpu_ms << "x\n\n";

        } // ← при выходе из блока bufB копирует данные обратно в B_gpu (RAII)

        // ─────────────────────────────────────────────────────────────────
        //  Проверка корректности
        // ─────────────────────────────────────────────────────────────────
        bool ok = true;
        for (size_t i = 0; i < N; ++i) {
            if (std::fabs(B_cpu[i] - B_gpu[i]) > 1e-4f) {
                ok = false;
                std::cerr << "Расхождение на индексе " << i
                          << ": CPU=" << B_cpu[i]
                          << "  GPU=" << B_gpu[i] << "\n";
                break;
            }
        }
        std::cout << "Результаты " << (ok ? "совпадают ✓" : "не совпадают ✗")
                  << "\n";

    } catch (const sycl::exception& e) {
        std::cerr << "SYCL exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
