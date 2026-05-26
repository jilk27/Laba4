#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>   // rand
#include <cmath>

// ──────────────────────────────────────────────
//  Параметры изображения
// ──────────────────────────────────────────────
static constexpr size_t WIDTH  = 1024;
static constexpr size_t HEIGHT = 1024;
static constexpr unsigned char THRESHOLD = 128;

// ──────────────────────────────────────────────
//  CPU-версия пороговой фильтрации
// ──────────────────────────────────────────────
void thresholdFilterCPU(const std::vector<unsigned char>& src,
                              std::vector<unsigned char>& dst,
                        size_t width, size_t height,
                        unsigned char T)
{
    for (size_t i = 0; i < width * height; ++i)
        dst[i] = (src[i] > T) ? 255u : 0u;
}

int main()
{
    try {
        // ── Выбор устройства ──────────────────────────────────────────────
        sycl::queue q(sycl::default_selector_v,
                      [](sycl::exception_list el) {
                          for (auto& e : el)
                              std::rethrow_exception(e);
                      });

        std::cout << "Устройство: "
                  << q.get_device().get_info<sycl::info::device::name>()
                  << "\n\n";

        // ── Инициализация изображения ─────────────────────────────────────
        const size_t NPIX = WIDTH * HEIGHT;

        std::vector<unsigned char> img(NPIX);
        std::vector<unsigned char> out_cpu(NPIX, 0);
        std::vector<unsigned char> out_gpu(NPIX, 0);

        std::srand(42);
        for (size_t i = 0; i < NPIX; ++i)
            img[i] = static_cast<unsigned char>(std::rand() % 256);

        // ─────────────────────────────────────────────────────────────────
        //  CPU
        // ─────────────────────────────────────────────────────────────────
        auto t0 = std::chrono::high_resolution_clock::now();
        thresholdFilterCPU(img, out_cpu, WIDTH, HEIGHT, THRESHOLD);
        auto t1 = std::chrono::high_resolution_clock::now();

        double cpu_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "CPU time : " << cpu_ms << " ms\n";

        // ─────────────────────────────────────────────────────────────────
        //  GPU / SYCL  (2D ND-range)
        // ─────────────────────────────────────────────────────────────────
        {
            // Хранение изображения — плоский 1D массив, индексация [row*W+col]
            sycl::buffer<unsigned char, 1> bufSrc(img.data(),     sycl::range<1>(NPIX));
            sycl::buffer<unsigned char, 1> bufDst(out_gpu.data(), sycl::range<1>(NPIX));

            auto t2 = std::chrono::high_resolution_clock::now();

            q.submit([&](sycl::handler& h) {
                auto accSrc = bufSrc.get_access<sycl::access::mode::read>(h);
                auto accDst = bufDst.get_access<sycl::access::mode::write>(h);

                const unsigned char T = THRESHOLD;
                const size_t W = WIDTH;

                // 2D ND-range: первое измерение — строки (height),
                //              второе — столбцы (width)
                h.parallel_for(sycl::range<2>(HEIGHT, WIDTH),
                               [=](sycl::id<2> idx) {
                    // idx[0] — row, idx[1] — col
                    size_t pos = idx[0] * W + idx[1];
                    accDst[pos] = (accSrc[pos] > T) ? 255u : 0u;
                });
            });

            q.wait();

            auto t3 = std::chrono::high_resolution_clock::now();
            double gpu_ms =
                std::chrono::duration<double, std::milli>(t3 - t2).count();

            std::cout << "GPU time : " << gpu_ms << " ms\n";
            std::cout << "Ускорение: " << cpu_ms / gpu_ms << "x\n\n";

        } // ← RAII: bufDst копирует out_gpu с устройства

        // ─────────────────────────────────────────────────────────────────
        //  Проверка корректности
        // ─────────────────────────────────────────────────────────────────
        bool ok = true;
        for (size_t i = 0; i < NPIX; ++i) {
            if (out_cpu[i] != out_gpu[i]) {
                ok = false;
                std::cerr << "Расхождение на пикселе " << i
                          << ": CPU=" << (int)out_cpu[i]
                          << "  GPU=" << (int)out_gpu[i] << "\n";
                break;
            }
        }
        std::cout << "Результаты " << (ok ? "совпадают ✓" : "не совпадают ✗")
                  << "\n";

        // ─────────────────────────────────────────────────────────────────
        //  Мини-статистика (для наглядности)
        // ─────────────────────────────────────────────────────────────────
        size_t white_pixels = 0;
        for (size_t i = 0; i < NPIX; ++i)
            if (out_cpu[i] == 255) ++white_pixels;

        std::cout << "Белых пикселей (>128): " << white_pixels
                  << " из " << NPIX
                  << " (" << 100.0 * white_pixels / NPIX << "%)\n";

    } catch (const sycl::exception& e) {
        std::cerr << "SYCL exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
