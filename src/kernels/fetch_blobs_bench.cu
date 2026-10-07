// RAM-to-VRAM staging: bytewise parity with the old kernel, optional alternating graph timings.
#include "strata/kernels/verify_kernels.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}
__global__ void reference(const unsigned long long* __restrict__ src, const int* __restrict__ n,
                          uint4* __restrict__ dst, long long per) {
    const long long total = (long long) *n * per;
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (long long) gridDim.x * blockDim.x) {
        const long long k = i / per, off = i - k * per;
        dst[i] = ((const uint4*) src[k])[off];
    }
}
void launch(bool old, const unsigned long long* src, const int* n, uint8_t* dst, size_t bytes,
            int count, cudaStream_t stream) {
    if (old) reference<<<384, 256, 0, stream>>>(src, n, (uint4*) dst, (long long) (bytes / 16));
    else strata::kernels::fetch_blobs(src, n, dst, (int64_t) bytes, std::max(count, 1), stream);
    check(cudaGetLastError(), "copy launch");
}
}

int main(int argc, char** argv) {
    try {
        const bool bench = argc == 2 && std::strcmp(argv[1], "--bench") == 0;
        if (argc > 1 && !bench) throw std::invalid_argument("usage: fetch_blobs_bench [--bench]");
        check(cudaSetDevice(0), "select GPU 0");
        cudaDeviceProp device{};
        check(cudaGetDeviceProperties(&device, 0), "device properties");
        std::printf("device=%s; timings alternate old/new; mapped source pool=256 MiB\n", device.name);
        constexpr size_t host_bytes = 256ull << 20;
        constexpr size_t blob = (3ull << 20) + 16;
        constexpr size_t capacity = 16 * blob + 256;
        constexpr int repeats = 100, max_count = 385;
        uint8_t *host, *alias, *output[2];
        unsigned long long* pointers;
        int* number;
        cudaStream_t stream;
        check(cudaStreamCreate(&stream), "create stream");
        check(cudaHostAlloc(&host, host_bytes, cudaHostAllocMapped), "allocate mapped RAM");
        check(cudaHostGetDevicePointer(&alias, host, 0), "RAM device alias");
        for (size_t i = 0; i < host_bytes; ++i) host[i] = (uint8_t) (i * 131 + (i >> 8) + (i >> 19));
        for (auto& buffer : output) check(cudaMalloc(&buffer, capacity), "allocate destination");
        check(cudaMalloc(&pointers, (size_t) repeats * max_count * sizeof(*pointers)), "allocate pointer rows");
        check(cudaMalloc(&number, sizeof(*number)), "allocate expert count");
        struct Case { int count; size_t bytes; };
        const Case cases[] = {{0, 48}, {1, 16}, {3, 48}, {7, 1040}, {385, 48},
                              {1, blob}, {2, blob}, {3, blob}, {7, blob}, {16, blob}};
        for (const auto test : cases) {
            const size_t copied = (size_t) test.count * test.bytes;
            std::vector<unsigned long long> rows((size_t) repeats * std::max(test.count, 1));
            for (int r = 0; r < repeats; ++r) {
                const size_t offset = r == 0 ? 0 : (((size_t) r * (67ull << 20)) % (host_bytes - copied)) & ~size_t(15);
                for (int k = 0; k < test.count; ++k)
                    rows[(size_t) r * test.count + k] = (unsigned long long) (alias + offset + (size_t) k * test.bytes);
            }
            check(cudaMemcpyAsync(pointers, rows.data(), rows.size() * sizeof(*pointers), cudaMemcpyHostToDevice, stream), "upload pointers");
            check(cudaMemcpyAsync(number, &test.count, sizeof(test.count), cudaMemcpyHostToDevice, stream), "upload count");
            std::vector<uint8_t> got(copied + 256);
            for (int old = 0; old < 2; ++old) {
                check(cudaMemsetAsync(output[old], 0xa5, copied + 256, stream), "fill destination guard");
                launch(old != 0, pointers, number, output[old], test.bytes, test.count, stream);
                check(cudaMemcpyAsync(got.data(), output[old], got.size(), cudaMemcpyDeviceToHost, stream), "read copied bytes");
                check(cudaStreamSynchronize(stream), "parity synchronize");
                if (std::memcmp(got.data(), host, copied) != 0 ||
                    !std::all_of(got.begin() + copied, got.end(), [](uint8_t x) { return x == 0xa5; }))
                    throw std::runtime_error("copied bytes or destination guard differ, count=" + std::to_string(test.count));
            }
            std::printf("parity PASS count=%d bytes=%zu\n", test.count, test.bytes);
            if (!bench || test.bytes != blob) continue;
            cudaGraphExec_t executables[2];
            for (int old = 0; old < 2; ++old) {
                cudaGraph_t graph;
                check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "capture start");
                for (int r = 0; r < repeats; ++r)
                    launch(old != 0, pointers + (size_t) r * test.count, number, output[old], test.bytes, test.count, stream);
                check(cudaStreamEndCapture(stream, &graph), "capture end");
                check(cudaGraphInstantiate(&executables[old], graph, nullptr, nullptr, 0), "instantiate graph");
                check(cudaGraphDestroy(graph), "destroy graph");
                check(cudaGraphLaunch(executables[old], stream), "warm graph");
                check(cudaStreamSynchronize(stream), "warm synchronize");
            }
            cudaEvent_t start, end;
            check(cudaEventCreate(&start), "create start event");
            check(cudaEventCreate(&end), "create end event");
            std::vector<float> times[2];
            for (int round = 0; round < 6; ++round) for (int order = 0; order < 2; ++order) {
                const int old = (round + order) % 2;
                check(cudaEventRecord(start, stream), "record start");
                check(cudaGraphLaunch(executables[old], stream), "timed graph");
                check(cudaEventRecord(end, stream), "record end");
                check(cudaEventSynchronize(end), "timing synchronize");
                float ms;
                check(cudaEventElapsedTime(&ms, start, end), "event elapsed time");
                times[old].push_back(ms / repeats);
            }
            for (auto& values : times) std::sort(values.begin(), values.end());
            const double newer = (times[0][2] + times[0][3]) / 2, older = (times[1][2] + times[1][3]) / 2;
            std::printf("bench count=%d bytes=%zu old_ms=%.6f new_ms=%.6f old_GB_s=%.3f new_GB_s=%.3f gain_pct=%.2f\n",
                        test.count, test.bytes, older, newer, copied / older / 1e6, copied / newer / 1e6, (older / newer - 1) * 100);
            for (auto executable : executables) check(cudaGraphExecDestroy(executable), "destroy executable");
            check(cudaEventDestroy(start), "destroy start event");
            check(cudaEventDestroy(end), "destroy end event");
        }
        check(cudaFree(number), "free count");
        check(cudaFree(pointers), "free pointers");
        for (auto buffer : output) check(cudaFree(buffer), "free destination");
        check(cudaFreeHost(host), "free mapped RAM");
        check(cudaStreamDestroy(stream), "destroy stream");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "fetch_blobs_bench: %s\n", error.what());
        return 1;
    }
}
