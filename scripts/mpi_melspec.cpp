// Experimental multi-rank driver. This is NOT src/main.cpp and is not a final
// submission entry. The protected CLI and validator remain unchanged.
#include "melspec.h"

#include <mpi.h>
#include <sched.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
using I64 = long long;

struct Args {
    std::string input, output;
    int sr = 44100, n_fft = 2048, hop = 512, n_mels = 128;
    double f_min = 0.0, f_max = 22050.0;
};

Args parse_args(int argc, char** argv) {
    if (argc < 3) throw std::runtime_error("usage: mpi-melspec input output [official options]");
    Args a{argv[1], argv[2]};
    for (int i = 3; i < argc; i += 2) {
        if (i + 1 >= argc) throw std::runtime_error("missing option value");
        const std::string key(argv[i]), value(argv[i + 1]);
        if (key == "--sr") a.sr = std::stoi(value);
        else if (key == "--n_fft") a.n_fft = std::stoi(value);
        else if (key == "--hop_length") a.hop = std::stoi(value);
        else if (key == "--n_mels") a.n_mels = std::stoi(value);
        else if (key == "--f_min") a.f_min = std::stod(value);
        else if (key == "--f_max") a.f_max = std::stod(value);
        else throw std::runtime_error("unknown option: " + key);
    }
    if (a.n_fft < 1 || (a.n_fft & (a.n_fft - 1)) || a.hop < 1 ||
        a.n_mels < 1 || a.sr < 1)
        throw std::runtime_error("invalid spectrogram parameters");
    return a;
}

I64 checked_samples(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open input: " + path);
    const auto bytes = f.tellg();
    if (bytes < 0 || bytes % 2 != 0 || bytes / 2 > std::numeric_limits<int>::max())
        throw std::runtime_error("invalid or oversized PCM input");
    return static_cast<I64>(bytes / 2);
}

void read_pcm(const std::string& path, I64 first, I64 count, std::vector<double>& y) {
    if (first < 0 || count < 0 || first + count > checked_samples(path))
        throw std::runtime_error("PCM slice out of bounds");
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open input: " + path);
    std::vector<int16_t> pcm(static_cast<size_t>(count));
    f.seekg(static_cast<std::streamoff>(first * 2));
    if (count && !f.read(reinterpret_cast<char*>(pcm.data()),
                         static_cast<std::streamsize>(count * 2)))
        throw std::runtime_error("failed to read PCM slice");
    y.resize(pcm.size());
    for (size_t i = 0; i < pcm.size(); ++i) y[i] = pcm[i] / 32768.0;
}

void write_float(const std::string& path, const std::vector<float>& output) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("cannot open output: " + path);
    if (!output.empty())
        f.write(reinterpret_cast<const char*>(output.data()),
                static_cast<std::streamsize>(output.size() * sizeof(float)));
    if (!f) throw std::runtime_error("failed to write output");
}

struct Partition {
    I64 begin, end;
    I64 count() const { return end - begin; }
};

Partition partition(I64 frames, int size, int rank) {
    return {frames * rank / size, frames * (rank + 1) / size};
}

std::string io_mode() {
    const char* mode = std::getenv("MELSPEC_MPI_IO");
    return mode ? mode : "bcast";
}

double seconds(Clock::time_point since) {
    return std::chrono::duration<double>(Clock::now() - since).count();
}

int run(int argc, char** argv) {
    const auto process_start = Clock::now();
    MPI_Init(&argc, &argv);
    const double init_s = seconds(process_start);
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    try {
        const Args a = parse_args(argc, argv);
        std::string mode = rank == 0 ? io_mode() : std::string();
        int mode_length = static_cast<int>(mode.size());
        MPI_Bcast(&mode_length, 1, MPI_INT, 0, MPI_COMM_WORLD);
        mode.resize(mode_length);
        MPI_Bcast(mode.data(), mode_length, MPI_CHAR, 0, MPI_COMM_WORLD);
        if (mode != "bcast" && mode != "scatter" && mode != "shared")
            throw std::runtime_error("MELSPEC_MPI_IO must be bcast, scatter or shared");
        I64 samples = 0;
        const auto read_start = Clock::now();
        std::vector<double> full, local;
        if (rank == 0 && mode != "shared") {
            samples = checked_samples(a.input);
            read_pcm(a.input, 0, samples, full);
        } else if (mode == "shared") {
            samples = checked_samples(a.input);
        }
        MPI_Bcast(&samples, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
        if (samples < a.n_fft) throw std::runtime_error("audio too short for n_fft");
        const I64 frames = 1 + (samples - a.n_fft) / a.hop;
        const Partition part = partition(frames, size, rank);
        const I64 count = part.count();
        const I64 first_sample = count ? part.begin * a.hop : 0;
        const I64 local_samples = count ? (count - 1) * a.hop + a.n_fft : 0;
        if (count && first_sample + local_samples > samples)
            throw std::runtime_error("internal frame partition overflow");
        const double read_s = seconds(read_start);

        const auto distribution_start = Clock::now();
        if (mode == "shared") {
            if (count) read_pcm(a.input, first_sample, local_samples, local);
        } else if (mode == "bcast") {
            if (rank != 0) full.resize(static_cast<size_t>(samples));
            MPI_Bcast(full.data(), samples, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            if (count)
                local.assign(full.begin() + first_sample,
                             full.begin() + first_sample + local_samples);
        } else {
            if (rank == 0) {
                for (int r = 0; r < size; ++r) {
                    const Partition p = partition(frames, size, r);
                    if (!p.count()) continue;
                    const I64 first = p.begin * a.hop;
                    const I64 length = (p.count() - 1) * a.hop + a.n_fft;
                    if (r == 0) {
                        local.assign(full.begin() + first, full.begin() + first + length);
                    } else {
                        MPI_Send(full.data() + first, length, MPI_DOUBLE, r, 100, MPI_COMM_WORLD);
                    }
                }
            } else if (count) {
                local.resize(static_cast<size_t>(local_samples));
                MPI_Recv(local.data(), local_samples, MPI_DOUBLE, 0, 100,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        const double distribution_s = seconds(distribution_start);

        const auto compute_start = Clock::now();
        std::vector<double> result;
        if (count)
            compute_melspectrogram(local, a.sr, a.n_fft, a.hop, a.n_mels,
                                   a.f_min, a.f_max, result);
        const double compute_s = seconds(compute_start);

        const auto convert_start = Clock::now();
        // The receive datatype consumes one global frame column at a time.
        // Pack from [mel][local_frame] to [local_frame][mel], preserving the
        // same double-to-float conversion as the serial final write.
        std::vector<float> packed(result.size());
        for (I64 t = 0; t < count; ++t)
            for (int m = 0; m < a.n_mels; ++m)
                packed[t * a.n_mels + m] =
                    static_cast<float>(result[m * count + t]);
        const double convert_pack_s = seconds(convert_start);

        const auto gather_start = Clock::now();
        std::vector<float> gathered;
        std::vector<int> recvcounts, displs;
        MPI_Datatype column_type = MPI_FLOAT;
        if (rank == 0) {
            gathered.resize(static_cast<size_t>(a.n_mels) * frames);
            for (int r = 0; r < size; ++r) {
                const Partition p = partition(frames, size, r);
                recvcounts.push_back(static_cast<int>(p.count()));
                displs.push_back(static_cast<int>(p.begin));
            }
            // Gatherv has ONE receive datatype. Each scalar advances one
            // global frame, so counts/displacements are frame indices.
            MPI_Datatype vector_type;
            MPI_Type_vector(a.n_mels, 1, frames, MPI_FLOAT, &vector_type);
            MPI_Type_create_resized(vector_type, 0, sizeof(float), &column_type);
            MPI_Type_commit(&column_type);
            MPI_Type_free(&vector_type);
        }
        MPI_Gatherv(packed.data(), static_cast<int>(packed.size()), MPI_FLOAT,
                    rank == 0 ? gathered.data() : nullptr,
                    rank == 0 ? recvcounts.data() : nullptr,
                    rank == 0 ? displs.data() : nullptr,
                    column_type, 0, MPI_COMM_WORLD);
        if (rank == 0) MPI_Type_free(&column_type);
        const double gather_s = seconds(gather_start);

        const auto write_start = Clock::now();
        if (rank == 0) write_float(a.output, gathered);
        const double write_s = seconds(write_start);
        const auto sync_start = Clock::now();
        MPI_Barrier(MPI_COMM_WORLD);
        const double sync_s = seconds(sync_start);
        char host[MPI_MAX_PROCESSOR_NAME];
        int host_length = 0;
        MPI_Get_processor_name(host, &host_length);
        host[host_length] = '\0';
        const auto finalize_start = Clock::now();
        MPI_Finalize();
        const double finalize_s = seconds(finalize_start);
        const double wall_s = seconds(process_start);
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        const int cpus = sched_getaffinity(0, sizeof(cpuset), &cpuset) == 0 ? CPU_COUNT(&cpuset) : -1;
        std::printf("MPI_TIMING rank=%d size=%d host=%s cpus=%d frames=%lld local_frames=%lld io=%s "
                    "init_s=%.9f read_s=%.9f distribution_s=%.9f compute_s=%.9f "
                    "convert_pack_s=%.9f gather_s=%.9f write_s=%.9f sync_s=%.9f "
                    "finalize_s=%.9f process_s=%.9f\n",
                    rank, size, host, cpus, frames, count, mode.c_str(), init_s, read_s,
                    distribution_s, compute_s, convert_pack_s, gather_s, write_s,
                    sync_s, finalize_s, wall_s);
        std::fflush(stdout);
        return 0;
    } catch (const std::exception& e) {
        if (rank == 0) std::cerr << "MPI driver error: " << e.what() << "\n";
        std::fflush(stderr);
        MPI_Abort(MPI_COMM_WORLD, 2);
        return 2;
    }
}
}  // namespace

int main(int argc, char** argv) {
    return run(argc, argv);
}
