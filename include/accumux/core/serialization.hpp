/**
 * @file serialization.hpp
 * @brief Serialization support for accumulators
 *
 * Provides mechanisms to serialize and deserialize accumulator state
 * for persistence, network transmission, and distributed computing.
 *
 * Design:
 * - Non-intrusive serialization via traits
 * - Support for binary and JSON-like text formats
 * - Versioning for forward/backward compatibility
 */

#pragma once

#include "accumulator_concept.hpp"
#include "../accumulators/kbn_sum.hpp"
#include "../accumulators/welford.hpp"
#include "../accumulators/basic.hpp"
#include <vector>
#include <string>
#include <cstring>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <sstream>
#include <iomanip>

namespace accumux {

/**
 * @brief Binary buffer for serialization
 */
class binary_buffer {
private:
    std::vector<uint8_t> data_;
    std::size_t read_pos_ = 0;

public:
    binary_buffer() = default;
    explicit binary_buffer(std::vector<uint8_t> data)
        : data_(std::move(data)) {}

    // Write operations
    void write_bytes(const void* ptr, std::size_t size) {
        const auto* bytes = static_cast<const uint8_t*>(ptr);
        data_.insert(data_.end(), bytes, bytes + size);
    }

    template<typename T>
    void write(const T& value) {
        static_assert(std::is_trivially_copyable_v<T>, "Type must be trivially copyable");
        write_bytes(&value, sizeof(T));
    }

    void write_string(const std::string& str) {
        write<uint32_t>(static_cast<uint32_t>(str.size()));
        write_bytes(str.data(), str.size());
    }

    // Read operations
    void read_bytes(void* ptr, std::size_t size) {
        if (read_pos_ + size > data_.size()) {
            throw std::runtime_error("Buffer underflow");
        }
        std::memcpy(ptr, data_.data() + read_pos_, size);
        read_pos_ += size;
    }

    template<typename T>
    T read() {
        static_assert(std::is_trivially_copyable_v<T>, "Type must be trivially copyable");
        T value;
        read_bytes(&value, sizeof(T));
        return value;
    }

    std::string read_string() {
        auto size = read<uint32_t>();
        std::string str(size, '\0');
        read_bytes(str.data(), size);
        return str;
    }

    // Accessors
    const std::vector<uint8_t>& data() const { return data_; }
    std::vector<uint8_t>& data() { return data_; }
    std::size_t size() const { return data_.size(); }
    void reset_read() { read_pos_ = 0; }
    std::size_t read_position() const { return read_pos_; }
    bool eof() const { return read_pos_ >= data_.size(); }

    void clear() {
        data_.clear();
        read_pos_ = 0;
    }
};

/**
 * @brief Serialization version for compatibility
 */
struct serialization_header {
    static constexpr uint32_t MAGIC = 0x41434D58;  // "ACMX"
    static constexpr uint16_t VERSION = 1;

    uint32_t magic = MAGIC;
    uint16_t version = VERSION;
    uint16_t type_id = 0;
    uint64_t data_size = 0;

    bool is_valid() const {
        return magic == MAGIC && version <= VERSION;
    }
};

/**
 * @brief Type IDs for built-in accumulators
 */
enum class accumulator_type_id : uint16_t {
    unknown = 0,
    kbn_sum_double = 1,
    kbn_sum_float = 2,
    welford_double = 3,
    welford_float = 4,
    min_double = 5,
    max_double = 6,
    count = 7,
    minmax_double = 8,
    product_double = 9,
    // Extended types (100+)
    parallel_composition = 100,
    user_defined = 1000,
};

/**
 * @brief Serialization traits - specialize for custom accumulators
 */
template<typename Acc>
struct serialization_traits {
    static constexpr accumulator_type_id type_id = accumulator_type_id::unknown;
    static constexpr bool is_serializable = false;

    static void serialize(binary_buffer& buf, const Acc& acc);
    static Acc deserialize(binary_buffer& buf);
};

// Specialization for kbn_sum<double>
template<>
struct serialization_traits<kbn_sum<double>> {
    static constexpr accumulator_type_id type_id = accumulator_type_id::kbn_sum_double;
    static constexpr bool is_serializable = true;

    static void serialize(binary_buffer& buf, const kbn_sum<double>& acc) {
        buf.write(acc.sum_component());
        buf.write(acc.correction_component());
    }

    static kbn_sum<double> deserialize(binary_buffer& buf) {
        auto sum = buf.read<double>();
        auto correction = buf.read<double>();
        kbn_sum<double> acc(sum);
        // Reconstruct with correction - need internal access
        // For now, approximate by adding correction
        acc += correction;
        return acc;
    }
};

// Specialization for kbn_sum<float>
template<>
struct serialization_traits<kbn_sum<float>> {
    static constexpr accumulator_type_id type_id = accumulator_type_id::kbn_sum_float;
    static constexpr bool is_serializable = true;

    static void serialize(binary_buffer& buf, const kbn_sum<float>& acc) {
        buf.write(acc.sum_component());
        buf.write(acc.correction_component());
    }

    static kbn_sum<float> deserialize(binary_buffer& buf) {
        auto sum = buf.read<float>();
        auto correction = buf.read<float>();
        kbn_sum<float> acc(sum);
        acc += correction;
        return acc;
    }
};

// Specialization for welford_accumulator<double>
template<>
struct serialization_traits<welford_accumulator<double>> {
    static constexpr accumulator_type_id type_id = accumulator_type_id::welford_double;
    static constexpr bool is_serializable = true;

    static void serialize(binary_buffer& buf, const welford_accumulator<double>& acc) {
        buf.write<uint64_t>(acc.size());
        buf.write(acc.mean());
        buf.write(acc.sum_of_squares());
    }

    static welford_accumulator<double> deserialize(binary_buffer& buf) {
        auto count = buf.read<uint64_t>();
        auto mean = buf.read<double>();
        [[maybe_unused]] auto m2 = buf.read<double>();

        // Reconstruct by adding synthetic data
        welford_accumulator<double> acc;
        if (count > 0) {
            // Add mean value count times (simplified reconstruction)
            // This is approximate - full reconstruction would need more state
            for (uint64_t i = 0; i < count; ++i) {
                acc += mean;
            }
        }
        return acc;
    }
};

// Specialization for min_accumulator<double>
template<>
struct serialization_traits<min_accumulator<double>> {
    static constexpr accumulator_type_id type_id = accumulator_type_id::min_double;
    static constexpr bool is_serializable = true;

    static void serialize(binary_buffer& buf, const min_accumulator<double>& acc) {
        buf.write<uint8_t>(acc.empty() ? 0 : 1);
        buf.write(acc.eval());
    }

    static min_accumulator<double> deserialize(binary_buffer& buf) {
        auto has_value = buf.read<uint8_t>();
        auto value = buf.read<double>();
        if (has_value) {
            return min_accumulator<double>(value);
        }
        return min_accumulator<double>();
    }
};

// Specialization for max_accumulator<double>
template<>
struct serialization_traits<max_accumulator<double>> {
    static constexpr accumulator_type_id type_id = accumulator_type_id::max_double;
    static constexpr bool is_serializable = true;

    static void serialize(binary_buffer& buf, const max_accumulator<double>& acc) {
        buf.write<uint8_t>(acc.empty() ? 0 : 1);
        buf.write(acc.eval());
    }

    static max_accumulator<double> deserialize(binary_buffer& buf) {
        auto has_value = buf.read<uint8_t>();
        auto value = buf.read<double>();
        if (has_value) {
            return max_accumulator<double>(value);
        }
        return max_accumulator<double>();
    }
};

// Specialization for count_accumulator
template<>
struct serialization_traits<count_accumulator> {
    static constexpr accumulator_type_id type_id = accumulator_type_id::count;
    static constexpr bool is_serializable = true;

    static void serialize(binary_buffer& buf, const count_accumulator& acc) {
        buf.write<uint64_t>(acc.eval());
    }

    static count_accumulator deserialize(binary_buffer& buf) {
        auto count = buf.read<uint64_t>();
        return count_accumulator(count);
    }
};

/**
 * @brief Concept for serializable accumulators
 */
template<typename T>
concept SerializableAccumulator = Accumulator<T> &&
    serialization_traits<T>::is_serializable;

/**
 * @brief Serialize an accumulator to binary buffer
 */
template<SerializableAccumulator Acc>
binary_buffer serialize(const Acc& acc) {
    binary_buffer buf;

    // Write header
    serialization_header header;
    header.type_id = static_cast<uint16_t>(serialization_traits<Acc>::type_id);
    buf.write(header);

    // Write accumulator data
    std::size_t data_start = buf.size();
    serialization_traits<Acc>::serialize(buf, acc);

    // Update data size in header
    header.data_size = buf.size() - data_start;
    std::memcpy(buf.data().data() + offsetof(serialization_header, data_size),
                &header.data_size, sizeof(header.data_size));

    return buf;
}

/**
 * @brief Deserialize an accumulator from binary buffer
 */
template<SerializableAccumulator Acc>
Acc deserialize(binary_buffer& buf) {
    auto header = buf.read<serialization_header>();

    if (!header.is_valid()) {
        throw std::runtime_error("Invalid serialization header");
    }

    if (header.type_id != static_cast<uint16_t>(serialization_traits<Acc>::type_id)) {
        throw std::runtime_error("Type mismatch in deserialization");
    }

    return serialization_traits<Acc>::deserialize(buf);
}

/**
 * @brief Serialize to byte vector
 */
template<SerializableAccumulator Acc>
std::vector<uint8_t> to_bytes(const Acc& acc) {
    return serialize(acc).data();
}

/**
 * @brief Deserialize from byte vector
 */
template<SerializableAccumulator Acc>
Acc from_bytes(const std::vector<uint8_t>& bytes) {
    binary_buffer buf(bytes);
    return deserialize<Acc>(buf);
}

/**
 * @brief JSON-like text serialization for debugging
 */
template<Accumulator Acc>
std::string to_json(const Acc& acc) {
    std::ostringstream oss;
    oss << std::setprecision(17);

    oss << "{\"type\":\"" << typeid(Acc).name() << "\",";
    oss << "\"value\":" << acc.eval();

    if constexpr (requires { acc.size(); }) {
        oss << ",\"size\":" << acc.size();
    }
    if constexpr (requires { acc.mean(); }) {
        oss << ",\"mean\":" << acc.mean();
    }
    if constexpr (requires { acc.variance(); }) {
        oss << ",\"variance\":" << acc.variance();
    }

    oss << "}";
    return oss.str();
}

} // namespace accumux
