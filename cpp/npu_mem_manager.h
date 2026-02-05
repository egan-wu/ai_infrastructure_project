#pragma once

#include <vector>
#include <map>
#include <mutex>
#include <algorithm>
#include <iostream>
#include <stdexcept>

struct FreeBlock {
    uint64_t offset;
    size_t size;
};

class NPUMemoryAllocator {
private:
    std::vector<FreeBlock> free_blocks;
    std::map<uint64_t, size_t> allocations;
    std::mutex mutex;
    uint64_t total_size;
    uint64_t start_offset;

public:
    NPUMemoryAllocator(uint64_t size, uint64_t start) : total_size(size), start_offset(start) {
        // Initial free block covers the entire available space
        free_blocks.push_back({start, size});
    }

    uint64_t allocate(size_t size) {
        std::lock_guard<std::mutex> lock(mutex);

        // Alignment (64 bytes)
        size_t aligned_size = size;
        if (aligned_size % 64 != 0) aligned_size += 64 - (aligned_size % 64);

        // First-fit strategy
        for (auto it = free_blocks.begin(); it != free_blocks.end(); ++it) {
            if (it->size >= aligned_size) {
                uint64_t offset = it->offset;

                if (it->size > aligned_size) {
                    // Split the block
                    it->offset += aligned_size;
                    it->size -= aligned_size;
                } else {
                    // Exact fit, remove block
                    free_blocks.erase(it);
                }

                allocations[offset] = aligned_size;
                return offset;
            }
        }

        throw std::runtime_error("NPU Out of Memory: Failed to allocate " + std::to_string(size) + " bytes.");
    }

    void free(uint64_t offset) {
        std::lock_guard<std::mutex> lock(mutex);

        auto alloc_it = allocations.find(offset);
        if (alloc_it == allocations.end()) {
            std::cerr << "[NPUMemoryAllocator] Warning: Double free or invalid pointer: " << offset << std::endl;
            return;
        }

        size_t size = alloc_it->second;
        allocations.erase(alloc_it);

        // Insert into free list sorted by offset
        auto it = free_blocks.begin();
        while (it != free_blocks.end() && it->offset < offset) {
            ++it;
        }

        // Insert the new free block
        it = free_blocks.insert(it, {offset, size});

        // Coalesce with next block if adjacent
        auto next = it;
        ++next;
        if (next != free_blocks.end() && (it->offset + it->size == next->offset)) {
            it->size += next->size;
            free_blocks.erase(next);
        }

        // Coalesce with prev block if adjacent
        if (it != free_blocks.begin()) {
            auto prev = it;
            --prev;
            if (prev->offset + prev->size == it->offset) {
                prev->size += it->size;
                free_blocks.erase(it);
            }
        }
    }

    // Debug helper
    void print_state() {
        std::lock_guard<std::mutex> lock(mutex);
        std::cout << "[Allocator State] Allocations: " << allocations.size()
                  << ", Free Blocks: " << free_blocks.size() << std::endl;
        for (const auto& block : free_blocks) {
            std::cout << "  Free: Offset " << block.offset << ", Size " << block.size << std::endl;
        }
    }
};
