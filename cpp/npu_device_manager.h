#pragma once

#include "npu_protocol.h"
#include "npu_mem_manager.h"
#include <unordered_map>
#include <memory>
#include <mutex>
#include <iostream>

class NPUDeviceContext {
public:
    int device_id;
    std::unique_ptr<SharedMemoryHandler> shm;
    std::unique_ptr<NPUMemoryAllocator> allocator;
    std::mutex cmd_mutex; // For serializing commands to this device

    NPUDeviceContext(int id) : device_id(id) {
        // Connect to SHM
        // Note: is_server = false (Host is client)
        shm = std::make_unique<SharedMemoryHandler>(get_shm_name(id), SHM_SIZE, false);

        if (!shm->is_valid()) {
            std::cerr << "[NPUDeviceContext] Failed to connect to NPU Device " << id
                      << " (SHM: " << get_shm_name(id) << ")" << std::endl;
            throw std::runtime_error("Device Connection Failed");
        }

        // Initialize Allocator
        // Calculate start offset (skip control struct)
        void* base = shm->buffer;
        void* data_start = shm->get_data_start();
        uint64_t start_offset = static_cast<char*>(data_start) - static_cast<char*>(base);
        uint64_t available_size = SHM_SIZE - start_offset;

        allocator = std::make_unique<NPUMemoryAllocator>(available_size, start_offset);
        std::cout << "[NPUDeviceContext] Device " << id << " Connected. Allocator initialized." << std::endl;
    }

    NPUControl* get_control() {
        return shm->get_control();
    }

    void* get_base_ptr() {
        return shm->buffer;
    }
};

class NPUDeviceManager {
private:
    std::unordered_map<int, std::unique_ptr<NPUDeviceContext>> devices;
    std::mutex manager_mutex;

    NPUDeviceManager() {}

public:
    static NPUDeviceManager& get_instance() {
        static NPUDeviceManager instance;
        return instance;
    }

    void init_device(int device_id) {
        std::lock_guard<std::mutex> lock(manager_mutex);
        if (devices.find(device_id) == devices.end()) {
            try {
                devices[device_id] = std::make_unique<NPUDeviceContext>(device_id);
            } catch (const std::exception& e) {
                std::cerr << "[NPUDeviceManager] Error initializing device " << device_id << ": " << e.what() << std::endl;
            }
        }
    }

    NPUDeviceContext* get_device(int device_id) {
        std::lock_guard<std::mutex> lock(manager_mutex);
        auto it = devices.find(device_id);
        if (it != devices.end()) {
            return it->second.get();
        }

        // Auto-connect attempt
        try {
             devices[device_id] = std::make_unique<NPUDeviceContext>(device_id);
             return devices[device_id].get();
        } catch(...) {
             std::cerr << "[NPUDeviceManager] Device " << device_id << " not found and connection failed." << std::endl;
             return nullptr;
        }
    }

    // Thread-local current device
    static int& current_device_index() {
        static thread_local int current_idx = 0;
        return current_idx;
    }
};
